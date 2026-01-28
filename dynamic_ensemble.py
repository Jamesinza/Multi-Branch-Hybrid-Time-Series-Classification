import gc
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
from scipy.stats import norm, pearsonr
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D, 
                                     Dropout, Add, LayerNormalization, MultiHeadAttention, Conv1D,
                                     Concatenate)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from keras_hub.layers import TransformerEncoder, SinePositionEncoding
from hmmlearn.hmm import GaussianHMM, CategoricalHMM, GMMHMM, PoissonHMM

# Enable mixed precision for speed gains
mixed_precision.set_global_policy('mixed_float16')

###############################################################################
# TIME SERIES DATA AUGMENTATION & PREPROCESSING FUNCTIONS
###############################################################################

def augment_time_series(series, noise_std=0.01, warp_factor=0.05):
    """
    Apply simple time series augmentations:
      - Add Gaussian noise.
      - Slightly warp the time axis via linear interpolation.
    """
    # Noise injection
    noisy = series + np.random.normal(0, noise_std, size=series.shape)
    
    # Time warping: create a new time axis slightly perturbed
    orig_idx = np.arange(len(noisy))
    warp = np.interp(orig_idx, 
                     orig_idx + np.random.uniform(-warp_factor, warp_factor, size=series.shape),
                     noisy)
    return warp.astype(np.float32)

def natural_visibility_graph(window):
    """
    Convert a 1D time series window into a visibility graph.
    Two nodes i and j are connected if for each intermediate k:
      window[k] < window[i] + (window[j]-window[i])*(k-i)/(j-i)
    """
    n = len(window)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            visible = True
            delta = window[j] - window[i]
            for k in range(i+1, j):
                if window[k] >= window[i] + delta * (k - i) / (j - i):
                    visible = False
                    break
            if visible:
                G.add_edge(i, j)
    return G

def compute_visibility_features(window):
    """
    Extract two features from the natural visibility graph:
      - Average node degree.
      - Clustering coefficient.
    """
    window = np.array(window).flatten()
    G = natural_visibility_graph(window)
    if len(G) == 0:
        return np.array([0, 0], dtype=np.float32)
    avg_degree = np.mean([d for _, d in G.degree()])
    clustering = nx.average_clustering(G)
    return np.array([avg_degree, clustering], dtype=np.float32)

def get_extra_features(X_raw, rng):
    """
    Augment the raw input with engineered features based on:
      - HMMs (Gaussian, Categorical, GMM, Poisson)
      - Natural Visibility Graph (NVG) features.
    """
    # Work on copies for different HMMs
    hs1 = X_raw.copy()
    hs2 = X_raw.copy()
    hs3 = X_raw.copy()
    hs4 = X_raw.copy()
    base_features = X_raw.copy()

    # HMM-based augmentation (one iteration sufficient, extendable if needed)
    for i in range(1):
        hmm_g = GaussianHMM(n_components=10 - i, covariance_type="full", random_state=rng)
        hmm_c = CategoricalHMM(n_components=10 - i, n_features=10, random_state=rng)
        hmm_g.fit(hs1)
        hmm_c.fit(hs2)
        hs1_pred = hmm_g.predict(hs1).reshape(-1, 1).astype(np.float32)
        hs2_pred = hmm_c.predict(hs2).reshape(-1, 1).astype(np.float32)
        base_features = np.hstack([base_features, hs1_pred, hs2_pred])
        hs1, hs2 = hs1_pred, hs2_pred

    # GMMHMM feature
    hmm_gmm = GMMHMM(n_components=10, n_mix=1, covariance_type="full", random_state=rng)
    hmm_gmm.fit(hs3)
    hs3_pred = hmm_gmm.predict(hs3).reshape(-1, 1).astype(np.float32)
    base_features = np.hstack([base_features, hs3_pred])

    # PoissonHMM feature
    hmm_pois = PoissonHMM(n_components=10, random_state=rng)
    hmm_pois.fit(hs4)
    hs4_pred = hmm_pois.predict(hs4).reshape(-1, 1).astype(np.float32)
    base_features = np.hstack([base_features, hs4_pred])
    
    # NVG-based features computed in sliding windows
    win_size = 10
    nvg_features = []
    for i in range(len(X_raw) - win_size):
        feat = compute_visibility_features(X_raw[i:i+win_size])
        nvg_features.append(feat)
    nvg_features = np.array(nvg_features, dtype=np.float32)
    
    base_features = base_features[win_size:]
    X_augmented = np.hstack([base_features, nvg_features])
    return X_augmented

def get_real_data(num_samples):
    """Load and preprocess real data from CSV."""
    print('\nBuilding dataframe using real data...')
    dataset = 'Take5'
    df = pd.read_csv(f'datasets/{dataset}_Full.csv')
    cols = ['A', 'B', 'C', 'D', 'E']
    df = df[cols].dropna().astype(np.int8)
    df = df.map(lambda x: f'{x:02d}')
    flattened = df.values.flatten()
    full_data = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int8)
    return full_data[:num_samples]

###############################################################################
# DATASET PIPELINE: DUAL BRANCH (ENGINEERED + RAW)
###############################################################################

def create_dual_dataset(raw_series, engineered, window_length, batch_size, shuffle=False):
    """
    Create a tf.data.Dataset that yields batches of:
       ({'eng_input': engineered_window, 'raw_input': raw_window}, target)
    where:
      - engineered: output of get_extra_features (2D array)
      - raw_series: original series (2D array with one feature)
    The target is taken as the next time step (classification target).
    """
    # Ensure alignment: engineered series is shorter (due to sliding window in NVG).
    n_samples = engineered.shape[0] - window_length
    eng_windows = []
    raw_windows = []
    targets = []
    for i in range(n_samples):
        eng_windows.append(engineered[i:i+window_length])
        raw_windows.append(raw_series[i:i+window_length])
        # Target: use the time step immediately after the window (from raw series)
        targets.append(raw_series[i+window_length, 0])
    eng_windows = np.array(eng_windows, dtype=np.float32)
    raw_windows = np.array(raw_windows, dtype=np.float32)
    targets = np.array(targets, dtype=np.int32)
    
    ds = tf.data.Dataset.from_tensor_slices(({'eng_input': eng_windows, 
                                               'raw_input': raw_windows}, targets))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).cache().repeat().prefetch(tf.data.AUTOTUNE)
    return ds

###############################################################################
# CUSTOM TIME2VEC LAYER FOR RELATIVE POSITIONAL ENCODING
###############################################################################

class Time2Vec(tf.keras.layers.Layer):
    """
    Time2Vec layer: Represents each time step as a linear component and periodic features.
    See “Time2Vec: Learning a Vector Representation of Time” for further details.
    """
    def __init__(self, kernel_size=1, **kwargs):
        super(Time2Vec, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # Bias and weight for the linear part.
        self.w0 = self.add_weight(name="w0", shape=(1,), initializer="uniform", trainable=True)
        self.b0 = self.add_weight(name="b0", shape=(1,), initializer="uniform", trainable=True)
        # Weights and bias for periodic features.
        self.w = self.add_weight(name="w", shape=(int(input_shape[-1]), self.kernel_size), initializer="uniform", trainable=True)
        self.b = self.add_weight(name="b", shape=(self.kernel_size,), initializer="uniform", trainable=True)
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs):
        # Linear term
        linear = self.w0 * inputs + self.b0
        # Periodic term
        periodic = tf.math.sin(tf.tensordot(inputs, self.w, axes=1) + self.b)
        return tf.concat([linear, periodic], axis=-1)

###############################################################################
# HYBRID MODEL ARCHITECTURE: DUAL-BRANCH (ENGINEERED + RAW)
###############################################################################

def create_dual_branch_model(seed, window_length, eng_feature_dim, num_classes=10):
    """
    Create a dual-branch model that processes:
      1. Engineered features using Transformer-inspired blocks with Time2Vec positional encoding.
      2. Raw time series using 1D CNN layers.
    Their outputs are concatenated for final classification.
    """
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Engineered branch input
    eng_input = Input(shape=(window_length, eng_feature_dim), name='eng_input')
    # Project input and add Time2Vec encoding (relative positional)
    proj_eng = Dense(128)(eng_input)
    pos_enc = Time2Vec(kernel_size=32)(proj_eng)
    x_eng = Add()([proj_eng, pos_enc])
    # Transformer-inspired block(s)
    num_heads = 4
    key_dim = 128 // num_heads
    for _ in range(2):
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x_eng, x_eng)
        attn = Dropout(0.2)(attn)
        x_eng = LayerNormalization()(Add()([x_eng, attn]))
        ffn = Dense(512, activation='gelu')(x_eng)
        ffn = Dense(128, activation='gelu')(ffn)
        ffn = Dropout(0.2)(ffn)
        x_eng = LayerNormalization()(Add()([x_eng, ffn]))
    eng_out = GlobalAveragePooling1D()(x_eng)
    
    # Raw branch input
    raw_input = Input(shape=(window_length, 1), name='raw_input')
    # 1D CNN block to extract local patterns
    x_raw = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(raw_input)
    x_raw = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x_raw)
    x_raw = GlobalAveragePooling1D()(x_raw)
    
    # Concatenate both branches
    concat = Concatenate()([eng_out, x_raw])
    dense = Dense(128, activation='gelu')(concat)
    dense = Dropout(0.3)(dense)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(dense)
    
    model = Model(inputs=[eng_input, raw_input], outputs=outputs)
    
    # Use AdamW with a cosine decay schedule (cyclical learning rate style)
    lr_schedule = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=1000)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  jit_compile=True)
    
    print(f"Created dual-branch model with seed {seed}")
    return model

###############################################################################
# ENSEMBLE & METRICS FUNCTIONS (UNCHANGED)
###############################################################################

def ensemble_predict_weighted(models, weights, dataset):
    preds = [w * model.predict(dataset, verbose=0) for model, w in zip(models, weights)]
    return np.sum(np.array(preds), axis=0)

def get_individual_predictions(models, dataset):
    return [np.argmax(model.predict(dataset, verbose=0), axis=1) for model in models]

def compute_error_correlations(preds, y_true):
    errors = np.array([(pred != y_true).astype(np.float32) for pred in preds])
    corr = np.corrcoef(errors)
    return np.nan_to_num(corr)

def compute_q_statistic_matrix(preds, y_true):
    n_models = len(preds)
    correctness = [pred == y_true for pred in preds]
    q_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            N11 = np.sum(np.logical_and(correctness[i], correctness[j]))
            N00 = np.sum(np.logical_and(~correctness[i], ~correctness[j]))
            N10 = np.sum(np.logical_and(correctness[i], ~correctness[j]))
            N01 = np.sum(np.logical_and(~correctness[i], correctness[j]))
            denom = (N11 * N00 + N10 * N01)
            q_matrix[i, j] = (N11 * N00 - N10 * N01)/denom if denom != 0 else 0.0
    return q_matrix

def compute_double_fault_matrix(preds, y_true):
    n_models = len(preds)
    correctness = [pred == y_true for pred in preds]
    df_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            N00 = np.sum(np.logical_and(~correctness[i], ~correctness[j]))
            N10 = np.sum(np.logical_and(correctness[i], ~correctness[j]))
            N01 = np.sum(np.logical_and(~correctness[i], correctness[j]))
            denom = (N00 + N10 + N01)
            df_matrix[i, j] = N00 / denom if denom != 0 else 0.0
    return df_matrix

def select_diverse_models(q_matrix, val_accuracies, threshold=0.3):
    sorted_indices = np.argsort(val_accuracies)[::-1]
    selected = []
    for idx in sorted_indices:
        if all(q_matrix[idx, sel] <= threshold for sel in selected):
            selected.append(idx)
    return selected

def recalc_q_matrix(selected_indices, full_q_matrix):
    return full_q_matrix[np.ix_(selected_indices, selected_indices)]

###############################################################################
# MAIN ENSEMBLE TRAINING & EVALUATION LOOP
###############################################################################

# Hyperparameters
batch_size = 1024
epochs = 100
window_length = 10
num_samples = 110_000
sub_samples = 90_000
n_models = 10

# Data & Scaling Setup
scaler = StandardScaler()
data = get_real_data(num_samples)
print(f'\nData shape: {data.shape}')
val_test_data = data[-10_000:]

# Augment and prepare dual inputs
# (Augmentation can be applied on the raw series for extra diversity)
augmented_raw = augment_time_series(data)
raw_series = augmented_raw.reshape(-1, 1).astype(np.float32)
# Engineered features via HMM+NVG (note: get_extra_features expects 2D input)
engineered_features = get_extra_features(raw_series, rng=42)
# For the raw branch, we trim the series to match engineered_features length.
raw_aligned = raw_series[len(raw_series) - engineered_features.shape[0]:]

# Create dataset splits for training/validation/test.
split = 10_000
train_eng = engineered_features[:-split]
train_raw = raw_aligned[:-split]
val_eng = engineered_features[-split:-split//2]
val_raw = raw_aligned[-split:-split//2]
test_eng = engineered_features[-split//2:]
test_raw = raw_aligned[-split//2:]

# Scale engineered features only (raw input is kept as is)
train_eng = scaler.fit_transform(train_eng)
val_eng = scaler.transform(val_eng)
test_eng = scaler.transform(test_eng)

# Build tf.data Datasets.
train_ds = create_dual_dataset(train_raw, train_eng, window_length, batch_size, shuffle=True)
val_ds   = create_dual_dataset(val_raw, val_eng, window_length, batch_size)
test_ds  = create_dual_dataset(test_raw, test_eng, window_length, batch_size)

# Prepare ensemble containers and training seeds.
ensemble_models = []
val_accuracies = []
seeds = [random.randint(0, num_samples - sub_samples - split) for _ in range(n_models)]

# For class imbalance, we assume targets are digits (0-9) derived from raw series.
# We extract targets from the validation dataset for later weighting.
# (Here we take the first batch as representative.)
sample_batch = next(iter(val_ds))
_, y_sample = sample_batch
unique_classes = np.unique(y_sample)
# Assume y_train here from train_ds (for brevity we use y_sample)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_sample.numpy().flatten())
class_weights_dict = dict(enumerate(class_weights))

for i, seed in enumerate(seeds):
    tf.keras.backend.clear_session()
    gc.collect()
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
    model = create_dual_branch_model(seed, window_length, eng_feature_dim=train_eng.shape[1])
    
    print(f"\nTraining model {i+1}/{n_models}")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, cooldown=5)
    ]
    
    steps_per_epoch = math.ceil(len(train_eng) / batch_size)
    val_steps = math.ceil((split//2) / batch_size)
    
    model.fit(train_ds,
              steps_per_epoch=steps_per_epoch,
              validation_data=val_ds,
              validation_steps=val_steps,
              epochs=epochs,
              callbacks=callbacks,
              class_weight=class_weights_dict,
              verbose=1)
    
    loss, accuracy = model.evaluate(val_ds, steps=val_steps, verbose=0)
    print(f"Validation accuracy for model {i+1}: {accuracy:.4f}")
    ensemble_models.append(model)
    val_accuracies.append(accuracy)

# Compute dynamic weights for ensemble based on validation accuracy.
total_acc = sum(val_accuracies)
weights = [acc / total_acc for acc in val_accuracies]
print("\nDynamic Model Weights:", weights)

# Evaluate full ensemble on test data.
ensemble_preds = ensemble_predict_weighted(ensemble_models, weights, test_ds)
# Extract targets from test dataset (assuming first batch for simplicity)
sample_test_batch = next(iter(test_ds))
_, y_test = sample_test_batch
ensemble_accuracy = np.mean(np.argmax(ensemble_preds, axis=1).astype(np.int8) == y_test.numpy().flatten())
print("\nWeighted Ensemble Accuracy on Test Set:", ensemble_accuracy)

# Compute per-model predictions for error metrics.
predictions_list = get_individual_predictions(ensemble_models, test_ds)
q_matrix = compute_q_statistic_matrix(predictions_list, y_test.numpy().flatten())
print("\nPairwise Q-statistic Matrix:\n", q_matrix)
df_matrix = compute_double_fault_matrix(predictions_list, y_test.numpy().flatten())
print("\nPairwise Double-Fault Matrix:\n", df_matrix)

# --- Diverse Model Selection using Q-statistic ---
selected_indices = select_diverse_models(q_matrix, val_accuracies, threshold=0.3)
print("Selected diverse model indices:", selected_indices)
reduced_q_matrix = recalc_q_matrix(selected_indices, q_matrix)
print("Reduced Q-statistic Matrix for selected models:\n", reduced_q_matrix)
selected_models = [ensemble_models[i] for i in selected_indices]
selected_val_acc = [val_accuracies[i] for i in selected_indices]
total_selected_acc = sum(selected_val_acc)
selected_weights = [acc / total_selected_acc for acc in selected_val_acc]
print("Selected Model Weights:", selected_weights)
selected_ensemble_preds = ensemble_predict_weighted(selected_models, selected_weights, test_ds)
selected_ensemble_accuracy = np.mean(np.argmax(selected_ensemble_preds, axis=1).astype(np.int8) == y_test.numpy().flatten())
print("Selected Ensemble Accuracy on Test Set:", selected_ensemble_accuracy)

# --- Alternative Ensemble Selection: Models above Uniform Weight ---
alt_selected_indices = [i for i, w in enumerate(weights) if w >= 1 / n_models]
print("\nSelected indices based on weight threshold:", alt_selected_indices)
alt_selected_models = [ensemble_models[i] for i in alt_selected_indices]
alt_selected_val_acc = [val_accuracies[i] for i in alt_selected_indices]
total_alt_acc = sum(alt_selected_val_acc)
alt_selected_weights = [acc / total_alt_acc for acc in alt_selected_val_acc]
print("Alternative Selected Model Weights:", alt_selected_weights)
alt_ensemble_preds = ensemble_predict_weighted(alt_selected_models, alt_selected_weights, test_ds)
alt_ensemble_accuracy = np.mean(np.argmax(alt_ensemble_preds, axis=1).astype(np.int8) == y_test.numpy().flatten())
print("Alternative Ensemble Accuracy on Test Set:", alt_ensemble_accuracy)
