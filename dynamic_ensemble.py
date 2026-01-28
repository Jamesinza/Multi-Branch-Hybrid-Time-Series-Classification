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
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout, Add, \
    LayerNormalization, MultiHeadAttention
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from keras_hub.layers import TransformerEncoder, SinePositionEncoding
from hmmlearn.hmm import GaussianHMM, CategoricalHMM, GMMHMM, PoissonHMM

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

###############################################################################
# DATASET & FEATURE EXTRACTION FUNCTIONS
###############################################################################

def create_dataset(sequence, window_length, batch_size, shuffle=False):
    """Create a cached and prefetch-enabled tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(window_length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_length + 1))
    # Assume the last column is the target; the remaining columns are features.
    ds = ds.map(lambda window: (window[:-1, :-1], window[-1, -1]),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.repeat().prefetch(tf.data.AUTOTUNE)
    return ds

def natural_visibility_graph(window):
    """
    Convert a 1D time series window into a visibility graph.
    Two nodes i and j (i < j) are connected if every intermediate k
    satisfies: window[k] < window[i] + (window[j]-window[i])*(k-i)/(j-i)
    """
    n = len(window)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            # Early break if any intermediate point blocks visibility.
            visible = True
            delta = window[j] - window[i]
            for k in range(i+1, j):
                # Linear interpolation from i to j at k.
                if window[k] >= window[i] + delta * (k - i) / (j - i):
                    visible = False
                    break
            if visible:
                G.add_edge(i, j)
    return G

def compute_visibility_features(window):
    """Compute average degree and clustering from the natural visibility graph."""
    window = np.array(window).flatten()
    G = natural_visibility_graph(window)
    if len(G) == 0:
        return np.array([0, 0], dtype=np.float32)
    avg_degree = np.mean([d for _, d in G.degree()])
    clustering = nx.average_clustering(G)
    return np.array([avg_degree, clustering], dtype=np.float32)

def get_extra_features(X_raw, rng):
    """
    Augment the raw input with HMM-based and NVG-based features.
    The raw data X_raw is assumed to be a 2D array.
    """
    # Make copies for HMM-based features
    hs1 = X_raw.copy()
    hs2 = X_raw.copy()
    hs3 = X_raw.copy()
    hs4 = X_raw.copy()
    base_features = X_raw.copy()

    # --- HMM Feature Augmentation ---
    # Loop over a single iteration or more if needed.
    for i in range(1):  
        # GaussianHMM and CategoricalHMM
        hmm_g = GaussianHMM(n_components=10 - i, covariance_type="full", random_state=rng)
        hmm_c = CategoricalHMM(n_components=10 - i, n_features=10, random_state=rng)
        hmm_g.fit(hs1)
        hmm_c.fit(hs2)
        hs1_pred = hmm_g.predict(hs1).reshape(-1, 1).astype(np.float32)
        hs2_pred = hmm_c.predict(hs2).reshape(-1, 1).astype(np.float32)
        base_features = np.hstack([base_features, hs1_pred, hs2_pred])
        hs1, hs2 = hs1_pred, hs2_pred  # Update for potential further iterations

    # GMMHMM Feature
    hmm_gmm = GMMHMM(n_components=10, n_mix=1, covariance_type="full", random_state=rng)
    hmm_gmm.fit(hs3)
    hs3_pred = hmm_gmm.predict(hs3).reshape(-1, 1).astype(np.float32)
    base_features = np.hstack([base_features, hs3_pred])

    # PoissonHMM Feature
    hmm_pois = PoissonHMM(n_components=10, random_state=rng)
    hmm_pois.fit(hs4)
    hs4_pred = hmm_pois.predict(hs4).reshape(-1, 1).astype(np.float32)
    base_features = np.hstack([base_features, hs4_pred])
    
    # --- NVG Feature Augmentation ---
    nvg_features = []
    # Compute NVG features in a sliding window of length 10.
    win_size = 10
    for i in range(len(X_raw) - win_size):
        feat = compute_visibility_features(X_raw[i:i+win_size])
        nvg_features.append(feat)
    nvg_features = np.array(nvg_features, dtype=np.float32)
    
    # Trim base_features to align with NVG features
    base_features = base_features[win_size:]
    # Append NVG features to base features.
    X_augmented = np.hstack([base_features, nvg_features])
    return X_augmented

def get_real_data(num_samples, dataset):
    """Load and preprocess real data from a CSV."""
    print('\nBuilding dataframe using real data...')
    df = pd.read_csv(f'datasets/UK/{dataset}_ascend.csv')
    print(df.head())
    if dataset == 'Euro' or 'Thunderball':
        cols = ['A', 'B', 'C', 'D', 'E']
    else:
        cols = ['A', 'B', 'C', 'D', 'E', 'F']
    df = df[cols].dropna().astype(np.int8)
    # Format each element as 2-digit string and then flatten digits into an array.
    df = df.map(lambda x: f'{x:02d}')
    flattened = df.values.flatten()
    full_data = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int8)
    return full_data[:num_samples]

###############################################################################
# ENSEMBLE & METRICS FUNCTIONS
###############################################################################

def ensemble_predict_weighted(models, weights, X):
    """
    Compute weighted predictions from an ensemble.
    The predictions are summed after weighting each model's softmax output.
    """
    preds = [w * model.predict(X, verbose=0) for model, w in zip(models, weights)]
    return np.sum(np.array(preds), axis=0)

def get_individual_predictions(models, X_test):
    """Obtain predictions from each model (as class indices) for test data."""
    return [np.argmax(model.predict(X_test, verbose=0), axis=1) for model in models]

def compute_error_correlations(preds, y_true):
    """
    Compute pairwise Pearson correlations between binary error vectors.
    Uses vectorized operations with safe correction when standard deviation is zero.
    """
    errors = np.array([(pred != y_true).astype(np.float32) for pred in preds])
    # Compute correlation matrix using np.corrcoef along the appropriate axis.
    corr = np.corrcoef(errors)
    # Replace nan values (from constant vectors) with zeros.
    corr = np.nan_to_num(corr)
    return corr

def compute_q_statistic_matrix(preds, y_true):
    """
    Compute the pairwise Q-statistic matrix.
    Q = (N11 * N00 - N10 * N01) / (N11 * N00 + N10 * N01)
    """
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
            q_matrix[i, j] = (N11 * N00 - N10 * N01) / denom if denom != 0 else 0.0
    return q_matrix

def compute_double_fault_matrix(preds, y_true):
    """
    Compute the pairwise double-fault measure matrix.
    DF = N00 / (N00 + N10 + N01)
    """
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
    """
    Greedily select models with pairwise Q-statistic below threshold.
    The models are sorted by decreasing validation accuracy.
    """
    sorted_indices = np.argsort(val_accuracies)[::-1]
    selected = []
    for idx in sorted_indices:
        if all(q_matrix[idx, sel] <= threshold for sel in selected):
            selected.append(idx)
    return selected

def recalc_q_matrix(selected_indices, full_q_matrix):
    """Extract submatrix corresponding to the selected model indices."""
    return full_q_matrix[np.ix_(selected_indices, selected_indices)]

###############################################################################
# MODEL ARCHITECTURE & TRAINING FUNCTIONS
###############################################################################

def create_random_model(seed, data, val_ds, X_test, y_test, scaler, window_length, batch_size):
    """
    Create a diverse Transformer-inspired model with random hyperparameters,
    and prepare training, validation, and test datasets.
    """
    print(f'Input data shape: {data.shape}')
    # Reshape raw data and separate target
    X_raw = data.reshape(-1, 1)
    y_data = data[10:].reshape(-1, 1)
    # Enhance features using HMM and NVG based approaches
    X_augmented = get_extra_features(X_raw, rng=42)
    
    features = X_augmented.shape[1]
    input_shape = (window_length, features)
    
    # Scale only the extra features (keep target unscaled)
    X_scaled = scaler.fit_transform(X_augmented)
    # Append unscaled target as final column
    X_full = np.hstack([X_scaled, y_data[10 - 10:]])  # Align target with augmented features

    # Split dataset indices: use the initial part for training and last part for validation/test.
    split = 1_000
    train_data = X_full[:-split]
    if val_ds is None:
        val_data = X_full[-split:-split//2]
        X_test_arr = X_full[-split//2:]
        # Build test arrays from the windows
        X_test_new = np.empty([len(X_test_arr)-window_length, window_length, features], dtype=np.float32)
        y_test_new = np.empty([len(X_test_arr)-window_length, 1], dtype=np.int8)
        for i in range(len(X_test_arr)-window_length):
            X_test_new[i] = X_test_arr[i:i+window_length, :-1]
            y_test_new[i] = X_test_arr[i+window_length, -1]
        X_test, y_test = X_test_new, y_test_new

        val_ds = create_dataset(val_data, window_length, batch_size)
    
    train_ds = create_dataset(train_data, window_length, batch_size, shuffle=True)
    
    # Derive class weights from training target to address imbalance.
    y_train = y_data[:len(X_augmented)][:, 0]
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    
    # Calculate steps per epoch
    train_steps = math.ceil(len(train_data) / batch_size)
    val_steps = math.ceil((split//2) / batch_size)
    
    # Random hyperparameters
    chosen_units      = random.choice([128])
    chosen_layers     = random.choice([2])
    chosen_activation = random.choice(['gelu'])
    chosen_dropout    = random.choice([0.3])
    chosen_optimizer  = random.choice(['adamw'])
    chosen_flat       = random.choice(['gap'])
    
    # Transformer-inspired architecture hyperparameters
    hid_dim = chosen_units
    num_heads = 2
    int_dim = hid_dim * 4
    key_dim = hid_dim // num_heads

    # Build model architecture using Keras Functional API.
    inputs = Input(shape=input_shape)
    proj = Dense(hid_dim)(inputs)
    pos_enc = SinePositionEncoding()(proj)
    x = Add()([proj, pos_enc])
    
    for _ in range(chosen_layers):
        mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
        if chosen_dropout > 0:
            mha = Dropout(chosen_dropout)(mha)
        x = Add()([x, mha])
        x = LayerNormalization()(x)
        ffn = Dense(int_dim, activation=chosen_activation)(x)
        ffn = Dense(hid_dim, activation=chosen_activation)(ffn)
        if chosen_dropout > 0:
            ffn = Dropout(chosen_dropout)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)
    
    # Global pooling (optionally average or max)
    if chosen_flat == 'gap':
        x = GlobalAveragePooling1D()(x)
    else:
        x = GlobalMaxPooling1D()(x)
        
    x = Dense(hid_dim, activation=chosen_activation)(x)
    if chosen_dropout > 0:
        x = Dropout(chosen_dropout)(x)
    outputs = Dense(10, activation='softmax', dtype='float32')(x)
    model = Model(inputs, outputs)
    
    # Select optimizer instance based on hyperparameter choice.
    if chosen_optimizer == 'adamw':
        optimizer_instance = tf.keras.optimizers.AdamW(learning_rate=1e-3)
    else:
        optimizer_instance = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer=optimizer_instance,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  jit_compile=True)
    
    print(f"Created model: layers={chosen_layers}, units={chosen_units}, activation={chosen_activation}, "
          f"dropout={chosen_dropout}, optimizer={chosen_optimizer}, pooling={chosen_flat}, seed={seed}")
    return model, train_ds, val_ds, X_test, y_test, train_steps, val_steps, class_weights_dict

def main():
    ###############################################################################
    # MAIN ENSEMBLE TRAINING & EVALUATION LOOP
    ###############################################################################

    # Hyperparameters
    batch_size = 32
    epochs = 100
    window_length = 10
    num_samples = 30_000
    sub_samples = 28_000
    test_samples = 1_000
    n_models = 10

    # Data and scaler setup
    dataset = 'HotPicks'
    scaler = StandardScaler()
    data = get_real_data(num_samples, dataset)
    print(f'\nData shape: {data.shape}')
    val_test_data = data[-test_samples:]

    # Prepare ensemble containers and random seeds for each model training.
    ensemble_models = []
    val_accuracies = []
    seeds = [random.randint(0, num_samples - sub_samples - test_samples) for _ in range(n_models)]

    # Initialize validation and test datasets once (they remain fixed across models)
    val_ds, X_test, y_test = None, None, None

    for i, seed in enumerate(seeds):
        tf.keras.backend.clear_session()
        gc.collect()
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        # Combine a training subset with the fixed validation/test segment.
        train_data = data[seed:seed + sub_samples]
        train_data = np.concatenate([train_data, val_test_data], axis=0)

        model, train_ds, val_ds, X_test, y_test, train_steps, val_steps, class_weights_dict = create_random_model(
            seed, train_data, val_ds, X_test, y_test, scaler, window_length, batch_size
        )

        print(f"\nTraining model {i+1}/{n_models}")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, cooldown=5)
        ]

        model.fit(train_ds,
                  steps_per_epoch=train_steps,
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

    # Calculate dynamic weights for ensemble based on validation accuracies.
    total_acc = sum(val_accuracies)
    weights = [acc / total_acc for acc in val_accuracies]
    print("\nDynamic Model Weights:", weights)

    # Evaluate the full ensemble on test data.
    ensemble_preds = ensemble_predict_weighted(ensemble_models, weights, X_test)
    ensemble_accuracy = np.mean(np.argmax(ensemble_preds, axis=1).astype(np.int8) == y_test.flatten())
    print("\nWeighted Ensemble Accuracy on Test Set:", ensemble_accuracy)

    # Compute per-model predictions for error-based metrics.
    predictions_list = get_individual_predictions(ensemble_models, X_test)
    err_corr = compute_error_correlations(predictions_list, y_test.flatten())
    print("\nPairwise Error Correlations:\n", err_corr)
    q_matrix = compute_q_statistic_matrix(predictions_list, y_test.flatten())
    print("\nPairwise Q-statistic Matrix:\n", q_matrix)
    df_matrix = compute_double_fault_matrix(predictions_list, y_test.flatten())
    print("\nPairwise Double-Fault Matrix:\n", df_matrix)

    # --- Model Selection: Diverse Models using Q-statistic ---
    selected_indices = select_diverse_models(q_matrix, val_accuracies, threshold=0.3)
    print("Selected diverse model indices:", selected_indices)
    reduced_q_matrix = recalc_q_matrix(selected_indices, q_matrix)
    print("Reduced Q-statistic Matrix for selected models:\n", reduced_q_matrix)

    selected_models = [ensemble_models[i] for i in selected_indices]
    selected_val_acc = [val_accuracies[i] for i in selected_indices]
    total_selected_acc = sum(selected_val_acc)
    selected_weights = [acc / total_selected_acc for acc in selected_val_acc]
    print("Selected Model Weights:", selected_weights)

    selected_ensemble_preds = ensemble_predict_weighted(selected_models, selected_weights, X_test)
    selected_ensemble_accuracy = np.mean(np.argmax(selected_ensemble_preds, axis=1).astype(np.int8) == y_test.flatten())
    print("Selected Ensemble Accuracy on Test Set:", selected_ensemble_accuracy)

    # --- Alternative Selection: Models with Weight Above Uniform Average ---
    alt_selected_indices = [i for i, w in enumerate(weights) if w >= 1 / n_models]
    print("\nSelected indices based on weight threshold:", alt_selected_indices)

    alt_selected_models = [ensemble_models[i] for i in alt_selected_indices]
    alt_selected_val_acc = [val_accuracies[i] for i in alt_selected_indices]
    total_alt_acc = sum(alt_selected_val_acc)
    alt_selected_weights = [acc / total_alt_acc for acc in alt_selected_val_acc]
    print("Alternative Selected Model Weights:", alt_selected_weights)

    alt_ensemble_preds = ensemble_predict_weighted(alt_selected_models, alt_selected_weights, X_test)
    alt_ensemble_accuracy = np.mean(np.argmax(alt_ensemble_preds, axis=1).astype(np.int8) == y_test.flatten())
    print("Alternative Ensemble Accuracy on Test Set:", alt_ensemble_accuracy)

if __name__ == '__main__':
    main()
