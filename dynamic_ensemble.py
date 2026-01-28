import gc
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.utils.class_weight import compute_class_weight
from keras_hub.layers import TransformerEncoder, SinePositionEncoding
from hmmlearn.hmm import GaussianHMM, CategoricalHMM, GMMHMM, PoissonHMM

# Environment variables
# tf.config.set_visible_devices([], 'GPU')
mixed_precision.set_global_policy('mixed_float16')


def select_diverse_models(q_matrix, val_accuracies, threshold=0.3):
    """
    Select a subset of models based on Q-statistics.
    
    Args:
        q_matrix (np.array): Pairwise Q-statistic matrix (n_models x n_models).
        val_accuracies (list or np.array): Validation accuracies for each model.
        threshold (float): Maximum allowed Q-statistic between two models to be considered diverse.
                           Models with pairwise Q-statistic above this threshold are considered too similar.
    
    Returns:
        list: Indices of the selected models.
    """
    n_models = len(val_accuracies)
    # Order models by descending validation accuracy.
    sorted_indices = np.argsort(val_accuracies)[::-1]
    selected = []
    
    for idx in sorted_indices:
        is_diverse = True
        for sel_idx in selected:
            # Check the Q-statistic (absolute value if you want symmetry in disagreement,
            # but here a high positive Q means similarity)
            if q_matrix[idx, sel_idx] > threshold:
                is_diverse = False
                break
        if is_diverse:
            selected.append(idx)
    return selected

# Ensemble prediction with dynamic weighting
def ensemble_predict_weighted(models, weights, X):
    predictions = np.array([w * model.predict(X, verbose=0) for model, w in zip(models, weights)])
    return np.sum(predictions, axis=0)

# Recalculate the Q-statistics for only the selected models (for confirmation)
def recalc_q_matrix(selected_indices, full_q_matrix):
    # Extract the submatrix corresponding to the selected models.
    return full_q_matrix[np.ix_(selected_indices, selected_indices)]  

def get_individual_predictions(models, X_test):
    """
    Get predictions for each model on the test set.
    
    Args:
        models (list): List of trained models.
        X_test (np.array): Test dataset inputs.
        
    Returns:
        predictions (list): List of prediction arrays (labels) for each model.
    """
    preds = []
    for model in models:
        # model.predict returns softmax outputs; take argmax to get predicted class.
        probas = model.predict(X_test, verbose=0)
        preds.append(np.argmax(probas, axis=1))
    return preds

def compute_error_correlations(preds, y_true):
    """
    Computes pairwise Pearson correlation coefficients between error vectors of models.
    
    Args:
        preds (list of np.array): List of predictions (labels) for each model.
        y_true (np.array): Ground truth labels.
    
    Returns:
        corr_matrix (np.array): Correlation matrix (n_models x n_models).
    """
    n_models = len(preds)
    n_samples = len(y_true)
    # Compute error vectors: 1 if error, 0 if correct.
    errors = np.array([ (pred != y_true).astype(np.float32) for pred in preds ])
    corr_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            # If constant vector (all zeros or ones), correlation is undefined. Set to 0.
            if np.std(errors[i]) == 0 or np.std(errors[j]) == 0:
                corr_matrix[i, j] = 0.0
            else:
                corr_matrix[i, j] = pearsonr(errors[i], errors[j])[0]
    return corr_matrix

def compute_q_statistic_matrix(preds, y_true):
    """
    Computes the Q-statistic for each pair of models.
    
    For two models, the Q-statistic is defined as:
        Q = (N11 * N00 - N10 * N01) / (N11 * N00 + N10 * N01)
    where:
        N11: both models correct
        N00: both models wrong
        N10: model i correct, model j wrong
        N01: model i wrong, model j correct
    
    Args:
        preds (list of np.array): List of predictions for each model.
        y_true (np.array): Ground truth labels.
    
    Returns:
        q_matrix (np.array): Matrix of Q-statistics (n_models x n_models).
    """
    n_models = len(preds)
    n_samples = len(y_true)
    # Compute correctness as boolean arrays (True if correct)
    correctness = [ (pred == y_true) for pred in preds ]
    q_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            # Count outcomes for the pair (i, j)
            N11 = np.sum(np.logical_and(correctness[i], correctness[j]))
            N00 = np.sum(np.logical_and(~correctness[i], ~correctness[j]))
            N10 = np.sum(np.logical_and(correctness[i], ~correctness[j]))
            N01 = np.sum(np.logical_and(~correctness[i], correctness[j]))
            denominator = (N11 * N00 + N10 * N01)
            if denominator == 0:
                q_matrix[i, j] = 0.0  # Avoid division by zero
            else:
                q_matrix[i, j] = (N11 * N00 - N10 * N01) / denominator
    return q_matrix

def compute_double_fault_matrix(preds, y_true):
    """
    Computes the double-fault measure for each pair of models.
    
    For two models, the double-fault measure is defined as:
        DF = N00 / (N00 + N10 + N01)
    where:
        N00: both models wrong
        N10: model i correct, model j wrong
        N01: model i wrong, model j correct
    
    Args:
        preds (list of np.array): List of predictions for each model.
        y_true (np.array): Ground truth labels.
    
    Returns:
        df_matrix (np.array): Matrix of double-fault measures (n_models x n_models).
    """
    n_models = len(preds)
    df_matrix = np.zeros((n_models, n_models))
    # Compute correctness as boolean arrays
    correctness = [ (pred == y_true) for pred in preds ]
    
    for i in range(n_models):
        for j in range(n_models):
            N00 = np.sum(np.logical_and(~correctness[i], ~correctness[j]))
            N10 = np.sum(np.logical_and(correctness[i], ~correctness[j]))
            N01 = np.sum(np.logical_and(~correctness[i], correctness[j]))
            denominator = (N00 + N10 + N01)
            if denominator == 0:
                df_matrix[i, j] = 0.0
            else:
                df_matrix[i, j] = N00 / denominator
    return df_matrix

def natural_visibility_graph(window):
    """
    Convert a 1D time series (window) into a visibility graph.
    
    Args:
        window (np.ndarray): 1D array of time series data.
        
    Returns:
        G: A networkx Graph representing the visibility graph.
    """
    n = len(window)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Connect each pair of points that can "see" each other.
    # Two nodes i and j (with i<j) are connected if for every k in between, 
    # window[k] < window[i] + (window[j] - window[i])*(k - i)/(j - i)
    for i in range(n):
        for j in range(i+1, n):
            visible = True
            for k in range(i+1, j):
                # Calculate the height of the line connecting i and j at position k
                height = window[i] + (window[j] - window[i]) * (k - i) / (j - i)
                if window[k] >= height:
                    visible = False
                    break
            if visible:
                G.add_edge(i, j)
    return G

def compute_visibility_features(window):
    """
    Compute a set of features from the natural visibility graph of a time series window.
    
    Returns:
        features: np.array of extracted features (e.g., average degree, clustering coefficient).
    """
    window = np.array(window).flatten()
    G = natural_visibility_graph(window)
    if len(G) == 0:
        return np.array([0, 0], dtype=np.float32)
    
    avg_degree = np.mean([d for _, d in G.degree()])
    clustering = nx.average_clustering(G)
    return np.array([avg_degree, clustering], dtype=np.float32)

# Prepare datasets for training and testing
def create_dataset(sequence, W, batch_size, shuffle=False):
    """
    Create a TensorFlow dataset for sequence prediction, with windows of size W+1,
    where W inputs are used to predict 1 output.
    
    Args:
        sequence (numpy.ndarray): Input sequence.
        W (int): Window size for inputs.
        batch_size (int): Batch size for training.
    
    Returns:
        tf.data.Dataset: Prepared dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices(sequence)
    dataset = dataset.window(W + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(W + 1))
    # dataset = dataset.map(lambda window: (window[:-1], window[-1, 0]))
    dataset = dataset.map(lambda window: (window[:-1, :-1], window[-1, -1]))
    if shuffle==True:
        dataset = dataset.batch(batch_size).shuffle(1000).repeat().prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    return dataset

def get_base_data(num_samples):
    print(f'\nBuilding dataframe using base data...\n')
    quick_df = pd.read_csv('datasets/Quick.csv')
    quick_df = quick_df.drop(columns=['Unnamed: 0']).dropna().reset_index(drop=True).astype(np.int8)
    quick_df = quick_df.map(lambda x: f'{x:02d}')
    flattened = quick_df.values.flatten()
    quick_df = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int8)
    return quick_df[:num_samples]

def get_real_data(num_samples):
    print(f'\nBuilding dataframe using real data...\n')
    dataset = 'Take5'
    target_df = pd.read_csv(f'datasets/{dataset}_Full.csv')
    cols = ['A', 'B', 'C', 'D', 'E']
    target_df = target_df[cols].dropna().astype(np.int8)
    target_df = target_df.map(lambda x: f'{x:02d}')
    flattened = target_df.values.flatten()
    full_data = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int8)
    return full_data[:num_samples]

def get_extra_features(X_raw, rnd):
    print(f'\nCreating HMM-based features...')
    hidden_states0 = X_raw.copy()
    hidden_states1 = X_raw.copy()
    hidden_states2 = X_raw.copy()
    hidden_states3 = X_raw.copy()
    hidden_states4 = X_raw.copy()
    nvg_data = X_raw.copy()
    
    # Stacked GaussianHMM and CategoricalHMM feature creation
    for i in range(1):
        hmm1 = GaussianHMM(n_components=10-i, covariance_type="full", random_state=rnd)
        hmm2 = CategoricalHMM(n_components=10-i, n_features=10, random_state=rnd)
        hmm1.fit(hidden_states1)
        hmm2.fit(hidden_states2)
        # Get hidden states
        hidden_states1 = hmm1.predict(hidden_states1)
        hidden_states2 = hmm2.predict(hidden_states2)
        # Reshape all data to feed back to HMM's
        hidden_states1 = hidden_states1.reshape(-1, 1).astype(np.float32)
        hidden_states2 = hidden_states2.reshape(-1, 1).astype(np.float32)
        # Adding data as new feature to X
        X_raw = np.hstack([X_raw, hidden_states1, hidden_states2])
        
    # for i in range(2):
    hmm3 = GMMHMM(n_components=10, n_mix=1, covariance_type="full", random_state=rnd)
    hmm3.fit(hidden_states3)
    hidden_states3 = hmm3.predict(hidden_states3)
    # Reshape all data to feed back to HMM
    hidden_states3 = hidden_states3.reshape(-1, 1).astype(np.float32)
    # Adding data as new feature to X
    X_raw = np.hstack([X_raw, hidden_states3])
    
    # PoissonHMM feature creation
    hmm4 = PoissonHMM(n_components=10, random_state=rnd)
    hmm4.fit(hidden_states4)
    hidden_states4 = hmm4.predict(hidden_states4)
    hidden_states4 = hidden_states4.reshape(-1, 1).astype(np.float32)
    # Adding final feature to X_raw
    X_raw = np.hstack([X_raw, hidden_states4])
    del hmm1,hmm2,hmm3,hmm4,hidden_states1,hidden_states2,hidden_states3,hidden_states4
    
    print(f'\nCreating NVG-based features...\n')
    nvg = None
    for i in range(len(nvg_data)-10):
        if nvg is None:
            nvg = compute_visibility_features(nvg_data[i:i+10])
        else:
            nvg = np.vstack([nvg, compute_visibility_features(nvg_data[i:i+10])])
            
    X_raw = np.hstack([X_raw[10:], nvg])
    return X_raw

# Define a function to create a diverse model
def create_random_model(seed, data, val_ds, X_test, y_test):
    print(f'data_shape: {data.shape}')
    X_raw = data.copy().reshape(-1,1) # Reshape data for HMM
    y_data = data[10:].copy().reshape(-1,1) # Keeping input and output data seperate
    X_raw = get_extra_features(X_raw, 42) # Creating extra features
    
    features = X_raw.shape[1]
    input_shape = (wl, features)
    
    # Scaling input data only
    X = scaler.fit_transform(X_raw)
    print(f'\nX_shape: {X.shape}')
    print(f'ydata_shape: {y_data.shape}\n')
    X = np.hstack([X, y_data]) # Adding unscaled output data before passing to tf.data pipeline
    
    # Create datasets for training and validation
    split = 10_000
    train_ds = create_dataset(X[:len(X)-split], wl, batch_size, shuffle=False) # Only shuffling training data
    if val_ds is None:
        val_data = X[len(X)-split:len(X)-split//2]
        val_ds = create_dataset(val_data, wl, batch_size)
        # test_ds = create_dataset(X[split*9:], wl, batch_size)

        # Create test dataset for ensemble predictions and testing
        test_data = X[-split//2:]
        X_test = np.empty([len(test_data)-wl, wl, features], dtype=np.float32)
        y_test = np.empty([len(test_data)-wl, 1], dtype=np.int8)
        for i in range(len(test_data)-wl):
            X_test[i] = test_data[i:i+wl, :-1]
            y_test[i] = test_data[i+wl, -1]
    
    y_train = y_data[:len(X_raw)][:,0] # Used for class weighting
    
    # Calculations for steps per epoch
    train_len = len(X)-split
    val_len = split//2
    # test_len = len(X[split:split+split//2:])
    train_steps = math.ceil(train_len / batch_size)
    val_steps = math.ceil(val_len / batch_size)
    # test_steps = math.ceil(test_len / batch_size)    
    
    # Random hyperparameters from predetermined lists
    num_layers = [2]
    units_options = [128]
    activation_options = ['gelu'] #['relu', 'gelu', 'swish', 'tanh', 'celu', 'elu', 'selu']
    dropout_options = [0.3]
    optimizer_options = ['adamw'] #['adam', 'adamw'] #, 'adagrad', 'adadelta', 'adamax', 'adafactor', 'nadam']
    # arch_options = ['conv', 'lstm', 'gru']
    flat_options = ['gap']
    
    # Pick random configuration
    chosen_units = random.choice(units_options)
    chosen_layers = random.choice(num_layers)
    chosen_activation = random.choice(activation_options)
    chosen_dropout = random.choice(dropout_options)
    chosen_optimizer = random.choice(optimizer_options)
    chosen_flat = random.choice(flat_options)
    
    hid_dim = chosen_units
    num_heads = 2
    int_dim = hid_dim*4
    key_dim = hid_dim//num_heads

    inputs = tf.keras.layers.Input(shape=(wl,features))
    proj = tf.keras.layers.Dense(hid_dim)(inputs)
    pos_enc = SinePositionEncoding()(proj)
    x = pos_enc + proj
    for _ in range(chosen_layers):        
        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x,x)
        if chosen_dropout > 0.0:
            mha = tf.keras.layers.Dropout(chosen_dropout)(mha)
        mha = tf.keras.layers.Add()([mha, x])
        x = tf.keras.layers.LayerNormalization()(mha)
        
        ffn = tf.keras.layers.Dense(int_dim, activation=chosen_activation)(x)
        ffn = tf.keras.layers.Dense(hid_dim, activation=chosen_activation)(ffn)
        if chosen_dropout > 0.0:
            ffn = tf.keras.layers.Dropout(chosen_dropout)(ffn)
        ffn = tf.keras.layers.Add()([ffn, x])
        x = tf.keras.layers.LayerNormalization()(ffn)

    if chosen_flat == 'gap':
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    else:
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        
    x = tf.keras.layers.Dense(hid_dim, activation=chosen_activation)(x)
    if chosen_dropout > 0.0:
        x = tf.keras.layers.Dropout(chosen_dropout)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=chosen_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  jit_compile=False,
                 )
    
    print(f"\nCreated model with {chosen_layers} layer(s), units: {chosen_units}, activation: {chosen_activation}, "
          f"dropout: {chosen_dropout}, optimizer: {chosen_optimizer}, flatten: {chosen_flat}, seed: {seed}")
    return model, train_ds, val_ds, X_test, y_test, train_steps, val_steps, y_train

# Hyperparameters
batch_size = 1024
epochs = 100
wl = 10
lr_rate = 1e-3

# Get data
scaler = StandardScaler()
num_samples = 110_000
sub_samples =  90_000
data = get_real_data(num_samples)
print(f'\ndata shape: {data.shape}')
val_test_data = data[-10_000:]

# Create and train an ensemble of models (e.g., 100 models)
n_models = 100
seeds = [random.randint(0, num_samples-sub_samples-10_000) for _ in range(n_models)]
ensemble_models = []
val_accuracies = []

# Val and Test sets will be created once and used for all models to ensure fair validation and testing
val_ds = None
X_test = None
y_test = None

for i, seed in enumerate(seeds):
    tf.keras.utils.clear_session(free_memory=True)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    train_data = data[seed:seed+sub_samples]
    print(f'\ntrain_data shape: {train_data.shape}')
    print(f'val_test_data shape: {val_test_data.shape}\n')
    train_data = np.concatenate([train_data, val_test_data], axis=0)

    (model, train_ds, val_ds, X_test,
     y_test, train_steps, val_steps, y_train) = create_random_model(seed, train_data, val_ds, X_test, y_test)
    print(f"\nTraining model {i+1}/{n_models}")
    callback = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                         patience=20, verbose=1, mode='auto',
                                         baseline=None, restore_best_weights=True,
                                         start_from_epoch=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                             patience=5, verbose=0, mode='auto',
                                             min_delta=0.0, cooldown=5, min_lr=0),
        # tf.keras.callbacks.ModelCheckpoint(f'checkpoints/{model_name}.keras',
        #                                    monitor='val_loss', save_best_only=True),
    ]

    unique_classes = np.unique(data)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    
    model.fit(train_ds,
              steps_per_epoch=train_steps,
              validation_data=val_ds,
              validation_steps=val_steps,
              epochs=epochs, batch_size=batch_size,
              callbacks=callback,
              class_weight=class_weights_dict,
             )
    
    # Evaluate on the validation set and record accuracy
    loss, accuracy = model.evaluate(val_ds, steps=val_steps, verbose=0)
    print(f"\nValidation accuracy for model {i+1}: {accuracy:.4f}")
    ensemble_models.append(model)
    val_accuracies.append(accuracy)

# Calculate dynamic weights based on validation accuracies
total_acc = sum(val_accuracies)
weights = [acc / total_acc for acc in val_accuracies]
print("\nModel Weights:", weights)

# Ensemble prediction with dynamic weighting
def ensemble_predict_weighted(models, weights, X):
    predictions = np.array([w * model.predict(X, verbose=0) for model, w in zip(models, weights)])
    return np.sum(predictions, axis=0)

# Evaluate weighted ensemble on test data
ensemble_preds = ensemble_predict_weighted(ensemble_models, weights, X_test)
ensemble_accuracy = np.mean(np.argmax(ensemble_preds, axis=1).astype(np.int8) == y_test.flatten())
print("\nDynamic Weighted Ensemble Accuracy on Test Set:", ensemble_accuracy)

# Checking for error correlations
predictions_list = get_individual_predictions(ensemble_models, X_test)

# Compute error correlations
err_corr = compute_error_correlations(predictions_list, y_test.flatten())
print("\nPairwise Error Correlations:\n", err_corr)

# Compute Q-statistic matrix
q_matrix = compute_q_statistic_matrix(predictions_list, y_test.flatten())
print("\nPairwise Q-statistic Matrix:\n", q_matrix)

# Compute Double-Fault matrix
df_matrix = compute_double_fault_matrix(predictions_list, y_test.flatten())
print("\nPairwise Double-Fault Matrix:\n", df_matrix)

# Using q_matrix and val_accuracies to select most diverse models.
# selected_indices = [0,5,6,7,8,9]
selected_indices = select_diverse_models(q_matrix, val_accuracies, threshold=0.3)
print("Selected model indices (diverse subset):", selected_indices)

reduced_q_matrix = recalc_q_matrix(selected_indices, q_matrix)
print("\nReduced Q-statistic Matrix for selected models:\n", reduced_q_matrix)

# --- Form new ensemble using only the selected models ---
selected_models = [ensemble_models[i] for i in selected_indices]
selected_val_acc = [val_accuracies[i] for i in selected_indices]

# Re-weight selected models based on validation accuracies.
# (Here we simply normalize the accuracies to sum to 1, but you could also use an inverse-error scheme.)
total_selected_acc = sum(selected_val_acc)
selected_weights = [acc / total_selected_acc for acc in selected_val_acc]
print("\nSelected Model Weights:", selected_weights)

# Evaluate the new ensemble's accuracy on the test set using dynamic weighting.
selected_ensemble_preds = ensemble_predict_weighted(selected_models, selected_weights, X_test)
selected_ensemble_accuracy = np.mean(np.argmax(selected_ensemble_preds, axis=1).astype(np.int8) == y_test.flatten())
print("\nDynamic Weighted Ensemble Accuracy on Test Set (Selected Models):", selected_ensemble_accuracy)

# --- Re-compute Error Metrics for the Selected Ensemble ---
# 1. Error Correlation Matrix
selected_predictions_list = get_individual_predictions(selected_models, X_test)
selected_err_corr = compute_error_correlations(selected_predictions_list, y_test.flatten())
print("\nSelected Ensemble - Pairwise Error Correlations:\n", selected_err_corr)

# 2. Double-Fault Matrix
selected_df_matrix = compute_double_fault_matrix(selected_predictions_list, y_test.flatten())
print("\nSelected Ensemble - Pairwise Double-Fault Matrix:\n", selected_df_matrix)

# --- Other model selection experiment ---
selected_indices = []
for i,v in enumerate(weights):
    if v >= 1 / n_models:
        selected_indices.append(i)
print(f'\nSelected indices based on weights:\n{selected_indices}\n')

# Form new ensemble using only the selected models
selected_models = [ensemble_models[i] for i in selected_indices]
selected_val_acc = [val_accuracies[i] for i in selected_indices]

# Re-weight selected models based on validation accuracies.
# (Here we simply normalize the accuracies to sum to 1, but you could also use an inverse-error scheme.)
total_selected_acc = sum(selected_val_acc)
selected_weights = [acc / total_selected_acc for acc in selected_val_acc]
print("\nSelected Model Weights:", selected_weights)

# Evaluate the new ensemble's accuracy on the test set using dynamic weighting.
selected_ensemble_preds = ensemble_predict_weighted(selected_models, selected_weights, X_test)
selected_ensemble_accuracy = np.mean(np.argmax(selected_ensemble_preds, axis=1).astype(np.int8) == y_test.flatten())
print("\nDynamic Weighted Ensemble Accuracy on Test Set (Selected Models):", selected_ensemble_accuracy)
