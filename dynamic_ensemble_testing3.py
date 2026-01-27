import gc
import math
# import keras
import random
import numpy as np
import pandas as pd

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import networkx as nx
from scipy.stats import norm
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.utils.class_weight import compute_class_weight
from keras_hub.layers import TransformerEncoder, SinePositionEncoding
from hmmlearn.hmm import GaussianHMM, CategoricalHMM, GMMHMM, PoissonHMM

# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import layers, models

# from sklearn.model_selection import train_test_split

# Set global seeds for reproducibility
# np.random.seed(42)
# tf.random.set_seed(42)
# random.seed(42)

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

# # Load and preprocess MNIST dataset
# (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
# # Normalize and flatten images
# x_train_full = x_train_full.reshape(-1, 28, 28).astype("float32") / 255.
# x_test = x_test.reshape(-1, 28, 28).astype("float32") / 255.
# # One-hot encode labels
# y_train_full = to_categorical(y_train_full, 10)
# y_test = to_categorical(y_test, 10)

# # Split the full training set into training and validation sets
# x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# Define a function to create a diverse model
def create_random_model(seed, data):
    X_raw = data.copy().reshape(-1,1) # Reshape data for HMM
    y_data = data[10:].copy().reshape(-1,1) # Keeping input and output data seperate
    X_raw = get_extra_features(X_raw, seed) # Creating extra features    
    
    features = X_raw.shape[1]
    input_shape = (wl, features)
    
    # Scaling input data only
    X = scaler.fit_transform(X_raw)
    X = np.hstack([X, y_data]) # Adding unscaled output data before passing to tf.data pipeline
    
    # Create datasets for training and validation
    split = int(len(X)*0.1)
    train_ds = create_dataset(X[:split*8], wl, batch_size, shuffle=False) # Only shuffling training data
    val_ds = create_dataset(X[split*8:split*9], wl, batch_size)
    # test_ds = create_dataset(X[split*9:], wl, batch_size)

    # Create test dataset for ensemble predictions and testing
    test_data = X[split*9:]
    X_test = np.empty([len(test_data)-wl, wl, features], dtype=np.float32)
    y_test = np.empty([len(test_data)-wl, 1], dtype=np.int8)
    for i in range(len(test_data)-wl):
        X_test[i] = test_data[i:i+wl, :-1]
        y_test[i] = test_data[i+wl, -1]
    
    y_train = y_data[:split*8][:,0] # Used for class weighting    
    
    # Calculations for steps per epoch
    train_len = len(X[:split*8])
    val_len = len(X[split*8:split*9])
    test_len = len(X[split*9:])
    train_steps = math.ceil(train_len / batch_size)
    val_steps = math.ceil(val_len / batch_size)
    # test_steps = math.ceil(test_len / batch_size)    
    
    # Random hyperparameters from predetermined lists
    num_layers = [1, 2]
    units_options = [32, 64]
    activation_options = ['relu', 'gelu', 'swish', 'tanh']
    dropout_options = [0.1, 0.3, 0.5]
    optimizer_options = ['adam', 'adamw', 'rmsprop']
    # arch_options = ['conv', 'lstm', 'gru']
    flat_options = ['gap', 'gmp']
    
    # Pick random configuration
    chosen_units = random.choice(units_options)
    chosen_layers = random.choice(num_layers)
    chosen_activation = random.choice(activation_options)
    chosen_dropout = random.choice(dropout_options)
    chosen_optimizer = random.choice(optimizer_options)
    # chosen_arch = random.choice(arch_options)
    chosen_flat = random.choice(flat_options)
    
    # Build the model
    # model = models.Sequential()
    # model.add(layers.Input(shape=input_shape))
    
    # for units in chosen_units:
    #     if chosen_arch == 'conv':
    #         model.add(layers.Conv1D(units, 3, padding='same', activation=chosen_activation))
    #         if chosen_dropout > 0.0:
    #             model.add(layers.Dropout(chosen_dropout))
    #     elif chosen_arch == 'lstm':
    #         model.add(layers.LSTM(units, return_sequences=True, activation=chosen_activation))
    #         if chosen_dropout > 0.0:
    #             model.add(layers.Dropout(chosen_dropout))
    #     else:
    #         model.add(layers.GRU(units, return_sequences=True, activation=chosen_activation))
    #         if chosen_dropout > 0.0:
    #             model.add(layers.Dropout(chosen_dropout))                
                
    # if chosen_flat == 'gap':
    #     model.add(layers.GlobalAveragePooling1D())
    # else:
    #     model.add(layers.GlobalMaxPooling1D())
    # # else:
    # #     model.add(layers.Flatten())        
        
    # model.add(layers.Dense(10, activation='softmax'))

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
          f"dropout: {chosen_dropout}, optimizer: {chosen_optimizer}, flatten: {chosen_flat}")
    return model, train_ds, val_ds, X_test, y_test, train_steps, val_steps, y_train

# Hyperparameters
batch_size = 1024
epochs = 5
wl = 10
lr_rate = 1e-3

# Get data
scaler = StandardScaler()
num_samples = 100_000
data = get_base_data(num_samples)    

# Create and train an ensemble of models (e.g., 100 models)
n_models = 100
seeds = [random.randint(0, 10000) for _ in range(n_models)]
ensemble_models = []
val_accuracies = []

for i, seed in enumerate(seeds):
    tf.keras.utils.clear_session(free_memory=True)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
    model, train_ds, val_ds, X_test, y_test, train_steps, val_steps, y_train = create_random_model(seed, data)
    print(f"\nTraining model {i+1}/{n_models}")
    callback = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                         patience=10, verbose=1, mode='auto',
                                         baseline=None, restore_best_weights=True,
                                         start_from_epoch=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                             patience=2, verbose=0, mode='auto',
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