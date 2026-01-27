import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Global configuration
GRID_HEIGHT = 32
GRID_WIDTH = 32
STATE_DIM = 8
NEIGHBORHOOD_SIZE = 3


def state_to_rgb(grid):
    grid = (grid - tf.reduce_min(grid)) / (tf.reduce_max(grid) - tf.reduce_min(grid) + 1e-8)
    rgb = grid[..., :3]  # Use first 3 channels for RGB
    return rgb.numpy()

def plot_grid(grid, step=0):
    rgb = state_to_rgb(grid[0])
    plt.figure(figsize=(4, 4))
    plt.imshow(rgb)
    plt.title(f"Universe at step {step}")
    plt.axis('off')
    plt.show()


def extract_patch(grid, radius=2):
    return tf.image.extract_patches(
        images=grid,
        sizes=[1, 2*radius+1, 2*radius+1, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='SAME'
    )


class ModulatedProtoUniverse(tf.keras.Model):
    def __init__(self, height, width, state_dim, neighborhood_size=3, mod_dim=16):
        super().__init__()
        self.height = height
        self.width = width
        self.state_dim = state_dim
        self.kernel_size = neighborhood_size
        self.mod_dim = mod_dim

        self.perception_layer = layers.Conv2D(64, self.kernel_size, padding='same', activation='relu')
        self.rule_core = layers.Conv2D(state_dim, 1, activation='tanh')

        self.meta_network = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(mod_dim, activation='relu'),
            layers.Dense(64, activation='sigmoid')
        ])

    def call(self, grid):
        mod = self.meta_network(grid)
        mod = tf.reshape(mod, (-1, 1, 1, 64))
        x = self.perception_layer(grid)
        x = x * mod
        delta = self.rule_core(x)
        return grid + delta


class TimeHopperObserver(tf.keras.Model):
    def __init__(self, state_dim, memory_dim=64, buffer_size=20):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='relu')
        ])
        self.rnn_cell = layers.LSTMCell(memory_dim)
        self.output_layer = layers.Dense(state_dim)

        self.memory_state = None
        self.state_buffer = []
        self.buffer_size = buffer_size

    def reset_memory(self, batch_size):
        self.memory_state = self.rnn_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

    def call(self, local_patch):
        encoded = self.encoder(local_patch)
        output, self.memory_state = self.rnn_cell(encoded, self.memory_state)
        prediction = self.output_layer(output)
        return prediction

    def add_to_buffer(self, grid):
        self.state_buffer.append(grid)
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)

    def jump_to_time(self, t):
        if 0 <= t < len(self.state_buffer):
            return self.state_buffer[t]
        else:
            return self.state_buffer[-1]


class Multiverse:
    def __init__(self, num_universes, universe_kwargs):
        self.num_universes = num_universes
        self.universes = [ModulatedProtoUniverse(**universe_kwargs) for _ in range(num_universes)]
        self.grids = [tf.random.uniform(
            (1, universe_kwargs['height'], universe_kwargs['width'], universe_kwargs['state_dim']),
            minval=-1, maxval=1) for _ in range(num_universes)]
        self.history = [[] for _ in range(num_universes)]
        self.observers = [TimeHopperObserver(universe_kwargs['state_dim']) for _ in range(num_universes)]
        self.predictions = [[] for _ in range(num_universes)]

    def step(self, t):
        for i in range(self.num_universes):
            grid = self.grids[i]
            universe = self.universes[i]
            observer = self.observers[i]

            new_grid = universe(grid)
            self.history[i].append(grid)
            self.grids[i] = new_grid

            patch = extract_patch(grid, radius=2)
            observer_pred = observer(patch)
            self.predictions[i].append(observer_pred)
            observer.add_to_buffer(grid)

    def visualize(self, step=0):
        for i, grid in enumerate(self.grids):
            print(f"Universe {i}")
            plot_grid(grid, step)


def compute_prediction_errors(multiverse):
    errors = []
    for i in range(multiverse.num_universes):
        preds = multiverse.predictions[i]
        truths = multiverse.history[i][1:]
        if len(preds) > 0:
            mse = tf.reduce_mean([tf.reduce_mean(tf.square(p - t)) for p, t in zip(preds, truths)])
            errors.append(mse.numpy())
    return errors


def simulate_multiverse(multiverse, steps=30, visualize_every=5):
    for t in range(steps):
        multiverse.step(t)
        if t % visualize_every == 0:
            multiverse.visualize(step=t)
        errors = compute_prediction_errors(multiverse)
        print(f"[Step {t}] Observer MSEs: {errors}")


if __name__ == "__main__":
    universe_kwargs = {
        'height': GRID_HEIGHT,
        'width': GRID_WIDTH,
        'state_dim': STATE_DIM,
        'neighborhood_size': NEIGHBORHOOD_SIZE
    }

    multiverse = Multiverse(num_universes=3, universe_kwargs=universe_kwargs)
    simulate_multiverse(multiverse, steps=30, visualize_every=5)
