<h1>Multi Branch Hybrid Time Series Classification with HMM+NVG Features</h1>

<p>
This project implements a dual branch deep learning pipeline for short window
time series classification. The approach combines engineered features derived
from Hidden Markov Models (HMMs) and Natural Visibility Graphs (NVG) with
raw time series, processed by a hybrid Transformer + CNN architecture. An
ensemble of multiple seeded models is trained and weighted to improve
robustness and exploit diversity.
</p>

<h2>Key Components</h2>

<ul>
  <li><strong>Time Series Augmentation:</strong> Adds Gaussian noise and small time warping to increase data variability.</li>
  <li><strong>Feature Engineering:</strong> Generates additional features using:
    <ul>
      <li>Gaussian, Categorical, GMM, and Poisson HMM predictions</li>
      <li>Natural Visibility Graph metrics (average degree, clustering coefficient)</li>
    </ul>
  </li>
  <li><strong>Dual-Branch Model Architecture:</strong>
    <ul>
      <li>Engineered branch: Transformer-inspired blocks with Time2Vec positional encoding</li>
      <li>Raw branch: 1D CNN layers to capture local temporal patterns</li>
      <li>Final concatenation and dense layers for classification</li>
    </ul>
  </li>
  <li><strong>Ensemble Strategy:</strong> Train multiple models with different random seeds, compute dynamic weights based on validation accuracy, and select diverse subsets using Q-statistics and double-fault analysis.</li>
</ul>

<h2>Data Pipeline</h2>
<p>
The input pipeline produces dual-windowed datasets aligned across:
</p>
<ul>
  <li><em>engineered features</em> from HMM+NVG transformations</li>
  <li><em>raw time series</em> values</li>
</ul>
<p>
Targets are defined as the next time step after each window. The pipeline
supports shuffling, batching, and prefetching for optimal GPU utilization.
</p>

<h2>Hybrid Model Details</h2>
<ul>
  <li><strong>Engineered branch:</strong> Dense projection + Time2Vec encoding → Multi-head attention → Feedforward → GlobalAveragePooling1D</li>
  <li><strong>Raw branch:</strong> Stacked 1D convolutions → GlobalAveragePooling1D</li>
  <li><strong>Fusion:</strong> Concatenate engineered and raw outputs → Dense → Dropout → Softmax classification</li>
  <li><strong>Optimization:</strong> AdamW with CosineDecayRestarts learning rate schedule, mixed precision for speed</li>
</ul>

<h2>Ensemble & Evaluation</h2>
<p>
Multiple models are trained independently with different seeds. Ensemble predictions are weighted by validation accuracy. Model diversity is encouraged via pairwise error statistics:
</p>
<ul>
  <li>Q-statistics matrix</li>
  <li>Double-fault matrix</li>
  <li>Diverse subsets selected by thresholding correlations</li>
</ul>
<p>
Alternative ensemble selection can use weights above uniform threshold.
Performance metrics are computed on a held-out test set.
</p>

<h2>Why This Approach?</h2>
<ul>
  <li>Short, noisy time series benefit from engineered representations that encode temporal structure (HMM, NVG).</li>
  <li>Dual-branch architectures let the model leverage both high-level engineered features and raw local patterns.</li>
  <li>Ensemble methods improve generalization and allow robust predictions in noisy environments.</li>
  <li>Dynamic ensemble weighting and diversity selection exploit complementary strengths of individual models.</li>
</ul>

<h2>Usage</h2>
<p>
1. Load raw data from CSV and generate engineered features using HMM+NVG.<br>
2. Create tf.data.Datasets for training, validation, and testing.<br>
3. Initialize and train dual-branch models with seeds to form an ensemble.<br>
4. Compute ensemble weights and select diverse subsets using Q-statistics.<br>
5. Evaluate ensemble accuracy on the test set.
</p>

<blockquote>
Combining engineered temporal representations with raw sequences and diverse ensembles allows robust short-window time series prediction beyond conventional CNN/RNN architectures.
</blockquote>
