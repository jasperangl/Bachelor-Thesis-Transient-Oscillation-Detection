# 📊 Evaluating and Analyzing LSTM Models for Transient Oscillation Detection

This repository accompanies a bachelor thesis project that explores how Long Short-Term Memory (LSTM) models can be used to detect transient neural oscillations—bursts of activity in EEG-like time-series data. The work emphasizes model performance across features, noise conditions, and architecture variants, and introduces interpretability analyses on LSTM internals.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Data Generation & Processing](#-data-generation--processing)
- [Feature Engineering](#-feature-engineering)
- [Model Training](#-model-training)
- [Evaluation & Visualization](#-evaluation--visualization)
- [Interpretability](#-interpretability)
- [Key Findings](#-key-findings)

---

## 🧠 Overview

Neural bursts are brief, context-sensitive oscillations essential to brain function. Traditional detection techniques struggle under noise or lack adaptability. This project explores the use of LSTM networks—particularly their architecture, depth, and input representation—to robustly detect these bursts in a synthetic, labeled dataset.

---

## 📁 Project Structure

```
├── 1. Simulate_bursts.ipynb               # Synthetic dataset generation
├── 2a. Feature Extractor.ipynb            # Hilbert and wavelet feature computation
├── 2b. Feature Visualizations.ipynb       # Correlation analysis & visualizations
├── 3a. LSTM Cell Analysis_ Model Generation.ipynb # Model training and hidden state extraction
├── 4.0 LSTM Set-Up Comparison.ipynb       # Architecture comparisons and benchmarking
├── 4.2a Cross-Model Performance Analysis.ipynb # Aggregated model evaluation
├── analysis_utils.py                      # Analysis utilities for hidden states & weights
├── data_utils.py                          # Data preprocessing, training, and helper functions
├── 3b. LSTM Cell Analysis_ Test Data Generation.ipynb     # Generates testing data for LSTM cell analysis
├── 3c.1 LSTM Cell Analysis_ Binary Visuals.ipynb           # Visualizes binary model hidden states and outputs
├── 3c2. LSTM Cell Analysis_ Multi Visuals.ipynb            # Visualizes multiclass model hidden states and outputs
├── 4.1a Cross-Model Comparisons.ipynb                      # RNN, BiLSTM, and MLP comparisons
├── 4.1b Intra-LSTM Comparisons.ipynb                       # Comparison of LSTM variants within single architecture
├── 4.2b LSTM Performance Analysis.ipynb                    # Detailed metric-based breakdown of LSTM performance
├── Colloquium New.pptx                    # Thesis presentation slides
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow/Keras
- NumPy, SciPy, Matplotlib
- Scikit-learn

### Installation

Clone the repository:

```bash
git clone https://github.com/jasperangl/Bachelor-Thesis-Transient-Oscillation-Detection.git
cd lstm-burst-detection
```

---

## 🧪 Data Generation & Processing

**Notebook:** `1. Simulate_bursts.ipynb`  
Generates synthetic LFP-like signals containing bursts across theta, alpha, beta, and gamma bands. Data is balanced and labeled per time point with controlled SNR levels.

### Utilities
`data_utils.py` handles:
- Flexible slicing based on frequency/noise/sample count
- Reshaping and one-hot encoding
- Persistent storage for training/testing splits

---

## 🛠 Feature Engineering

**Notebook:** `2a. Feature Extractor.ipynb`  
Extracts two types of features:
- **Hilbert Amplitudes** (low temporal precision, high amplitude fidelity)
- **Morlet Wavelets** (high temporal/spectral resolution)

**Notebook:** `2b. Feature Visualizations.ipynb`  
Performs correlation analysis to assess informativeness of features versus raw signals.

---

## 🧬 Model Training

**Notebook:** `3a. LSTM Cell Analysis_ Model Generation.ipynb`  
Trains LSTM-based models on different input types (signal, features, combinations).  

Key functionalities:
- Multi-epoch saving with custom callbacks
- Performance tracking and model persistence
- Simple, Deep, and BiLSTM variants
- Binary and multi-class settings

---

## 📈 Evaluation & Visualization

**Notebook:** `4.0 LSTM Set-Up Comparison.ipynb`  
Compares model variants by:
- Input types
- Model complexity (~5K parameters)
- Binary vs. multiclass setups

**Notebook:** `4.2a Cross-Model Performance Analysis.ipynb`  
Benchmarks RNNs, BiLSTMs, and MLPs using standardized metrics (F1, Accuracy, MCC, Recall).

---

## 🔍 Interpretability

**File:** `analysis_utils.py`  
Provides detailed visualizations and ranking:
- Weighted LSTM hidden activations
- Cell-level contributions over time
- Visual differentiation of "good" vs "bad" cells
- Epoch-wise model predictions and hidden state tracking

---

## 🧾 Key Findings

From the thesis and `Colloquium New.pptx` presentation:

- **Feature importance:** Hilbert & wavelet features boost simpler models; deep LSTMs learn from raw signals.
- **Architecture matters:** BiLSTMs outperform LSTM and MLP variants in noisy or feature-poor setups.
- **Cell specialization:** Specific cells adapt to signal type, frequency, or noise level.
- **Gating behavior:** Cell states and gates reflect structured internal filtering aligned with signal semantics.

---
