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

### 1. Data Simulation & Preprocessing
- `1. Simulate_bursts.ipynb` — Generates a synthetic EEG-like dataset with bursts across frequency bands and SNR levels.
- `data_utils.py` — Utility functions for preprocessing, feature shaping, train/test splitting, model training, and evaluation.

### 2. Feature Engineering & Visualization
- `2a. Feature Extractor.ipynb` — Computes Hilbert and Morlet wavelet features for each sample.
- `2b. Feature Visualizations.ipynb` — Correlates features with ground truth and provides visual inspection of signal vs. derived features.

### 3. LSTM Training & Hidden State Analysis
- `3a. LSTM Cell Analysis_ Model Generation.ipynb` — Trains various LSTM architectures while saving intermediate models for interpretability.
- `3b. LSTM Cell Analysis_ Test Data Generation.ipynb` — Uses trained models to generate test predictions and extract hidden/cell states over epochs.
- `3c.1 LSTM Cell Analysis_ Binary Visuals.ipynb` — Visualizes hidden activations for binary LSTM models across top/bottom contributing cells.
- `3c2. LSTM Cell Analysis_ Multi Visuals.ipynb` — Similar visualizations for multi-class LSTM outputs across multiple features.

### 4. Model Comparison & Performance Benchmarking
- `4.0 LSTM Set-Up Comparison.ipynb` — Baseline comparison of simple vs deep LSTM setups using same input space.
- `4.1a Cross-Model Comparisons.ipynb` — Trains RNNs, BiLSTMs, MLPs, and compares them under equivalent parameter counts.
- `4.1b Intra-LSTM Comparisons.ipynb` — Trains different LSTM and BiLSTM variants depths.
- `4.2a Cross-Model Performance Analysis.ipynb` — Consolidates performance metrics (MCC, Accuracy, Inference Times) across all model types.
- `4.2b LSTM Performance Analysis.ipynb` — Analyzes LSTM and BiLSTM models on their performance in respect to their complexity.

### 🔍 Utilities & Presentation
- `analysis_utils.py` — Contains ranking, visualization, and interpretability tools for hidden states and dense layer activations.
- `Colloquium New.pptx` — Final thesis presentation summarizing motivations, methods, results, and interpretability findings.
- `README.md` — This documentation file.

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow/Keras
- NumPy, SciPy, Matplotlib
- Scikit-learn

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
