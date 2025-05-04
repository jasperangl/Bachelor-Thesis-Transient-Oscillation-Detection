# 📊 Evaluating and Analyzing LSTM Models for Transient Oscillation Detection

This repository accompanies a bachelor thesis project that explores how Long Short-Term Memory (LSTM) models can be used to detect transient neural oscillations—bursts of activity in EEG-like time-series data. The work emphasizes model performance across features, noise conditions, and architecture variants, and introduces interpretability analyses on LSTM internals.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Key Findings](#-key-findings)

---

## 🧠 Overview

This thesis evaluated LSTM-based models for detecting transient neural oscillations across frequency bands and noise levels, with BiLSTMs achieving the most robust performance, though detection in the theta band remained challenging under low SNR. A comparison of input representations showed that computed features benefited simpler models, while deeper recurrent networks performed well on raw input alone. Under matched model complexity, LSTM variants consistently outperformed MLPs and threshold-based baselines. Interpretability analyses revealed that gate and state dynamics adapted to encode burst structure, suggesting the emergence of functionally specialized units within LSTM layers.

---

## 📁 Project Structure

### 1. Data Simulation & Preprocessing
- `1. Simulate_bursts.ipynb` — Generates a synthetic EEG-like dataset with bursts across frequency bands and SNR levels.

### 2. Feature Engineering & Visualization
- `2a. Feature Extractor.ipynb` — Computes Hilbert and Morlet wavelet features for each sample.
- `2b. Feature Visualizations.ipynb` — Correlates features with ground truth and provides visual inspection of signal vs. derived features.

### 3. LSTM Training & Hidden State Analysis
- `3a. LSTM Cell Analysis_ Model Generation.ipynb` — Trains various LSTM architectures while saving intermediate models for interpretability.
- `3a2. LSTM Gate Analysis_ Model Generation and Visualization` — Trains various LSTM architectures while saving all gate and state behavior and intermediate models for epoch analysis.
- `3b. LSTM Cell Analysis_ Test Data Generation.ipynb` — Uses trained models to generate test predictions and extract hidden/cell states over epochs.
- `3c.1 LSTM Cell Analysis_ Binary Visuals.ipynb` — Visualizes hidden activations for binary LSTM models across top/bottom contributing cells.
- `3c2. LSTM Cell Analysis_ Multi Visuals.ipynb` — Similar visualizations for multi-class LSTM outputs across multiple features.

### 4. Model Comparison & Performance Benchmarking
- `4.0 LSTM Set-Up Comparison.ipynb` — Compares simple LSTM models on different batch sizes and sequence lengths.
- `4.1a Cross-Model Comparisons.ipynb` — Trains RNNs, BiLSTMs, MLPs, and compares them under equivalent parameter counts.
- `4.1b Intra-LSTM Comparisons.ipynb` — Trains different LSTM and BiLSTM variants depths.
- `4.2a Cross-Model Performance Analysis.ipynb` — Consolidates performance metrics (MCC, Accuracy, Inference Times) across all model types.
- `4.2b LSTM Performance Analysis.ipynb` — Analyzes LSTM and BiLSTM models on their performance in respect to their complexity.

### Utilities & Presentation
- `data_utils.py` — Utility functions for preprocessing, feature shaping, train/test splitting, model training, and evaluation.
- `analysis_utils.py` — Contains ranking, visualization, and interpretability tools for hidden states and dense layer activations.
- `Colloquium New.pptx` — Final thesis presentation summarizing motivations, methods, results, and interpretability findings.
- `README.md` — This documentation file.

---


## 🧾 Key Findings

From the thesis and `Colloquium New.pptx` presentation:

- **Feature importance:** Hilbert & wavelet features boost simpler models; deep LSTMs learn from raw signals.
- **Architecture matters:** BiLSTMs outperform LSTM and MLP variants in noisy or feature-poor setups.
- **Cell specialization:** Specific cells adapt to signal type, frequency, or noise level.
- **Gating behavior:** Cell states and gates reflect structured internal filtering aligned with signal semantics.

---
