# Explainable Deep Learning for Cardiac Arrhythmia Detection

Deep learning models for automated ECG arrhythmia classification, built around a custom DenseNet + Multi-Scale Multi-Head Attention architecture, with Grad-CAM-based explainability to make predictions clinically interpretable.

## Overview

This project tackles arrhythmia classification on the MIT-BIH Arrhythmia Database, where each ECG heartbeat is classified into one of five categories: Normal, Supraventricular, Ventricular, Fusion, and Unknown. Beyond just achieving high accuracy, the project focuses on two things that matter in a clinical setting: correctly catching dangerous arrhythmias (Ventricular and Supraventricular beats) even when they're rare in the data, and explaining *why* the model made a given prediction by highlighting which part of the ECG waveform it focused on.

## Dataset

- **Source**: MIT-BIH Arrhythmia Database (via Kaggle, `shayanfazeli/heartbeat`)
- **Format**: Each row is one heartbeat segment, 187 timesteps sampled at 360 Hz (~0.52 seconds), with an integer class label (0-4)
- **Classes**: Normal, Supraventricular, Ventricular, Fusion, Unknown
- **Challenge**: Severe class imbalance — the Normal class dominates with tens of thousands of samples, while Fusion has only a few hundred

## Approach

### 1. Data Preprocessing & Augmentation

Because the Fusion class is heavily underrepresented, ECG-aware data augmentation was applied to balance the training set:
- **Time-shift**: simulates natural alignment offsets in real ECG recordings
- **Amplitude scaling**: simulates variation from electrode placement
- **Gaussian noise**: simulates sensor noise present in real recordings

Class weights were also computed via scikit-learn so that minority classes contribute proportionally to the loss, and labels were one-hot encoded for categorical cross-entropy training.

### 2. Model Architectures

Three models were built and compared:

**Model 1 — Baseline CNN**: A simple 3-block 1D convolutional network used as a performance floor. Uses SpatialDropout1D (drops entire feature maps) for stronger regularization on sequential ECG data than standard Dropout.

**Model 2 — Improved DenseNet**: Adds DenseNet-style dense blocks, where each layer receives all previous feature maps (improving feature reuse and gradient flow), combined with transition layers between blocks. Without transition layers, channel counts would explode to ~450 after three dense blocks — transition layers compress this back down to a controlled 32 → 64 → 128 progression, following the DenseNet-BC design (Huang et al., 2017).

**Model 3 — Proposed: DenseNet + Multi-Scale Multi-Head Attention**: The core contribution of this project. Adds two multi-head self-attention blocks at *different temporal resolutions* within the network. A single late-stage attention block (as used in prior work) operates on only ~23 timesteps (~64ms at 360Hz) — too coarse to distinguish fine-grained morphological differences between arrhythmia types. By placing attention at multiple scales, the model can focus on both broad rhythm patterns and fine waveform details (e.g., QRS complex shape, P-wave morphology).

### 3. Evaluation Strategy

For an imbalanced clinical dataset, accuracy alone is misleading — the Normal class can dominate the metric while the model fails on dangerous minority classes. This project reports:

- **Macro F1**: averages F1 across all five classes equally, so minority-class performance isn't hidden
- **Sensitivity (Recall)**: how many dangerous beats (Ventricular, Supraventricular) were correctly caught — critical, since a missed Ventricular beat can be life-threatening
- **Specificity**: how often false alarms were avoided — important to prevent alarm fatigue in clinical settings
- **PPV (Precision)**: when the model flags a dangerous beat, how often it's correct
- **Clinical priority scoring**: a custom metric that weights Ventricular > SVT > Fusion > overall accuracy, used to select the final model — because the "best" model clinically isn't always the one with the highest raw accuracy

### 4. Explainability with Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM, Selvaraju et al., 2017) was adapted for 1D ECG signals to visualize which regions of each heartbeat the model focused on when making its prediction. This produces:

- Per-class heatmap overlays on the ECG waveform, annotated with clinical regions (P-wave, QRS complex, T-wave)
- A **quantified region-importance table** — for each arrhythmia class, the percentage of model attention falling on each ECG region. This is a novel contribution not previously reported for MIT-BIH classification
- **Misclassification analysis**: for each clinically critical class, a correct prediction and a misclassified one are compared side-by-side with Grad-CAM overlays, revealing *which ECG region* led the model astray

## Results

| Model | Accuracy | Macro F1 | Ventricular F1 | Fusion F1 | Parameters |
|---|---|---|---|---|---|
| CNN Baseline | 0.9819 | 0.9102 | 0.9610 | 0.8140 | 224,901 |
| DenseNet (Improved) | **0.9866** | **0.9234** | **0.9657** | 0.8073 | 318,501 |
| Proposed (DenseNet + Multi-Scale Attention) | 0.9844 | 0.9145 | 0.9594 | 0.7875 | 442,917 |

The Improved DenseNet achieved the best overall accuracy and Macro F1, while the CNN Baseline performed best on the rare Fusion class. Final model selection for Grad-CAM analysis used a **clinical priority score** (weighting Ventricular and SVT sensitivity above raw accuracy) rather than accuracy alone — reflecting how model selection should work in a real diagnostic pipeline.

### Clinical Metrics (Sensitivity / Specificity / PPV)

| Class | Risk Level | CNN Baseline (Sens / Spec / PPV) | DenseNet Improved (Sens / Spec / PPV) |
|---|---|---|---|
| Normal | Low | 0.987 / 0.972 / 0.994 | 0.993 / 0.970 / 0.994 |
| Supraventricular | High | 0.853 / 0.992 / 0.745 | 0.854 / 0.997 / 0.864 |
| Ventricular | High | 0.970 / 0.997 / 0.953 | 0.972 / 0.997 / 0.960 |
| Fusion | Medium | 0.864 / 0.998 / 0.769 | 0.815 / 0.999 / 0.800 |
| Unknown | Low | 0.991 / 0.999 / 0.989 | 0.990 / 1.000 / 0.993 |

## An Ablation Study

A separate ablation experiment compares the proposed architecture against a version without transition layers, isolating the effect of channel compression on parameter count and performance — included in the notebook for completeness.

## Tech Stack

- **TensorFlow / Keras** — model building and training (Conv1D, DenseNet blocks, MultiHeadAttention, BatchNormalization, SpatialDropout1D)
- **scikit-learn** — class weight computation, evaluation metrics, classification reports
- **NumPy / Pandas** — data handling
- **Matplotlib / Seaborn** — visualizations including training curves, confusion matrices, and Grad-CAM overlays
- **KaggleHub** — dataset access

## Repository Contents

- `explainable-deep-learning-based-predictive-modelli.ipynb` — full notebook covering data loading, preprocessing, all three model architectures, training, evaluation, Grad-CAM explainability, and ablation study

## How to Run

1. Open the notebook in Kaggle or Jupyter (a Kaggle environment is recommended since it uses `kagglehub` for dataset access and GPU acceleration for training)
2. Run cells sequentially — the notebook downloads the MIT-BIH dataset automatically via `kagglehub.dataset_download('shayanfazeli/heartbeat')`
3. Training each model (CNN, DenseNet, Proposed) takes the longest; results, comparison tables, and Grad-CAM visualizations are generated in later cells

## Key Takeaways

This project demonstrates an end-to-end deep learning pipeline for a real-world biomedical classification problem: handling severe class imbalance with domain-appropriate augmentation, designing and comparing multiple architectures with clear engineering rationale, evaluating models using clinically meaningful metrics rather than raw accuracy alone, and using explainability techniques (Grad-CAM) to validate that the model's predictions align with established medical knowledge about ECG morphology.
