# Explainable Cardiac Arrhythmia Detection

**Three 1D deep-learning models compared on 109,446 MIT-BIH heartbeats, with Grad-CAM to show where the model looks, and an honest account of what the numbers do not prove.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Data-MIT--BIH-0B7285?style=flat-square)

Five-class heartbeat classification on the MIT-BIH arrhythmia data (Kaggle `shayanfazeli/heartbeat`). A CNN, a DenseNet and a DenseNet with multi-scale attention are trained and compared over two runs, and Grad-CAM is used to look at where a model attends along the beat.

Headline result: with the usual by-beat split, all three models reach 97.9 to 98.7% accuracy and 0.90 to 0.92 macro F1. Holding the data, models and settings fixed and changing only the split to patient-wise drops macro F1 to 0.38 to 0.43 (5 seeds). Most of the headline figure therefore comes from having the same patients in train and test, and the larger models are not clearly better than the plain CNN in either setting.

![Sample beats of the five classes](figures/sample_beats.png)

## Project and my contribution

Minor project at SRM Institute of Science and Technology (January to May 2026), done in a team of 2 with a faculty mentor.

My part:

- **Data:** dataset handling and preprocessing, including the class-imbalance handling.
- **Training pipeline:** training and debugging the models.
- **Explainability:** the Grad-CAM visualisation and region analysis.

## Dataset

- 87,554 training beats and 21,892 test beats, 187 samples each. The signal is resampled to 125 Hz, so a beat window is about 1.5 s. Each beat starts at its R-peak and is zero-padded, so there is no P-wave in the window.
- Five AAMI classes: Normal, Supraventricular, Ventricular, Fusion, Unknown.
- Severe imbalance in training: Normal 72,471 beats (82.8%) and Fusion 641 (0.7%), about 113 to 1.

## What was done

- **Imbalance handling:** time-shift, amplitude scaling and Gaussian-noise augmentation on the minority classes, plus balanced class weights in the loss.
- **Models:**
  - Baseline 1D CNN with SpatialDropout1D (224,901 parameters)
  - DenseNet-style 1D network with transition layers (318,501 parameters)
  - The same DenseNet with two multi-head self-attention blocks at different temporal scales (442,917 parameters)
- **Evaluation:** accuracy, macro F1, and per-class sensitivity, specificity and PPV. A clinical priority score (Ventricular, then Supraventricular, then Fusion, then accuracy) is used to pick the model for Grad-CAM.
- **Explainability:** Grad-CAM for 1D signals, averaged per approximate ECG region over correctly classified test beats.

## Results (test set, 21,892 beats)

The whole pipeline was run twice with different random initialisation. Both runs are shown, because the ranking changed.

| Model | Accuracy (run 1 / run 2) | Macro F1 (run 1 / run 2) | Fusion F1 (run 1 / run 2) |
|---|---|---|---|
| CNN baseline | 98.19% / 97.93% | 0.910 / 0.903 | 0.814 / 0.808 |
| DenseNet with transition layers | 98.66% / 98.34% | 0.923 / 0.911 | 0.807 / 0.788 |
| DenseNet + multi-scale attention | 98.44% / 98.46% | 0.915 / 0.909 | 0.788 / 0.771 |

![F1 score per class for the three models, run 2](figures/model_comparison.png)

What this shows:
- The DenseNet had the best macro F1 in both runs, by about 0.7 to 1.3 points over the CNN.
- The attention model had the best accuracy in run 2 only. Across both runs it never clearly beat the plain DenseNet, and it has the most parameters.
- The simple CNN was best on the rare Fusion class in both runs.
- The accuracy gaps (0.1 to 0.7 points) are about the size of the change between runs, so these models are not clearly separable on accuracy.

Run 2 per-class sensitivity and PPV (Supraventricular and Ventricular are the high-risk classes):

| Model | SVT sensitivity / PPV | Ventricular sensitivity / PPV | Fusion sensitivity / PPV |
|---|---|---|---|
| CNN baseline | 0.845 / 0.712 | 0.974 / 0.939 | 0.846 / 0.774 |
| DenseNet | 0.845 / 0.812 | 0.969 / 0.948 | 0.827 / 0.753 |
| DenseNet + attention | 0.822 / 0.840 | 0.963 / 0.953 | 0.821 / 0.727 |

In run 2 the clinical priority score selected the **CNN baseline** (0.916) for Grad-CAM, ahead of the DenseNet (0.911) and the attention model (0.900).

## Patient-wise evaluation: how much does the split matter?

The results above use a by-beat split, where beats of the same patient appear in both train and test. To measure what that is worth, I re-ran the whole comparison with **everything held fixed except the split**: the same beats (cut from the raw PhysioNet records, 4 AAMI classes, paced recordings excluded), the same three models, the same augmentation, class weights and training settings, and the same train / validation / test sizes.

| | Train | Validation | Test |
|---|---|---|---|
| Patient-wise | 17 DS1 recordings | 5 other DS1 recordings | 22 DS2 recordings |
| By-beat | random beats from all 44 recordings | random beats | random beats |

Each setting was run with 5 seeds. Notebook: `inter_patient_multi_seed.ipynb`. Full numbers: `results/multi_seed_README.txt`, `results/multi_seed_all_runs.csv`, `results/multi_seed_per_class.csv`.

**Mean ± standard deviation over 5 seeds**

| Model | By-beat accuracy | Patient-wise accuracy | By-beat macro F1 | Patient-wise macro F1 |
|---|---|---|---|---|
| CNN baseline | 0.9849 ± 0.0009 | 0.8202 ± 0.0226 | 0.903 ± 0.006 | 0.417 ± 0.026 |
| DenseNet | 0.9829 ± 0.0032 | 0.8230 ± 0.0247 | 0.896 ± 0.014 | 0.382 ± 0.022 |
| DenseNet + attention | 0.9833 ± 0.0031 | 0.8617 ± 0.0315 | 0.894 ± 0.013 | 0.435 ± 0.020 |

**F1 per class (mean over 5 seeds)**

| Class | Test beats | By-beat | Patient-wise |
|---|---|---|---|
| Normal | 44,232 | 0.99 | 0.90 to 0.92 |
| Ventricular | 3,221 | 0.96 to 0.97 | 0.57 to 0.69 |
| Supraventricular | 1,837 | 0.81 to 0.83 | 0.06 to 0.12 |
| Fusion | 388 | 0.81 to 0.82 | 0.00 to 0.04 |

Ranges are across the three models.

**What this shows**

- **Changing only the split drops macro F1 from about 0.90 to about 0.41.** The gap of roughly 0.49 is far larger than the spread between seeds (0.01 to 0.03), so it is not noise.
- **Most of the 98% headline figure comes from having the same patients on both sides.** The model partly recognises a patient's beat shape rather than the arrhythmia.
- **Ventricular beats still transfer** to unseen patients (recall about 0.85), but with many false alarms (precision 0.43 to 0.58).
- **Supraventricular and Fusion beats essentially do not transfer** (F1 below 0.12).
- **Validation does not warn you.** Validation accuracy stayed at 94 to 95% even on 5 held-out recordings, while test accuracy on the 22 DS2 recordings was 82 to 86%.
- **No model is clearly better.** The attention model leads patient-wise by 0.02 to 0.05 macro F1, which is one to two standard deviations.

**Limits of this experiment**

- Fusion has only 20 validation beats and 388 test beats, so its scores are unstable.
- A single-beat window carries no rhythm timing, and supraventricular beats are mostly recognised by arriving early. Part of that weak score may come from the input rather than the split. This was not tested. Adding R-R interval features is the obvious next step.
- A single earlier run with validation taken from the training patients is kept in `inter_patient_evaluation.ipynb` and `results/inter_patient_run1.txt`.

## Grad-CAM attention by region (run 2, CNN baseline)

Mean Grad-CAM attention (%) per approximate region, averaged over up to 200 correctly classified test beats per class. Beats with an empty heatmap are excluded.

| Class | QRS (0-12) | ST-segment (12-28) | T-wave (28-65) | Late beat (65+) | Beats used |
|---|---|---|---|---|---|
| Normal | 49.7 | 11.9 | 14.3 | 24.1 | 175 / 200 |
| Supraventricular | 9.8 | 27.4 | 30.3 | 32.4 | 151 / 200 |
| Ventricular | 30.8 | 11.6 | 23.9 | 22.6 | 172 / 200 |
| Fusion | 5.2 | 52.3 | 24.2 | 16.7 | 134 / 137 |
| Unknown | 13.6 | 33.2 | 16.0 | 17.2 | 185 / 200 |

![Grad-CAM example beats](figures/gradcam_examples.png)

How to read it:
- Normal beats are dominated by the QRS region. Fusion beats put most attention on the ST segment.
- Ventricular attention is spread across the QRS, T-wave and late-beat regions, and Supraventricular attention is diffuse. These do not give a clean clinical story, so this table should not be read as proof that the model uses clinically correct features.
- The region boundaries are approximate. They are read off the class-mean beats, not clinical annotations. The 125 Hz, R-peak-first layout means there is no P-wave to analyse.

## Limitations

- **The split is by beat, not by patient.** The Kaggle train and test files mix beats from the same recordings, so a patient's beats can appear on both sides. This probably inflates the accuracy. A patient-wise split (for example the DS1/DS2 protocol) is the proper test.
- **Why supraventricular beats fail on unseen patients is untested.** See the section above. Adding R-R interval features is the next step.
- **Only two runs per model.** That is enough to see that the ranking is unstable, but not to put confidence intervals on the differences.
- **Grad-CAM regions are approximate** (see above), and the Grad-CAM table is for one selected model in one run.
- **An early version of the region analysis was wrong.** It used boundaries meant for a window centred on the R-peak and a single beat per class. Both were fixed in the final notebook.
- **No ablation.** An earlier ablation attempt crashed and was removed, so no ablation results are claimed.
- **Research code, not a diagnostic tool.**

## Training curves

![Training and validation curves](figures/training_curves.png)

## Run it

1. Open `cardiac_arrhythmia_classification.ipynb` in Kaggle with a GPU and add the dataset `shayanfazeli/heartbeat`.
2. Run all cells. Training the three models takes the longest.

The main notebook is saved with its outputs (run 2), so its numbers and figures can be checked without running it. The two patient-wise notebooks are saved without outputs, and their logs and result tables are in `results/`. To repeat them, run on Kaggle with the GPU and Internet on, because the records are downloaded from PhysioNet. To run it outside Kaggle, install `requirements.txt` and change the two CSV paths in the data-loading cell.

## Tech stack

TensorFlow/Keras, scikit-learn, NumPy, pandas, Matplotlib, Seaborn.

## References

- Kachuee, Fazeli, Sarrafzadeh. ECG Heartbeat Classification: A Deep Transferable Representation (2018), the source of the dataset.
- Huang et al. Densely Connected Convolutional Networks (CVPR 2017).
- Selvaraju et al. Grad-CAM (ICCV 2017).
