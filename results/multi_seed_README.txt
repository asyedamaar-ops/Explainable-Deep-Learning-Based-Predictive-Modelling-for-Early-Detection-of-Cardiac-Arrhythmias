By-beat split against patient-wise split, on identical data (5 seeds)
Notebook: inter_patient_multi_seed.ipynb
Run on Kaggle, 15 November 2026, GPU T4 x2, 7106 s for all 30 trainings.
Seeds 0 to 4. Figures copied from the Kaggle run log and its summary files.

Design: everything fixed except the split.
  - the same beats, cut from the raw PhysioNet records (4 AAMI classes, paced recordings excluded)
  - the same three models, augmentation, class weights and training settings
  - the same train / validation / test sizes

                   Train                       Validation             Test
  patient-wise     17 DS1 recordings           5 other DS1 recordings 22 DS2 recordings
  by-beat          random beats, all 44 recordings, same three sizes

Beat counts (patient-wise): train 38,969 | validation 12,020 | test 49,678
  Normal            35,264   10,579   44,232
  Supraventricular     835      108    1,837
  Ventricular        2,475    1,313    3,221
  Fusion               395       20      388

Mean +/- standard deviation over the 5 seeds:

                        by-beat                     patient-wise
  Model                 accuracy      macro F1      accuracy      macro F1
  CNN baseline          0.9849+/-0.0009  0.9034+/-0.0061  0.8202+/-0.0226  0.4167+/-0.0257
  DenseNet              0.9829+/-0.0032  0.8956+/-0.0143  0.8230+/-0.0247  0.3824+/-0.0224
  DenseNet + attention  0.9833+/-0.0031  0.8940+/-0.0128  0.8617+/-0.0315  0.4348+/-0.0195

Validation accuracy reached during training (mean):
  by-beat      0.9853 / 0.9837 / 0.9834
  patient-wise 0.9379 / 0.9499 / 0.9476   (on 5 held-out DS1 recordings)

Per-class F1 (mean over seeds) is in multi_seed_per_class.csv.
Every individual run is in multi_seed_all_runs.csv.
