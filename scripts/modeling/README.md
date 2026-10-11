# MRI-QRF modeling scripts — revised and legacy workflows

The repository contains **both revised 128×128 calibrated-evaluation scripts and older experiment code**. Choose the appropriate file; do not interchange legacy and revised outputs.

## Revised 128×128 classical ML evaluation

- [run_calibrated_ml_binary_128.py](run_calibrated_ml_binary_128.py): binary CDR=0 vs CDR>0 evaluation.
- [run_calibrated_ml_multiclass_128.py](run_calibrated_ml_multiclass_128.py): four-class CDR-based evaluation.
- [merge_ml_subject_predictions.py](merge_ml_subject_predictions.py): merges saved subject-prediction CSVs.

The two evaluation scripts **load pre-trained RandomForest and LinearSVM joblib models** for each dataset/seed, and report clean plus 15 calibrated conditions. They do not perform training or generate distortions. Default local paths reference `D:\\OASIS_1`, `D:\\OASIS_2` and `D:\\MRI-QRF_calibrated`. The expected distorted test directory is `distortions_128_calibrated`.

Outputs include slice- and subject-level predictions/metrics, ARR summaries and a DRI averaged across the 15 conditions per dataset/model/seed. The merger expects `subject_predictions/<dataset>/<model>/seed_<seed>/*.csv`.

## Revised 128×128 deep learning

- [run_dl_binary_calibrated.py](run_dl_binary_calibrated.py): binary classification.
- [run_dl_multiclass_calibrated.py](run_dl_multiclass_calibrated.py): four-class classification.

Both scripts **train and evaluate** SimpleCNN and ResNet18, with clean training/validation images and calibrated held-out test images. Settings include three training seeds (13/47/101), batch 32, at most 20 epochs, early stopping patience 5, class-weighted cross-entropy, and checkpoint selection by clean-validation Macro-F1. They produce local `best_model.pt` along with slice/subject predictions and metrics. Default paths use `/workspace/OASIS_1`, `/workspace/OASIS_2` and `/workspace/MRI_QRF/results/` and assume a CUDA GPU.

**Neither DL script performs SSIM calibration**: run the training-set-only calibration scripts and frozen-parameter test generator first. The script names do not encode SSIM target values; the matching distortion folder and parameter CSV must correspond to **0.95/0.85/0.70**.

## Older scripts — not the revised main analysis

- `run_ml_models_mri_qrf.py` and `subject_level_analysis_mri_qrf.py`: earlier classical workflows, including 32/64-resolution sensitivity runs.
- `run_dl_models_mri_qrf.py`: earlier DL setup referencing the uncalibrated `distortions_128` layout.

Do not use the older fixed-distortion outputs in the revised main comparison. The final study's 2,000-replicate subject bootstrap and GEE inference also require separate statistical code; older GLMM tools are not replacements.

## Reproduction limitations

Original/derived OASIS images, trained classical model binaries, DL checkpoint files, the final frozen calibration CSV, and verified final GEE outputs are **not shipped with the repository**. Edit the study-specific path constants for your machine, check the cohort/sample counts, and verify outputs against the manuscript before claiming exact numerical reproduction. No model-training/evaluation run was executed as part of this README revision.
