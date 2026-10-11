# Trained model checkpoints — optional external artifacts

**No trained checkpoint upload is required for this code repository.** Large `.pt`, `.pth`, and `.joblib` files are deliberately not bundled. Please do not upload the full trained-model directory solely to make the README's examples appear complete.

Revised main experiments use **128×128** inputs and separate binary/four-class tasks, with training seeds **13, 47 and 101**.

- Classical models: Random Forest and Linear SVM; the revised `run_calibrated_ml_binary_128.py` and `run_calibrated_ml_multiclass_128.py` scripts **load pre-existing local `.joblib` models**. They do *not* train them. The binaries therefore must be created using the corresponding clean training setup before these evaluation scripts can run.
- Deep models: SimpleCNN and ResNet18 trained from scratch via `scripts/modeling/run_dl_binary_calibrated.py` and `run_dl_multiclass_calibrated.py`; these scripts save `best_model.pt` locally under their configured output paths. Validation Macro-F1 selects the checkpoint.

Old `ML_32`, `ML_64`, or uncalibrated `DL_128` artifacts should not be labeled as the revised primary 128×128 experiment.

**Optional public artifact strategy:** retain the source and model configurations on GitHub; publish numerical results and reproducibility information; if checkpoints must be shared, place them in a suitable external archival store, include stable links, checksums, task/dataset/model/seed metadata, and respect storage and licensing terms. Git LFS is an option only when explicitly chosen; it is not a requirement.

Model checkpoints alone do not reproduce the manuscript without the same subject splits, clean preprocessing, frozen calibration parameters, and analysis scripts. Never bundle source or derived OASIS images without permission.
