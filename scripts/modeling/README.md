# Modeling scripts

These scripts preserve the study implementation used during the experiment.

## Classical scripts
`run_ml_models_mri_qrf.py` and `subject_level_analysis_mri_qrf.py` were executed on the local Windows study layout and therefore contain the original `D:\...` defaults.

For a new machine, edit only the path constants and, for the classical runner, the `RESOLUTION` / `OUTPUT_ROOT` pair. Do not change model hyperparameters when reproducing the fixed experiment.

Classical runs:
- 32 → `ML_32`
- 64 → `ML_64`
- 128 → `ML_128`

## Deep-learning script
`run_dl_models_mri_qrf.py` preserves the RunPod `/workspace` layout used for the DL experiment.

Expected:
```text
/workspace/OASIS_1/clean_128
/workspace/OASIS_1/distortions_128
/workspace/OASIS_2/clean_128
/workspace/OASIS_2/distortions_128
/workspace/MRI_QRF
```

The DL script correctly allows OASIS-1 validation to contain zero Moderate Dementia subjects.
