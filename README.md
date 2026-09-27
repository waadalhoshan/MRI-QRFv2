# MRI-QRF: MRI Quality Robustness Framework

This repository is the replication package for the MRI Quality Robustness Framework (MRI-QRF), a controlled framework for measuring how Alzheimer’s disease (AD) MRI classifiers respond to image-quality degradation.

The study uses the official **OASIS-1 cross-sectional** and **OASIS-2 longitudinal** MRI datasets, four CDR-based classes, two classical machine-learning models, two deep-learning models, five distortion types at three severities, three model-training seeds, and a blinded expert-readability study.

> **Important data-use note:** Original or derived OASIS MRI images are **not redistributed** in this repository. Obtain OASIS data from the official OASIS project and comply with the applicable data-use agreement. This repository contains code, protocol definitions, documentation, result schemas, and upload locations for trained models and non-image results.

## Study at a glance

| Component | Fixed protocol |
|---|---|
| Datasets | OASIS-1, OASIS-2 |
| Classes | Non-Demented; Very Mild Dementia; Mild Dementia; Moderate Dementia |
| CDR mapping | 0, 0.5, 1, 2 |
| OASIS-1 selection | MR1, mpr-1, one MRI per subject |
| OASIS-2 selection | last available MRI visit per subject, same-visit CDR, mpr-1 |
| Split seed | 2026 |
| Corruption seed | 42 |
| Training seeds | 13, 47, 101 |
| Classical ML | Random Forest; Linear SVM |
| DL | SimpleCNN; ResNet18 trained from scratch |
| DL input | 128×128 grayscale |
| Classical sensitivity | 32×32, 64×64, 128×128 |
| Main classical resolution | 64×64 |
| Distortions | Motion, Rician noise, rotation, Gaussian blur, brightness/contrast |
| Severities | Mild, moderate, severe |
| Robustness metrics | ARR, DRI, CD-DRI |
| Statistical unit | Subject/patient for inferential analysis |
| Reader study | 160 blinded images; readability only |

## Cohorts used in the study

### OASIS-1
- 235 labelled subjects:
  - Non-Demented: 135
  - Very Mild Dementia: 70
  - Mild Dementia: 28
  - Moderate Dementia: 2
- Subject split:
  - train: 165
  - validation: 34
  - test: 36
- 20 axial slices per subject:
  - train: 3,300 images
  - validation: 680 images
  - test: 720 images
  - total: 4,700 images

### OASIS-2
- 150 subjects:
  - Non-Demented: 73
  - Very Mild Dementia: 53
  - Mild Dementia: 21
  - Moderate Dementia: 3
- Subject split:
  - train: 104
  - validation: 23
  - test: 23
- 20 axial slices per subject:
  - train: 2,080 images
  - validation: 460 images
  - test: 460 images
  - total: 3,000 images

The Moderate Dementia class is extremely small. Class-specific results for this class should be interpreted as exploratory.

## Repository layout

```text
MRI-QRF-replication/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── .gitattributes
├── config/
│   ├── study_protocol.yaml
│   └── paths.example.yaml
├── docs/
│   ├── 01_data_access.md
│   ├── 02_cohort_and_splits.md
│   ├── 03_preprocessing.md
│   ├── 04_distortion_pipeline.md
│   ├── 05_model_training.md
│   ├── 06_statistics.md
│   ├── 07_reader_study.md
│   ├── 08_results_and_models_upload.md
│   ├── 09_reproduction_checklist.md
│   └── model_cards/
├── references/
│   └── oasis.bib
├── requirements/
├── scripts/
│   ├── data_preparation/
│   ├── distortions/
│   ├── modeling/
│   ├── statistics/
│   ├── reporting/
│   └── reader_study/
├── data/                 # local-only MRI data; ignored by Git
├── models/               # upload trained model artifacts here
└── results/              # upload non-image result files here
```

## End-to-end reproduction

### 1. Obtain OASIS data
See `docs/01_data_access.md`.

Expected original metadata files:
- OASIS-1 cross-sectional demographics spreadsheet
- OASIS-2 longitudinal demographics spreadsheet

Expected source image type:
- OASIS Analyze-format T1 MRI (`.hdr` + `.img`)

Do not commit the original OASIS data to GitHub.

### 2. Create the labelled cohorts
Example:

```bash
python scripts/data_preparation/organize_oasis1.py \
  --raw-root /path/to/OASIS1_raw \
  --metadata /path/to/oasis_cross-sectional.xlsx \
  --output-root /path/to/OASIS_1/organized \
  --manifest /path/to/OASIS_1/organized/oasis1_manifest.csv
```

```bash
python scripts/data_preparation/organize_oasis2.py \
  --raw-root /path/to/OASIS2_raw \
  --metadata /path/to/oasis_longitudinal_demographics.xlsx \
  --output-root /path/to/OASIS_2/organized \
  --manifest /path/to/OASIS_2/organized/oasis2_manifest.csv
```

### 3. Reproduce the subject split

For exact manuscript replication, place the final study split manifests in `data/manifests/` if available. Otherwise, the included deterministic split script reproduces the fixed class counts using seed 2026.

```bash
python scripts/data_preparation/split_subjects.py \
  --dataset OASIS_1 \
  --manifest /path/to/OASIS_1/organized/oasis1_manifest.csv \
  --split-root /path/to/OASIS_1/split \
  --output-manifest /path/to/OASIS_1/organized/oasis1_subject_split.csv
```

Repeat with `--dataset OASIS_2`.

### 4. Preprocess to 128×128

```bash
python scripts/data_preparation/preprocess_to_128.py \
  --dataset OASIS_1 \
  --split-root /path/to/OASIS_1/split \
  --output-root /path/to/OASIS_1/clean_128 \
  --manifest /path/to/OASIS_1/clean_128/oasis1_slice_manifest.csv
```

Repeat for OASIS-2.

The exact preprocessing protocol is documented in `docs/03_preprocessing.md`.

### 5. Generate the 15 degraded test conditions

```bash
python scripts/distortions/generate_distortions.py \
  --dataset OASIS_1 \
  --clean-test-root /path/to/OASIS_1/clean_128/test \
  --output-root /path/to/OASIS_1/distortions_128 \
  --global-seed 42
```

Repeat for OASIS-2.

Expected distorted-image counts:
- OASIS-1: 720 × 15 = 10,800
- OASIS-2: 460 × 15 = 6,900

### 6. Derive 64×64 and 32×32 copies

The lower-resolution images are derived **after** the 128×128 distortion is generated. Distortions are not regenerated independently at lower resolutions.

```bash
python scripts/data_preparation/derive_resolutions.py \
  --source /path/to/OASIS_1/clean_128 \
  --output /path/to/OASIS_1/clean_64 \
  --size 64
```

Run the same command for clean/distorted 64 and 32 copies for both datasets.

### 7. Validate prepared data

```bash
python scripts/data_preparation/validate_prepared_data.py \
  --oasis1-root /path/to/OASIS_1 \
  --oasis2-root /path/to/OASIS_2
```

### 8. Train/evaluate classical ML

The executed classical script is preserved as:

```text
scripts/modeling/run_ml_models_mri_qrf.py
```

It was run separately at 32, 64, and 128 by changing `RESOLUTION` and `OUTPUT_ROOT`. The main classical configuration is 64×64; 32×32 and 128×128 are the resolution-sensitivity analysis.

Models:
- Random Forest: 300 trees, max depth 10, `max_features="sqrt"`, `class_weight="balanced_subsample"`
- Linear SVM: StandardScaler + LinearSVC, C=1.0, class balanced, max_iter=1000

### 9. Aggregate classical predictions to subject level

```text
scripts/modeling/subject_level_analysis_mri_qrf.py
```

Aggregation:
- Random Forest: mean class probability across a subject’s 20 slices
- Linear SVM: mean decision score across slices
- final subject class: argmax of mean score

### 10. Train/evaluate DL

The executed RunPod script is:

```text
scripts/modeling/run_dl_models_mri_qrf.py
```

SimpleCNN:
- 3 conv blocks: 32 → 64 → 128
- batch norm, ReLU, max pooling
- adaptive average pooling
- dropout 0.30
- AdamW, lr 1e-3, weight decay 1e-4

ResNet18:
- trained from scratch (`weights=None`)
- single-channel first convolution
- SGD, lr 0.01, momentum 0.9, weight decay 1e-4

Both:
- batch 32
- max 20 epochs
- early stopping patience 5
- best checkpoint by clean validation Macro-F1
- weighted CE: `sqrt(N/N_c)`, normalized to mean 1
- seeds 13, 47, 101

The observed best checkpoints occurred before the 20-epoch cap, so the study retained the 20-epoch protocol.

### 11. Statistical analysis

See `docs/06_statistics.md`.

Included:
- patient-clustered bootstrap for Δ Accuracy and Δ Macro-F1
- logistic mixed-effects analysis in R:
  `correct ~ condition + (1|patient) + (1|training_seed)`
- clean condition as reference
- Holm correction across 15 clean-vs-corrupted contrasts

### 12. Expert reader study

See `docs/07_reader_study.md`.

The repository includes:
- selection/preparation script
- Google Apps Script backend
- HTML frontend
- instructions for keeping the answer key private

**Never commit `KEY_DO_NOT_SHARE/`.**

## Robustness metrics

### Accuracy Retention Rate

```text
ARR = corrupted accuracy / clean baseline accuracy × 100
```

ARR can exceed 100 if a model happens to classify the corrupted set more accurately than its own clean baseline.

### Dataset Robustness Index

```text
DRI = mean ARR across the 15 distortion × severity conditions
```

DRI may also exceed 100. This does **not** mean the degraded images have higher image quality or clinical value.

### Cross-Dataset Diagnostic Robustness Index

```text
CD-DRI = (DRI_OASIS-1 + DRI_OASIS-2) / 2
```

## Uploading final models and results

See `docs/08_results_and_models_upload.md`.

Large model files (`*.pt`, `*.pth`, `*.joblib`) are configured for **Git LFS** in `.gitattributes`. Install Git LFS before committing model artifacts.

## Reproducibility notes

- Subject split seed: 2026
- Corruption seed: 42
- Model seeds: 13, 47, 101
- No corrupted images enter training or validation.
- Subject, not slice, is the independent statistical unit.
- Raw per-sample predictions should be retained.
- OASIS images are not redistributed.

## Citation

See `CITATION.cff` and `references/oasis.bib`.
