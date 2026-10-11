# MRI-QRF — MRI Quality Robustness Framework

Research code and documentation accompanying **“MRI-QRF: An Evidence-Based Framework for Evaluating the Robustness of Classification Models to MRI Quality Degradation in Alzheimer's Disease Classification”** (Oshan, Alhomaidani, and Alhoshan, 2026; manuscript).

**Scope:** OASIS-1 and OASIS-2, binary (CDR=0 vs CDR>0) and four-class (CDR 0/0.5/1/2) tasks; Random Forest, linear SVM, SimpleCNN and ResNet18; five simulated image distortions at three severity levels; training seeds 13, 47 and 101.

> **Status:** This is a research-code replication package, **not a turnkey reproduction or clinically validated system**. MRI inputs, final calibration CSV, trained checkpoints and final GEE analysis scripts/results are not bundled. Some older scripts remain for historical comparison. Do not interpret the presence of a script as evidence that it reproduces a reported result without matching the corresponding study run.

## Protocol overview

| Item | Revised study setting |
|---|---|
| Data | Official OASIS-1 and OASIS-2; one selected MRI observation per participant |
| OASIS-1 labelled cohort | 235 subjects (165 train, 34 validation, 36 test) |
| OASIS-2 labelled cohort | 150 subjects (104 train, 23 validation, 23 test) |
| Slice processing | 20 axial slices/subject; 128×128 grayscale |
| Tasks | Binary CDR=0 vs >0; four-class CDR=0, 0.5, 1, 2 |
| Models | RandomForest, LinearSVM, SimpleCNN, ResNet18 |
| Main comparison | 128×128 images (flattened for classical models); older 32/64 scripts are not main evaluation |
| Distortion families | Motion, Rician noise, rotation, Gaussian blur, brightness/contrast |
| SSIM engineering targets | Mild 0.95, moderate 0.85, severe 0.70 |
| Calibration | Per dataset and distortion, on clean training subjects only; selected parameters frozen before test generation |
| Seeds | Subject split 2026; corruption 42; models 13, 47, 101 |
| Statistical analysis | Subject-clustered 2,000-resample bootstrap; seed-adjusted logistic GEE, planned Holm adjustments |
| Reader study | 160 blinded images, two radiologists; readability, **not** diagnostic validity |

The SSIM levels and robustness categories are **heuristic engineering/reporting choices**, not clinical MRI-quality thresholds. OASIS-1 has only two Moderate Dementia subjects in the selected cohort; strong class-specific inference is unsupported.

## Metrics and interpretation

For a given dataset × task × model × condition, **ARR = 100 × (degraded accuracy / clean accuracy)**, provided the clean denominator is nonzero. In the revised manuscript, **DRI = mean ARR over the 15 distortion×severity conditions** within one dataset × task × model; it is **not** a per-family DRI. **CD-DRI = (DRI(OASIS-1) + DRI(OASIS-2))/2** for the corresponding task/model. Report clean accuracy alongside retention; ARR/DRI may exceed 100 without implying improved image quality.

Interpretation bands (≥90 high, 75–<90 moderate, 60–<75 fragile, <60 low) are **heuristic only**, not clinically validated.

The reader study rates image readability rather than Alzheimer diagnosis or SSIM threshold validity. Statistical analyses operate at subject level, not by treating slices as independent observations.

## Model weights and data sharing

Model weights are **not required to be committed to GitHub**. Retain local training code, configurations and numerical outputs; optionally deposit large weights in an external repository permitted by applicable licenses. Do not redistribute raw/derived OASIS MRI images without permission or post unapproved participant/reader-level material.

## Other documentation

- [Modeling scripts](scripts/modeling/README.md)
- [Distortion calibration](docs/04_distortion_pipeline.md)
- [Inference methods and gaps](docs/06_statistics.md)
- [Reader study methods](docs/07_reader_study.md)
- [Result artifact guidance](docs/08_results_and_models_upload.md)

## Citation

**Current manuscript:** Oshan, M., Alhomaidani, M., & Alhoshan, W. (2026). *MRI-QRF: An Evidence-Based Framework for Evaluating the Robustness of Classification Models to MRI Quality Degradation in Alzheimer's Disease Classification*. Manuscript (publication details pending).

**OASIS-1:** Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience, 19*(9), 1498–1507. https://doi.org/10.1162/jocn.2007.19.9.1498

**OASIS-2:** Marcus, D. S., Fotenos, A. F., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2010). Open Access Series of Imaging Studies: Longitudinal MRI data in nondemented and demented older adults. *Journal of Cognitive Neuroscience, 22*(12), 2677–2684. https://doi.org/10.1162/jocn.2009.21407

Dataset access: https://www.oasis-brains.org/
