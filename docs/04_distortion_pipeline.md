# 4. Distortion pipeline — final manuscript protocol

The revised MRI-QRF main study uses **five distortion families × three severities** at a canonical **128×128** resolution. Models are trained and validated only on clean images; synthetic distortions are introduced at held-out test time.

## Dataset-adaptive severity calibration

For each dataset, distortion family, and severity level, candidate strengths are evaluated **on clean training subjects only**. For each candidate, SSIM is calculated between the clean and degraded counterpart of each training slice, averaged within subjects, and summarized as the median across subjects. The candidate whose median subject-level SSIM is nearest the target is selected and **frozen before any test evaluation**.

| Severity | Engineering SSIM target |
|---|---:|
| Mild | 0.95 |
| Moderate | 0.85 |
| Severe | 0.70 |

The selected numerical parameters may differ across OASIS-1 and OASIS-2. Classifier predictions and accuracy **are never used to choose** corruption strength. These SSIM targets are engineering conventions, **not clinically validated severity thresholds**.

| Family | Calibrated quantity |
|---|---|
| Motion | Fraction of k-space lines affected; maximum displacement |
| Rician noise | Noise standard deviation |
| Rotation | In-plane rotation magnitude |
| Gaussian blur | Gaussian standard deviation |
| Brightness/contrast | Brightness offset and contrast scaling |

Stochastic realizations are independently sampled per image with a reproducible seed derived from global seed **42**, image identity, and condition. Noise fields, motion line/displacement selection, rotation sign, and brightness/contrast direction can vary per image even though their calibrated strength is fixed for the condition. Gaussian blur is deterministic.

## Current implementation limitation

The committed `scripts/distortions/generate_distortions.py` **still implements an older fixed-parameter corruption schedule**, not the final training-set adaptive SSIM calibration. This is a **replication gap**, not an alternative description of the final experiment. The final calibrated parameter manifest and calibration/generation implementation have not yet been provided. Do not label the current generator's output as the final manuscript's held-out corrupted set.

Expected test-condition image counts after correct calibration: OASIS-1 **720 × 15 = 10,800**; OASIS-2 **460 × 15 = 6,900**. No MRI source or derived images should be redistributed without permission.

## Revised scripts now included

- `scripts/distortions/calibrate_all_distortions.py`: initial training-only SSIM calibration (Pass 2).
- `scripts/distortions/calibrate_all_distortions_pass3_targeted.py`: targeted refinement of six conditions, creating `calibrated_parameters_FINAL.csv`.
- `scripts/distortions/generate_calibrated_test_distortions.py`: frozen-parameter test generation (SSIM 0.95/0.85/0.70).

**Execution requirements:** The uploaded scripts contain local `D:\\` paths; adjust to the user's local clean-image directories before running. The generated calibration CSV is not committed; it must be generated from the licensed training data and verified to pass its ±0.02 tolerance checks. No imaging data or model weights are stored here. The legacy `generate_distortions.py` remains a distinct fixed-parameter implementation.
