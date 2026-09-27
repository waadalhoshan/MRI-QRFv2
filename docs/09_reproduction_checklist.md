# 9. Reproduction checklist

Use this checklist before creating a release.

## Data
- [ ] OASIS-1 obtained from official source
- [ ] OASIS-2 obtained from official source
- [ ] no OASIS images committed to Git
- [ ] OASIS-1 MR1/mpr-1 selection verified
- [ ] OASIS-2 last-visit/same-visit-CDR selection verified

## Cohorts
- [ ] OASIS-1 = 235 subjects
- [ ] OASIS-2 = 150 subjects
- [ ] CDR mapping 0/0.5/1/2 verified
- [ ] no subject leakage
- [ ] split seed = 2026
- [ ] exact study split manifests uploaded if available

## Clean preprocessing
- [ ] `nib.as_closest_canonical`
- [ ] 1st/99th percentile normalization on finite nonzero voxels
- [ ] background restored to 0
- [ ] 20 axial slices from central 40%
- [ ] no crop
- [ ] bilinear 128×128 resize
- [ ] 8-bit grayscale PNG
- [ ] OASIS-1 total = 4700
- [ ] OASIS-2 total = 3000

## Distortions
- [ ] test only
- [ ] global seed = 42
- [ ] all 15 conditions
- [ ] OASIS-1 distorted total = 10800
- [ ] OASIS-2 distorted total = 6900
- [ ] lower-resolution distortions derived from 128×128 corrupted images

## Classical ML
- [ ] RF fixed hyperparameters
- [ ] Linear SVM fixed hyperparameters
- [ ] seeds 13,47,101
- [ ] clean training/validation only
- [ ] 64 main + 32/128 sensitivity
- [ ] raw prediction scores saved
- [ ] subject-level aggregation completed

## DL
- [ ] SimpleCNN architecture verified
- [ ] ResNet18 weights=None
- [ ] weighted CE verified
- [ ] batch 32
- [ ] max epochs 20
- [ ] patience 5
- [ ] best validation Macro-F1 checkpoint
- [ ] seeds 13,47,101
- [ ] raw prediction probabilities saved
- [ ] subject-level aggregation completed

## Robustness / statistics
- [ ] ARR calculated relative to each clean baseline
- [ ] DRI averages 15 ARR values
- [ ] CD-DRI averages two dataset DRIs
- [ ] patient-clustered bootstrap = 2000
- [ ] GLMM clean reference
- [ ] Holm correction
- [ ] ARR/DRI >100 interpreted correctly

## Reader study
- [ ] 160 unique source slices
- [ ] 10 clean + 150 corrupted
- [ ] blinding verified
- [ ] answer key excluded from public repository
- [ ] readability task only

## Repository release
- [ ] results uploaded
- [ ] model artifacts uploaded or external release links documented
- [ ] Git LFS configured
- [ ] checksums regenerated
- [ ] README paths/commands tested
