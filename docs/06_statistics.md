# 6. Statistical analysis

## Independent unit

The **subject/patient** is the independent observational unit. Slice-level predictions are preserved for reproducibility and subject aggregation, but slices are not treated as independent patients.

## Mixed-effects logistic model

For each model/dataset analysis, correctness is modeled as:

```text
correct ~ condition + (1 | patient) + (1 | training_seed)
```

- response: correct / incorrect
- clean condition: reference
- patient: random intercept
- training seed: random intercept
- 15 corrupted conditions contrasted against clean
- Holm correction across the 15 contrasts

The repository provides:
- `prepare_glmm_input.py`
- `mixed_effects_logistic.R`

The R implementation uses `lme4` and `emmeans`.

Because there are only three training-seed levels, the seed random effect may be weakly estimated; warnings/singular fits should be retained and reported rather than suppressed.

## Patient-clustered bootstrap

The repository provides `patient_cluster_bootstrap.py`.

For each bootstrap replicate:
1. subjects are resampled with replacement;
2. all observations for a selected subject are retained;
3. clean and corrupted conditions remain paired;
4. metric changes are calculated within each training seed;
5. seed-specific changes are averaged within the bootstrap replicate.

Outputs:
- Δ Accuracy from clean
- 95% CI
- Δ Macro-F1 from clean
- 95% CI

Number of bootstrap resamples: **2000**

The Moderate Dementia class is too small for strong class-specific inferential claims.
