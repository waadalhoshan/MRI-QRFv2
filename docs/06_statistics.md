# 6. Statistical inference — final manuscript protocol

The independent observational unit is the **subject**, not the MRI slice. Analyses account for subject-level repeated observations and training seed. Two-sided testing uses an adjusted significance threshold of p < 0.05.

## Bootstrap uncertainty

For each distortion family, average the distorted-minus-clean **accuracy change** across mild, moderate, and severe conditions, and estimate a 95% confidence interval using **2,000 subject-clustered bootstrap resamples**, retaining the pairing between clean and distorted observations.

## Severity contrasts

Fit logistic **generalized estimating equations (GEE)** to subject-level correct/incorrect classification, cluster by subject, and adjust for training seed. Obtain three contrasts against clean: mild, moderate, and severe. Apply **Holm correction across the three planned severity contrasts** within each dataset × task × model combination.

## Severity trend

Exclude clean images and fit a subject-clustered, seed-adjusted logistic GEE with mild → moderate → severe as an ordered predictor. Apply a conservative Holm adjustment across trend comparisons within each distortion family. Non-monotonic responses must be reported rather than forced into a monotonic interpretation.

## Cross-dataset analysis

For binary and multiclass classification **separately**, fit subject-level logistic GEE using degraded observations, adjusting for severity and seed and clustering by subject. Examine Dataset×Distortion, Model×Distortion, and Dataset×Distortion×Model interactions. CD-DRI is a **descriptive average of dataset-specific DRI**, not itself the unit of inferential testing.

## Source-code status

The committed `scripts/statistics/mixed_effects_logistic.R` and `prepare_glmm_input.py` describe an **earlier mixed-effects logistic regression** with 15-condition Holm correction. They are **not implementations of the final GEE protocol**. The final revised GEE scripts, actual analysis inputs, and verified outputs are needed before claiming complete computational replication.
