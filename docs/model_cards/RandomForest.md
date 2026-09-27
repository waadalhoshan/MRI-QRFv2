# Random Forest model card

**Role:** classical robustness baseline.

## Input
Flattened grayscale pixels. Main resolution 64×64 (4,096 features); 32×32 and 128×128 are sensitivity analyses.

## Fixed configuration
- `n_estimators=300`
- `max_depth=10`
- `max_features="sqrt"`
- `class_weight="balanced_subsample"`
- `random_state=seed`
- `n_jobs=-1`

## Training
Clean training data only. Clean validation/test and corrupted test sets are evaluated without retraining.

## Subject aggregation
Average class probabilities across slices and take argmax.

## Limitations
Pixel-vector classifier; no explicit anatomical modeling. Moderate Dementia class is extremely sparse.
