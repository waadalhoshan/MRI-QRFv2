# Linear SVM model card

**Role:** classical linear robustness baseline.

## Input
Flattened grayscale pixels.

## Fixed configuration
Pipeline:
1. `StandardScaler`
2. `LinearSVC(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)`

## Subject aggregation
Average class decision scores across slices and take argmax.

## Limitations
Linear pixel-space decision boundary; sensitive to scaling/intensity perturbations. Moderate Dementia is extremely sparse.
