# 5. Model training and evaluation

All models are trained only on clean training data and evaluated first on the clean held-out test set. The trained model is then frozen and applied to all 15 corrupted test conditions.

Training seeds: **13, 47, 101**

## Classical machine learning

### Random Forest
- 300 trees
- max depth = 10
- max features = square root
- class weight = balanced subsample
- fixed seed per run

### Linear SVM
- flattened grayscale pixels
- StandardScaler
- LinearSVC
- C = 1.0
- class weight = balanced
- max_iter = 1000

### Resolution analysis
- 32×32: 1,024 features
- 64×64: 4,096 features
- 128×128: 16,384 features

The main classical configuration is 64×64. The 32×32 and 128×128 results are retained as an input-resolution sensitivity analysis.

### Subject-level aggregation
Because the independent observational unit is the subject:
- RF: average class probabilities across slices
- Linear SVM: average decision scores across slices
- final class = argmax(mean score)

## Deep learning

### SimpleCNN
Three blocks:
1. Conv 1→32 + BatchNorm + ReLU + MaxPool
2. Conv 32→64 + BatchNorm + ReLU + MaxPool
3. Conv 64→128 + BatchNorm + ReLU + MaxPool

Then:
- AdaptiveAvgPool(1×1)
- Flatten
- Dropout 0.30
- Linear 128→4

Optimizer:
- AdamW
- lr 1e-3
- weight decay 1e-4

### ResNet18
- torchvision ResNet18
- `weights=None`
- first convolution changed to 1 input channel
- final FC changed to 4 classes
- trained from scratch

Optimizer:
- SGD
- lr 0.01
- momentum 0.9
- weight decay 1e-4

### Shared DL settings
- input 1×128×128
- batch size 32
- max epochs 20
- early stopping patience 5
- model checkpoint selected by clean validation Macro-F1
- no test-time augmentation

### Class weighting
Weighted cross-entropy:

```text
w_c ∝ sqrt(N / N_c)
```

Weights are normalized to mean 1.

The Moderate Dementia training count is very small. This weighting softens, rather than fully inversely weights, the class imbalance.

## Clean performance metrics
- Accuracy
- Macro-F1
- Weighted-F1
- Balanced Accuracy
- per-class precision, recall, F1, support
- confusion matrix

## Robustness metrics

```text
ARR = corrupted accuracy / clean accuracy × 100
DRI = mean ARR over all 15 corrupted conditions
CD-DRI = mean DRI across OASIS-1 and OASIS-2
```

ARR and DRI can exceed 100. This indicates measured corrupted-condition accuracy exceeded the corresponding clean baseline; it does not mean image quality improved.
