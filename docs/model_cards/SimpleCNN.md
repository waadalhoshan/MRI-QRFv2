# SimpleCNN model card

## Architecture
- Conv 1→32 + BN + ReLU + MaxPool
- Conv 32→64 + BN + ReLU + MaxPool
- Conv 64→128 + BN + ReLU + MaxPool
- AdaptiveAvgPool(1×1)
- Dropout 0.30
- Linear 128→4

## Optimization
- AdamW
- lr 1e-3
- weight decay 1e-4
- batch 32
- max 20 epochs
- patience 5
- best clean-validation Macro-F1 checkpoint

## Loss
Weighted cross entropy, `sqrt(N/N_c)` normalized to mean 1.

## Input
1×128×128 grayscale.

## Subject aggregation
Average softmax probabilities across slices and take argmax.
