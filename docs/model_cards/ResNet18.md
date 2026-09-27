# ResNet18 model card

## Architecture
Torchvision ResNet18 trained from scratch (`weights=None`).

Changes:
- first convolution: 1→64, 7×7, stride 2, padding 3
- final fully-connected layer: 4 outputs

## Optimization
- SGD
- lr 0.01
- momentum 0.9
- weight decay 1e-4
- batch 32
- max 20 epochs
- patience 5
- best clean-validation Macro-F1 checkpoint

## Loss
Weighted cross entropy, `sqrt(N/N_c)` normalized to mean 1.

## Subject aggregation
Average softmax probabilities across slices and take argmax.
