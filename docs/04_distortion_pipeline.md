# 4. Distortion pipeline

Distortions are applied **only to held-out test images**. Training and validation remain clean.

There are 5 distortion types × 3 severities = **15 degraded test conditions**.

Global corruption seed: **42**.

A deterministic image-condition seed is derived from:
- global seed
- dataset name
- relative image path
- distortion name
- severity

This makes every corrupted image reproducible independently of processing order.

## Parameters

| Distortion | Mild | Moderate | Severe |
|---|---|---|---|
| Motion | 10% k-space lines; displacement ≤1 px | 20%; ≤3 px | 40%; ≤6 px |
| Rician noise | σ=0.035 | σ=0.075 | σ=0.15 |
| Rotation | ±5° | ±10° | ±15° |
| Gaussian blur | σ=0.75 px | σ=1.5 px | σ=2.5 px |
| Brightness/contrast | ±5% / ±10% | ±10% / ±20% | ±15% / ±30% |

Approximate Gaussian-blur FWHM:
- mild: 1.8 px
- moderate: 3.5 px
- severe: 5.9 px

## Motion implementation

The motion operator works in 2D Fourier space. A severity-dependent subset of phase-encoding lines is selected. Selected lines receive Fourier phase shifts generated from random integer translations within the severity-specific maximum displacement. The inverse transform produces the corrupted magnitude image.

This is a controlled simulation, not a full physical model of all patient-motion trajectories.

## Rician noise

For normalized image `x`:

```text
sqrt((x + n1)^2 + n2^2)
```

where `n1` and `n2` are independent Gaussian noise fields with the severity-specific σ.

## Rotation

The magnitude is fixed by severity. The sign is chosen deterministically from ±1 using the image-specific RNG.

## Brightness / contrast

Contrast is adjusted around middle gray (0.5), followed by an additive brightness change. Signs are selected deterministically from ±1. Output is clipped to `[0,1]`.

## Expected counts

- OASIS-1: 720 test images × 15 = **10,800**
- OASIS-2: 460 × 15 = **6,900**
