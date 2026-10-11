# 7. Expert reader study

The expert study evaluates **image readability only**. Readers are not asked to diagnose AD, stage dementia, or make clinical decisions.

## Design
Total: **160 images**
- 10 clean
- 150 distorted = 5 distortions × 3 severities × 10 images

Constraints:
- every displayed image comes from a different underlying source slice
- the same source slice is never shown twice
- each 10-image condition is balanced as closely as possible across the four classes
- each group is balanced 5/5 between OASIS-1 and OASIS-2 where feasible
- available Moderate Dementia test slices are intentionally spread because the class is sparse
- images are enlarged to 512×512 for presentation
- all 160 images are shuffled and renamed `IMG_001` … `IMG_160`

## Blinding
Readers do not see:
- dataset
- class
- clean/distorted status
- distortion type
- severity

The private answer key must not be placed in the public repository.

## Rating scale
- Readable
- Borderline
- Unreadable

Expected readability used only for analysis:
- clean → Readable
- mild → Readable
- moderate → Borderline
- severe → Unreadable

This is **expected_readability**, not ground truth.

## Web application
`google_apps_script/Code.gs`:
- builds a blinded Drive manifest
- verifies `IMG_001` … `IMG_160`
- serves images
- saves/upserts ratings
- restores progress by rater code
- logs consent

`google_apps_script/index.html`:
- consent/introduction page
- one image at a time
- keyboard shortcuts
- Back navigation
- progress bar
- cross-device resume using the same rater code

## Final reported agreement and scope

- Original (pre-reconciliation) quadratic-weighted Cohen's κ: **0.420**.
- **Original exact agreement: 53.1%**; **same-or-adjacent-category agreement: 84.4%**. These are different statistics and must not be interchanged.
- After **algorithmic reconciliation of Borderline disagreements**, weighted κ was **0.674**; this was **not** a second independent reader assessment.
- Readability effects vary by corruption family and do not uniformly follow the engineering mild/moderate/severe labels.

The study evaluates readability, **not** Alzheimer diagnosis, clinically established MRI acceptability, or validation of the SSIM targets or DRI tiers.
