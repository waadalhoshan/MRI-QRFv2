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

## Analysis
Recommended reporting:
- rating distributions by severity/distortion
- quadratic-weighted Cohen's κ between readers
- concordance with expected readability
- descriptive human-vs-model comparison

Do not describe the reader study as validation of AD diagnostic accuracy.
