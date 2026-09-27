# 8. Uploading final results and model artifacts

This package intentionally creates upload locations without embedding private or very large study outputs.

## Models

Upload trained artifacts under:

```text
models/
├── classical/
│   ├── ML_32/
│   ├── ML_64/
│   └── ML_128/
└── deep/
    └── DL_128/
```

Recommended hierarchy:

```text
models/deep/DL_128/
  OASIS_1/
    SimpleCNN/seed_13/best_model.pt
    ...
  OASIS_2/
    ResNet18/seed_101/best_model.pt
```

Classical models should preserve dataset/model/seed naming.

### Git LFS
GitHub rejects normal Git blobs larger than 100 MB. `.gitattributes` is configured for:
- `.pt`
- `.pth`
- `.ckpt`
- `.joblib`
- `.pkl`

Before committing model files:

```bash
git lfs install
git lfs track "*.pt" "*.pth" "*.joblib"
```

Check your GitHub/Git-LFS storage quota before uploading all checkpoints.

## Results

Upload non-image results under:

```text
results/
├── classical/
│   ├── ML_32/
│   ├── ML_64/
│   ├── ML_128/
│   └── subject_level_analysis/
├── deep/
│   └── DL_128/
├── statistics/
├── reader_study/
└── figures/
```

Recommended files to preserve:
- raw per-slice predictions
- raw per-subject predictions
- condition-level metrics
- per-class metrics
- confusion matrices
- validation metrics
- training histories
- run summaries
- ARR
- DRI
- CD-DRI
- across-seed mean/SD summaries
- bootstrap CIs
- mixed-model estimates and Holm-adjusted p-values

## Do not upload
- OASIS source images
- derived OASIS PNGs unless redistribution permission is explicitly confirmed
- blinded reader-study source MRI images if derived from OASIS and redistribution is not permitted
- `answer_key_DO_NOT_SHARE.csv`
- identifying reader information

Anonymized aggregate reader-study outputs can be placed in `results/reader_study/` if consistent with the study's consent/ethics requirements.
