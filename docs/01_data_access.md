# 1. Data access

MRI-QRF uses official OASIS MRI data obtained directly from the Open Access Series of Imaging Studies.

- **OASIS-1:** cross-sectional MRI dataset
- **OASIS-2:** longitudinal MRI dataset

Official project: https://www.oasis-brains.org/

The repository intentionally does **not** redistribute:
- `.hdr` / `.img` MRI files
- source or derived OASIS image slices
- derived distorted images
- private reader-study answer keys

Researchers should obtain the datasets from OASIS and comply with the applicable OASIS data-use terms.

## Required source information

### OASIS-1
The study uses:
- first MRI session (`MR1`)
- first T1 acquisition (`mpr-1`)
- subjects with available CDR equal to 0, 0.5, 1, or 2

### OASIS-2
The study uses:
- one observation per subject
- the **last available MRI visit**
- CDR from the same visit
- first T1 acquisition (`mpr-1`)

This prevents longitudinal visits from the same person being treated as independent observations.

## Citations

See `references/oasis.bib`.

- Marcus et al., 2007: OASIS cross-sectional dataset
- Marcus et al., 2010: OASIS longitudinal dataset
