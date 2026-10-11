# Blinded expert image-readability interface

This folder contains a **Google Apps Script reader-study interface**, not an MRI diagnosis tool. It supports blinded presentation of 160 images to the readers; it does not measure clinical diagnostic performance or validate the SSIM calibration targets.

Files:
- `Code.gs`: server-side Google Apps Script functions for serving and storing responses.
- `index.html`: reader-facing image/rating interface.

## Setup (requires a configured Google account, Drive and Sheets)

1. Prepare the **160-image** blinded display collection: 10 clean images plus 10 images from each of 15 distortion×severity conditions, with unique source slices as described in [the reader-study protocol](../../../docs/07_reader_study.md). Do not upload patient-identifiable or OASIS-derived images to an externally shared service without the necessary permissions and institutional approval.
2. Keep the private condition/key mapping and any identifying participant metadata **outside this public repository and outside reader-accessible Drive folders**.
3. Create/configure the linked Google Sheet and confirm the sheet/table names expected by `Code.gs` (including `Ratings` and `Consent_Log` where applicable).
4. Configure the Drive folder identifier in the Apps Script code for `IMG_001.png` through `IMG_160.png`, then run `buildManifestFromDrive()` if supported by the deployed code.
5. Verify all 160 image IDs, image accessibility, deployment permissions and consent workflow. Deploy the app to run as the study owner, and provide each authorized reader a unique non-identifying code.
6. Before collecting actual ratings, test one consenting mock session, including saving/updating ratings, Back navigation and resuming with the same rater code.

The interface is designed to save/update ratings by reader code and image identifier rather than treating re-entries as new independent observations. Consult the actual Apps Script configuration before deployment; **this README is not evidence that a web app is currently deployed**.

## What ratings mean

Readers assign **Readable**, **Borderline**, or **Unreadable**. These are subjective image-readability judgements, not AD diagnoses. An expected severity-to-readability mapping, if used for descriptive analysis, is **not** reader ground truth. Report original reader agreement separately from any **algorithmically reconciled** agreement; do not present reconciliation as additional independent clinical reading.

The final outputs and original reader data/key are **not supplied here**. Never publish a private answer key, restricted MRI images, consent records, or identifiable reader information.

See [docs/07_reader_study.md](../../../docs/07_reader_study.md) for study design and interpretation.
