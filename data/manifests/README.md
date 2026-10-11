# Subject splits, slice manifests and calibration metadata

These folders contain **placeholders**, not necessarily the specific manifest files used to produce manuscript results.

For exact numerical reproduction, preserve and, only where permissible, release:

- The selected OASIS-1 and OASIS-2 cohort manifests and **actual subject-level train/validation/test assignments**.
- The slice-to-subject mapping and image processing manifest for the 20 selected axial slices per subject.
- The final dataset-specific frozen `calibrated_parameters_FINAL.csv`, produced by the Pass-2 and Pass-3 scripts (30 combinations: two datasets × five families × three levels), including selected parameter, achieved SSIM and tolerance information.

The revised scripts' calibration table is ordinarily written under local `D:\\MRI-QRF_calibrated\\results\\calibration_pass3`, not automatically into this GitHub folder. The published repository does not currently include that final table. **Never fabricate or copy parameter values from an older fixed-distortion experiment**.

Expected study counts: OASIS-1 235 subjects split 165/34/36; OASIS-2 150 split 104/23/23. Binary and multiclass tasks reuse the same subject assignments.

Check all CSVs for subject IDs, original scan names/paths, demographic fields, restricted metadata or reader identities before public release. When redistribution is not allowed, provide schemas and generation instructions instead.

See [the data README](../README.md), [splits](../../docs/02_cohort_and_splits.md) and [distortion calibration](../../docs/04_distortion_pipeline.md).
