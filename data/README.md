# Local data layout and sharing restrictions

**MRI data are not hosted in this public repository.** Obtain OASIS-1/OASIS-2 from the official source and follow their applicable terms. Raw source volumes, derived 128×128 slices and corrupted versions must not be committed or redistributed without permission.

The executed revised scripts assume **dataset-specific external folders**, not a single unified `data/` directory:

```text
OASIS_1/
  clean_128/
    train/<class>/<subject>/*.png
    val/<class>/<subject>/*.png
    test/<class>/<subject>/*.png
  distortions_128_calibrated/<family>/<severity>/<class>/<subject>/*.png
OASIS_2/
  ... same arrangement ...
```

`<class>` denotes a source CDR class (not the derived binary label). These data folder trees must stay private/local; data preparation can start from source OASIS paired MRI volumes. The revised calibration/classical scripts default to Windows `D:\\OASIS_1` and `D:\\OASIS_2`, while DL scripts default to `/workspace/OASIS_1` and `/workspace/OASIS_2`. Change these path constants for your machine.

**Calibration:** use only `clean_128/train`, selecting each distortion/level's parameter based on median subject-level SSIM against 0.95/0.85/0.70. The frozen parameter CSV then governs held-out corruption generation into `distortions_128_calibrated`. The older `distortions_128` folder and 32/64-resolution outputs refer to legacy/sensitivity configurations, not the revised main results.

Non-image manifests may be placed in [manifests/](manifests/) **only after checking permissions and participant-identification risk**. Exact participant assignment is not guaranteed by re-running the split seed alone.
