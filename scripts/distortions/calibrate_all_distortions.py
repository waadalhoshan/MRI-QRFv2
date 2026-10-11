#!/usr/bin/env python3
"""
MRI-QRF dataset-calibrated corruption calibration (Pass 2)

Purpose
-------
Calibrate mild/moderate/severe corruption parameters separately for OASIS-1
and OASIS-2 using ALL clean 128x128 training slices.

Calibration target:
    mild      SSIM ~= 0.95
    moderate  SSIM ~= 0.85
    severe    SSIM ~= 0.70

Important design choices
------------------------
1. Calibration uses CLEAN TRAINING DATA ONLY.
2. Every training slice is used.
3. SSIM is computed per slice.
4. Slice SSIM values are averaged within subject.
5. Candidate parameters are summarized by the MEDIAN subject-level SSIM.
6. A candidate is acceptable only if it is within TARGET_TOLERANCE of target.
7. Randomness is deterministic and candidate-consistent:
   - Rician candidates share the same underlying Gaussian draws per image.
   - Rotation candidates share the same +/- direction per image.
   - Brightness/contrast uses a deterministic balanced 4-sign design.
   - Motion candidates share a nested k-space line ordering and displacement field.
8. No corrupted calibration images are saved.

This script DOES NOT modify the test set and DOES NOT train any model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import rotate, gaussian_filter
from skimage.metrics import structural_similarity as ssim


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    "OASIS_1": Path(r"D:\OASIS_1\clean_128\train"),
    "OASIS_2": Path(r"D:\OASIS_2\clean_128\train"),
}

OUTPUT_ROOT = Path(
    r"D:\MRI-QRF_calibrated\results\calibration_pass2"
)

GLOBAL_SEED = 42

TARGETS = {
    "mild": 0.95,
    "moderate": 0.85,
    "severe": 0.70,
}

TARGET_TOLERANCE = 0.02


# ============================================================
# REFINED CANDIDATE GRIDS
# ============================================================

CANDIDATES = {
    "motion": [
        (0.005, 1),
        (0.010, 1),
        (0.020, 1),
        (0.030, 1),
        (0.040, 1),
        (0.050, 1),
        (0.075, 1),
        (0.100, 1),
        (0.100, 2),
        (0.150, 2),
        (0.200, 3),
        (0.300, 4),
        (0.400, 6),
        (0.500, 8),
        (0.600, 10),
        (0.700, 12),
    ],

    "rician_noise": [
        0.0005,
        0.0010,
        0.0020,
        0.0030,
        0.0040,
        0.0050,
        0.0075,
        0.0100,
        0.0150,
        0.0200,
        0.0250,
        0.0350,
        0.0500,
        0.0750,
        0.1000,
        0.1500,
        0.2000,
        0.2500,
        0.3000,
    ],

    "rotation": [
        0.10,
        0.25,
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
        3.00,
        4.00,
        5.00,
        6.00,
        8.00,
        10.00,
        12.00,
        15.00,
        18.00,
        20.00,
        25.00,
        30.00,
    ],

    "gaussian_blur": [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.75,
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
        2.50,
        3.00,
        3.50,
        4.00,
        5.00,
    ],

    "brightness_contrast": [
        (0.005, 0.010),
        (0.010, 0.020),
        (0.015, 0.030),
        (0.020, 0.040),
        (0.030, 0.060),
        (0.040, 0.080),
        (0.050, 0.100),
        (0.070, 0.150),
        (0.100, 0.200),
        (0.120, 0.250),
        (0.150, 0.300),
        (0.180, 0.350),
        (0.200, 0.400),
        (0.250, 0.500),
        (0.300, 0.600),
    ],
}

DISTORTIONS = [
    "motion",
    "rician_noise",
    "rotation",
    "gaussian_blur",
    "brightness_contrast",
]


# ============================================================
# DETERMINISTIC HELPERS
# ============================================================

def stable_seed(*parts) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def get_subject_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = rel.parts

    # Expected: <class>/<subject>/<slice>.png
    if len(parts) < 3:
        raise RuntimeError(
            f"Unexpected path structure under {root}: {rel}"
        )

    return parts[1]


def compute_ssim(clean: np.ndarray, corrupted: np.ndarray) -> float:
    return float(
        ssim(
            clean,
            corrupted,
            data_range=1.0,
        )
    )


def parameter_string(parameter) -> str:
    if isinstance(parameter, tuple):
        return "(" + ",".join(f"{float(x):g}" for x in parameter) + ")"
    return f"{float(parameter):g}"


# ============================================================
# COMMON-RANDOM-NUMBER DISTORTIONS
# ============================================================

def apply_rician_common(
    x: np.ndarray,
    sigma: float,
    dataset: str,
    rel: str,
) -> np.ndarray:
    """
    All sigma candidates for the same image share the same N(0,1) fields.
    Only sigma changes.
    """
    rng = np.random.default_rng(
        stable_seed(GLOBAL_SEED, dataset, rel, "rician_noise")
    )

    z1 = rng.normal(0.0, 1.0, x.shape)
    z2 = rng.normal(0.0, 1.0, x.shape)

    y = np.sqrt(
        (x + sigma * z1) ** 2
        + (sigma * z2) ** 2
    )

    return np.clip(y, 0.0, 1.0)


def apply_rotation_common(
    x: np.ndarray,
    magnitude_deg: float,
    dataset: str,
    rel: str,
) -> np.ndarray:
    """
    All angle magnitudes for the same image use the same direction.
    """
    rng = np.random.default_rng(
        stable_seed(GLOBAL_SEED, dataset, rel, "rotation")
    )

    sign = float(rng.choice([-1.0, 1.0]))
    angle = sign * magnitude_deg

    y = rotate(
        x,
        angle,
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    return np.clip(y, 0.0, 1.0)


def brightness_contrast_signs(
    dataset: str,
    rel: str,
) -> tuple[float, float]:
    """
    Deterministic balanced four-way sign assignment:
      (+,+), (+,-), (-,+), (-,-)

    Across many images this avoids candidate-to-candidate sign noise.
    """
    idx = stable_seed(
        GLOBAL_SEED,
        dataset,
        rel,
        "brightness_contrast_sign_pair",
    ) % 4

    combos = [
        (+1.0, +1.0),
        (+1.0, -1.0),
        (-1.0, +1.0),
        (-1.0, -1.0),
    ]

    return combos[idx]


def apply_brightness_contrast_common(
    x: np.ndarray,
    bmag: float,
    cmag: float,
    dataset: str,
    rel: str,
) -> np.ndarray:
    bsign, csign = brightness_contrast_signs(dataset, rel)

    b = bsign * bmag
    c = csign * cmag

    y = (x - 0.5) * (1.0 + c) + 0.5 + b

    return np.clip(y, 0.0, 1.0)


def apply_blur(
    x: np.ndarray,
    sigma: float,
) -> np.ndarray:
    y = gaussian_filter(
        x,
        sigma=sigma,
        mode="nearest",
    )

    return np.clip(y, 0.0, 1.0)


def motion_random_structure(
    h: int,
    dataset: str,
    rel: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one nested random structure per image:
    - one random ordering of k-space rows
    - one U(-1,1) x-displacement value per row
    - one U(-1,1) y-displacement value per row

    Larger motion candidates use a prefix of the same row ordering and
    scale the same displacement fields by max_disp.
    """
    rng = np.random.default_rng(
        stable_seed(GLOBAL_SEED, dataset, rel, "motion")
    )

    row_order = rng.permutation(h)
    ux = rng.uniform(-1.0, 1.0, size=h)
    uy = rng.uniform(-1.0, 1.0, size=h)

    return row_order, ux, uy


def apply_motion_common(
    x: np.ndarray,
    fraction: float,
    max_disp: int,
    dataset: str,
    rel: str,
) -> np.ndarray:
    h, w = x.shape

    k = np.fft.fftshift(
        np.fft.fft2(x)
    )

    row_order, ux, uy = motion_random_structure(
        h,
        dataset,
        rel,
    )

    n_lines = max(
        1,
        int(round(fraction * h)),
    )

    lines = row_order[:n_lines]

    fx = np.fft.fftshift(
        np.fft.fftfreq(w)
    )

    fy = np.fft.fftshift(
        np.fft.fftfreq(h)
    )

    for row in lines:
        dx = int(np.rint(ux[row] * max_disp))
        dy = int(np.rint(uy[row] * max_disp))

        phase = np.exp(
            -2j
            * np.pi
            * (
                fx * dx
                + fy[row] * dy
            )
        )

        k[row, :] *= phase

    y = np.abs(
        np.fft.ifft2(
            np.fft.ifftshift(k)
        )
    )

    return np.clip(y, 0.0, 1.0)


def apply_corruption(
    x: np.ndarray,
    distortion: str,
    parameter,
    dataset: str,
    rel: str,
) -> np.ndarray:

    if distortion == "rician_noise":
        return apply_rician_common(
            x,
            float(parameter),
            dataset,
            rel,
        )

    if distortion == "rotation":
        return apply_rotation_common(
            x,
            float(parameter),
            dataset,
            rel,
        )

    if distortion == "gaussian_blur":
        return apply_blur(
            x,
            float(parameter),
        )

    if distortion == "brightness_contrast":
        bmag, cmag = parameter
        return apply_brightness_contrast_common(
            x,
            float(bmag),
            float(cmag),
            dataset,
            rel,
        )

    if distortion == "motion":
        fraction, max_disp = parameter
        return apply_motion_common(
            x,
            float(fraction),
            int(max_disp),
            dataset,
            rel,
        )

    raise ValueError(distortion)


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_dataset(
    dataset_name: str,
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print()
    print("=" * 90)
    print(f"CALIBRATING {dataset_name}")
    print("=" * 90)

    files = sorted(root.rglob("*.png"))

    if not files:
        raise RuntimeError(
            f"No PNG files found under: {root}"
        )

    subjects = [
        get_subject_id(p, root)
        for p in files
    ]

    print(f"Training images:   {len(files)}")
    print(f"Training subjects: {len(set(subjects))}")

    rows = []

    for distortion in DISTORTIONS:

        print()
        print(f"Distortion: {distortion}")

        candidate_list = CANDIDATES[distortion]

        for candidate_index, parameter in enumerate(candidate_list):

            print(
                f"  {candidate_index + 1:02d}/"
                f"{len(candidate_list):02d} "
                f"parameter={parameter}"
            )

            for path in files:

                rel = path.relative_to(root).as_posix()
                subject_id = get_subject_id(path, root)

                x = load_image(path)

                y = apply_corruption(
                    x=x,
                    distortion=distortion,
                    parameter=parameter,
                    dataset=dataset_name,
                    rel=rel,
                )

                ssim_value = compute_ssim(
                    x,
                    y,
                )

                rows.append({
                    "dataset": dataset_name,
                    "distortion": distortion,
                    "candidate_index": candidate_index,
                    "parameter": parameter_string(parameter),
                    "subject_id": subject_id,
                    "relative_path": rel,
                    "ssim": ssim_value,
                })

    slice_df = pd.DataFrame(rows)

    subject_df = (
        slice_df
        .groupby(
            [
                "dataset",
                "distortion",
                "candidate_index",
                "parameter",
                "subject_id",
            ],
            as_index=False,
        )
        .agg(
            mean_slice_ssim=("ssim", "mean"),
        )
    )

    candidate_df = (
        subject_df
        .groupby(
            [
                "dataset",
                "distortion",
                "candidate_index",
                "parameter",
            ],
            as_index=False,
        )
        .agg(
            n_subjects=("subject_id", "nunique"),
            median_subject_ssim=("mean_slice_ssim", "median"),
            mean_subject_ssim=("mean_slice_ssim", "mean"),
            sd_subject_ssim=("mean_slice_ssim", "std"),
            q1_subject_ssim=(
                "mean_slice_ssim",
                lambda x: float(np.percentile(x, 25)),
            ),
            q3_subject_ssim=(
                "mean_slice_ssim",
                lambda x: float(np.percentile(x, 75)),
            ),
        )
    )

    return slice_df, candidate_df


# ============================================================
# PARAMETER SELECTION
# ============================================================

def select_calibrated_parameters(
    candidate_summary: pd.DataFrame,
) -> pd.DataFrame:

    selected_rows = []

    severity_order = [
        "mild",
        "moderate",
        "severe",
    ]

    for dataset_name in sorted(
        candidate_summary["dataset"].unique()
    ):

        for distortion in DISTORTIONS:

            subset = (
                candidate_summary[
                    (candidate_summary["dataset"] == dataset_name)
                    &
                    (candidate_summary["distortion"] == distortion)
                ]
                .sort_values("candidate_index")
                .copy()
            )

            if subset.empty:
                raise RuntimeError(
                    f"No candidates for {dataset_name} / {distortion}"
                )

            previous_index = -1

            for severity in severity_order:

                target = TARGETS[severity]

                # Enforce increasing candidate severity:
                # moderate cannot select a weaker candidate than mild,
                # severe cannot select a weaker candidate than moderate.
                eligible = subset[
                    subset["candidate_index"] > previous_index
                ].copy()

                if eligible.empty:
                    raise RuntimeError(
                        f"No eligible candidates remain for "
                        f"{dataset_name}/{distortion}/{severity}"
                    )

                eligible["absolute_target_error"] = np.abs(
                    eligible["median_subject_ssim"] - target
                )

                best = (
                    eligible
                    .sort_values(
                        [
                            "absolute_target_error",
                            "candidate_index",
                        ]
                    )
                    .iloc[0]
                )

                previous_index = int(
                    best["candidate_index"]
                )

                error = float(
                    best["absolute_target_error"]
                )

                selected_rows.append({
                    "dataset": dataset_name,
                    "distortion": distortion,
                    "severity": severity,
                    "target_ssim": target,
                    "selected_candidate_index": previous_index,
                    "selected_parameter": best["parameter"],
                    "achieved_median_subject_ssim": float(
                        best["median_subject_ssim"]
                    ),
                    "achieved_mean_subject_ssim": float(
                        best["mean_subject_ssim"]
                    ),
                    "achieved_sd_subject_ssim": float(
                        best["sd_subject_ssim"]
                    ),
                    "absolute_target_error": error,
                    "within_tolerance": (
                        error <= TARGET_TOLERANCE
                    ),
                    "n_subjects": int(
                        best["n_subjects"]
                    ),
                })

    return pd.DataFrame(selected_rows)


def add_reachability_diagnostics(
    candidate_summary: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:

    out = selected.copy()

    min_ssim = (
        candidate_summary
        .groupby(
            ["dataset", "distortion"],
            as_index=False,
        )["median_subject_ssim"]
        .min()
        .rename(
            columns={
                "median_subject_ssim":
                "candidate_min_median_ssim"
            }
        )
    )

    max_ssim = (
        candidate_summary
        .groupby(
            ["dataset", "distortion"],
            as_index=False,
        )["median_subject_ssim"]
        .max()
        .rename(
            columns={
                "median_subject_ssim":
                "candidate_max_median_ssim"
            }
        )
    )

    out = (
        out
        .merge(
            min_ssim,
            on=["dataset", "distortion"],
            how="left",
        )
        .merge(
            max_ssim,
            on=["dataset", "distortion"],
            how="left",
        )
    )

    out["target_inside_candidate_ssim_range"] = (
        (out["target_ssim"] >= out["candidate_min_median_ssim"])
        &
        (out["target_ssim"] <= out["candidate_max_median_ssim"])
    )

    return out


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_slice = []
    all_candidates = []

    for dataset_name, root in DATASETS.items():

        slice_df, candidate_df = calibrate_dataset(
            dataset_name,
            root,
        )

        all_slice.append(slice_df)
        all_candidates.append(candidate_df)

    slice_results = pd.concat(
        all_slice,        ignore_index=True,
    )

    candidate_summary = pd.concat(
        all_candidates,
        ignore_index=True,
    )

    selected = select_calibrated_parameters(
        candidate_summary
    )

    selected = add_reachability_diagnostics(
        candidate_summary,
        selected,
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    slice_results.to_csv(
        OUTPUT_ROOT
        / "calibration_slice_level_ssim_pass2.csv",
        index=False,
    )

    candidate_summary.to_csv(
        OUTPUT_ROOT
        / "calibration_candidate_summary_subject_level_pass2.csv",
        index=False,
    )

    selected.to_csv(
        OUTPUT_ROOT
        / "calibrated_parameters_pass2.csv",
        index=False,
    )

    failures = selected[
        ~selected["within_tolerance"]
    ].copy()

    failures.to_csv(
        OUTPUT_ROOT
        / "calibration_conditions_outside_tolerance.csv",
        index=False,
    )

    config = {
        "global_seed": GLOBAL_SEED,
        "targets": TARGETS,
        "target_tolerance": TARGET_TOLERANCE,
        "datasets": {
            k: str(v)
            for k, v in DATASETS.items()
        },
        "candidate_grids": {
            k: [
                parameter_string(x)
                for x in values
            ]
            for k, values in CANDIDATES.items()
        },
        "aggregation": (
            "SSIM per slice -> mean SSIM per subject -> "
            "median across training subjects"
        ),
        "randomness": {
            "rician_noise": (
                "same base Gaussian fields reused across candidate sigmas"
            ),
            "rotation": (
                "same deterministic sign reused across candidate angles"
            ),
            "brightness_contrast": (
                "deterministic balanced four-way brightness/contrast sign design"
            ),
            "motion": (
                "same nested k-space row order and displacement field reused "
                "across candidate strengths"
            ),
            "gaussian_blur": "deterministic",
        },
        "selection_rule": (
            "Within each dataset/distortion, select candidate closest "
            "to each target SSIM while enforcing increasing candidate "
            "severity from mild to moderate to severe."
        ),
    }

    with open(
        OUTPUT_ROOT / "calibration_config_pass2.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("CALIBRATED PARAMETERS — PASS 2")
    print("=" * 100)

    display_cols = [
        "dataset",
        "distortion",
        "severity",
        "target_ssim",
        "selected_parameter",
        "achieved_median_subject_ssim",
        "absolute_target_error",
        "within_tolerance",
        "target_inside_candidate_ssim_range",
    ]

    print(
        selected[display_cols]
        .round(5)
        .to_string(index=False)
    )

    print()
    print("=" * 100)

    if failures.empty:
        print(
            "ALL 30 DATASET × DISTORTION × SEVERITY "
            "CALIBRATIONS ARE WITHIN TOLERANCE."
        )
    else:
        print(
            f"{len(failures)} calibration conditions remain "
            f"outside ±{TARGET_TOLERANCE:.2f} SSIM tolerance."
        )
        print(
            "Do NOT generate the calibrated test set yet."
        )

    print("=" * 100)
    print()
    print(f"Results saved to:\n{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()