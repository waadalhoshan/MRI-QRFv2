#!/usr/bin/env python3
"""
MRI-QRF dataset-calibrated corruption calibration — targeted Pass 3

Purpose
-------
Refine ONLY the six Pass-2 conditions that remained outside the ±0.02 SSIM
tolerance, while preserving all successful Pass-2 selections.

Targets:
    mild      SSIM = 0.95
    moderate  SSIM = 0.85
    severe    SSIM = 0.70

This script:
1. Reads the Pass-2 calibrated parameter table.
2. Re-runs calibration only for the six unresolved conditions.
3. Uses ALL clean 128x128 training slices.
4. Uses the same deterministic/common-random-number logic as Pass 2.
5. Selects the candidate closest to the target SSIM.
6. Replaces only those six rows in the Pass-2 table.
7. Writes a FINAL 30-condition calibration table.

It does NOT touch the test set and does NOT train any model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import rotate
from skimage.metrics import structural_similarity as ssim


# ============================================================
# PATHS / CONFIG
# ============================================================

DATASETS = {
    "OASIS_1": Path(r"D:\OASIS_1\clean_128\train"),
    "OASIS_2": Path(r"D:\OASIS_2\clean_128\train"),
}

PASS2_ROOT = Path(
    r"D:\MRI-QRF_calibrated\results\calibration_pass2"
)

PASS2_PARAMETERS = (
    PASS2_ROOT / "calibrated_parameters_pass2.csv"
)

OUTPUT_ROOT = Path(
    r"D:\MRI-QRF_calibrated\results\calibration_pass3"
)

GLOBAL_SEED = 42
TARGET_TOLERANCE = 0.02

TARGETS = {
    "mild": 0.95,
    "moderate": 0.85,
    "severe": 0.70,
}


# ============================================================
# TARGETED PASS-3 GRIDS
# ============================================================
#
# These are centered around the exact Pass-2 gaps:
#
# OASIS-1 Rician:
#   sigma 0.005 -> SSIM ~0.8755
#   sigma 0.0075 -> SSIM ~0.7913
#   sigma 0.010 -> SSIM ~0.7248
#   sigma 0.015 -> SSIM ~0.6390
#
# OASIS-1 rotation:
#   2 deg -> ~0.8902
#   3 deg -> ~0.8201
#
# OASIS-2 motion:
#   (0.15,2) -> ~0.8951
#   (0.20,3) -> ~0.8299
#   (0.30,4) -> ~0.7391
#   (0.40,6) -> ~0.6187
#
# OASIS-2 rotation:
#   2 deg -> ~0.7343
#   3 deg -> ~0.6269
#

REFINEMENT_GRIDS = {
    ("OASIS_1", "rician_noise", "moderate"): [
        0.0050,
        0.0055,
        0.0060,
        0.0065,
        0.0070,
        0.0075,
    ],

    ("OASIS_1", "rician_noise", "severe"): [
        0.0100,
        0.0105,
        0.0110,
        0.0115,
        0.0120,
        0.0125,
        0.0130,
        0.0135,
        0.0140,
        0.0145,
        0.0150,
    ],

    ("OASIS_1", "rotation", "moderate"): [
        2.00,
        2.10,
        2.20,
        2.30,
        2.40,
        2.50,
        2.60,
        2.70,
        2.80,
        2.90,
        3.00,
    ],

    # Motion is discrete in both line fraction and displacement.
    # We therefore refine with nearby combinations rather than pretending
    # it is a perfectly continuous scalar parameter.
    ("OASIS_2", "motion", "moderate"): [
        (0.150, 2),
        (0.160, 2),
        (0.170, 2),
        (0.180, 2),
        (0.190, 2),
        (0.200, 2),
        (0.170, 3),
        (0.180, 3),
        (0.190, 3),
        (0.200, 3),
        (0.210, 3),
        (0.220, 3),
        (0.230, 3),
    ],

    ("OASIS_2", "motion", "severe"): [
        (0.280, 4),
        (0.300, 4),
        (0.320, 4),
        (0.340, 4),
        (0.360, 4),
        (0.300, 5),
        (0.320, 5),
        (0.340, 5),
        (0.360, 5),
        (0.380, 5),
        (0.350, 6),
        (0.375, 6),
        (0.400, 6),
    ],

    ("OASIS_2", "rotation", "severe"): [
        2.00,
        2.10,
        2.20,
        2.30,
        2.40,
        2.50,
        2.60,
        2.70,
        2.80,
        2.90,
        3.00,
    ],
}


# ============================================================
# HELPERS
# ============================================================

def stable_seed(*parts) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        return np.asarray(img, dtype=np.float32) / 255.0


def get_subject_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = rel.parts

    # Expected layout:
    # <class>/<subject>/<slice>.png
    if len(parts) < 3:
        raise RuntimeError(
            f"Unexpected path structure under {root}: {rel}"
        )

    return parts[1]


def parameter_string(parameter) -> str:
    if isinstance(parameter, tuple):
        return "(" + ",".join(
            f"{float(v):g}" for v in parameter
        ) + ")"

    return f"{float(parameter):g}"


def compute_ssim(
    clean: np.ndarray,
    corrupted: np.ndarray,
) -> float:
    return float(
        ssim(
            clean,
            corrupted,
            data_range=1.0,
        )
    )


# ============================================================
# SAME COMMON-RANDOM-NUMBER LOGIC AS PASS 2
# ============================================================

def apply_rician_common(
    x: np.ndarray,
    sigma: float,
    dataset: str,
    rel: str,
) -> np.ndarray:

    rng = np.random.default_rng(
        stable_seed(
            GLOBAL_SEED,
            dataset,
            rel,
            "rician_noise",
        )
    )

    z1 = rng.normal(
        0.0,
        1.0,
        x.shape,
    )

    z2 = rng.normal(
        0.0,
        1.0,
        x.shape,
    )

    y = np.sqrt(
        (x + sigma * z1) ** 2
        + (sigma * z2) ** 2
    )

    return np.clip(
        y,
        0.0,
        1.0,
    )


def apply_rotation_common(
    x: np.ndarray,
    magnitude_deg: float,
    dataset: str,
    rel: str,
) -> np.ndarray:

    rng = np.random.default_rng(
        stable_seed(
            GLOBAL_SEED,
            dataset,
            rel,
            "rotation",
        )
    )

    sign = float(
        rng.choice(
            [-1.0, 1.0]
        )
    )

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

    return np.clip(
        y,
        0.0,
        1.0,
    )


def motion_random_structure(
    h: int,
    dataset: str,
    rel: str,
):
    rng = np.random.default_rng(
        stable_seed(
            GLOBAL_SEED,
            dataset,
            rel,
            "motion",
        )
    )

    row_order = rng.permutation(h)
    ux = rng.uniform(
        -1.0,
        1.0,
        size=h,
    )
    uy = rng.uniform(
        -1.0,
        1.0,
        size=h,
    )

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

    row_order, ux, uy = (
        motion_random_structure(
            h,
            dataset,
            rel,
        )
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

        dx = int(
            np.rint(
                ux[row] * max_disp
            )
        )

        dy = int(
            np.rint(
                uy[row] * max_disp
            )
        )

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

    return np.clip(
        y,
        0.0,
        1.0,
    )


def apply_targeted_corruption(
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

    if distortion == "motion":
        fraction, max_disp = parameter

        return apply_motion_common(
            x,
            float(fraction),
            int(max_disp),
            dataset,
            rel,
        )

    raise ValueError(
        f"Unsupported Pass-3 distortion: {distortion}"
    )


# ============================================================
# CALIBRATE ONE TARGET CONDITION
# ============================================================

def calibrate_condition(
    dataset_name: str,
    distortion: str,
    severity: str,
    candidate_grid,
) -> tuple[pd.DataFrame, dict]:

    root = DATASETS[dataset_name]

    files = sorted(
        root.rglob("*.png")
    )

    if not files:
        raise RuntimeError(
            f"No PNG files found under {root}"
        )

    target = TARGETS[severity]

    print()
    print("=" * 90)
    print(
        f"{dataset_name} | "
        f"{distortion} | "
        f"{severity} | "
        f"target={target:.2f}"
    )
    print("=" * 90)
    print(f"Images: {len(files)}")
    print(
        f"Subjects: "
        f"{len(set(get_subject_id(p, root) for p in files))}"
    )

    rows = []

    for candidate_index, parameter in enumerate(candidate_grid):

        print(
            f"  {candidate_index + 1:02d}/"
            f"{len(candidate_grid):02d} "
            f"{parameter}"
        )

        for path in files:

            rel = path.relative_to(
                root
            ).as_posix()

            subject_id = (
                get_subject_id(
                    path,
                    root,
                )
            )

            x = load_image(path)

            y = apply_targeted_corruption(
                x=x,
                distortion=distortion,
                parameter=parameter,
                dataset=dataset_name,
                rel=rel,
            )

            value = compute_ssim(
                x,
                y,
            )

            rows.append({
                "dataset":
                    dataset_name,

                "distortion":
                    distortion,

                "severity":
                    severity,

                "target_ssim":
                    target,

                "candidate_index":
                    candidate_index,

                "parameter":
                    parameter_string(
                        parameter
                    ),

                "subject_id":
                    subject_id,

                "relative_path":
                    rel,

                "ssim":
                    value,
            })

    slice_df = pd.DataFrame(
        rows
    )

    subject_df = (
        slice_df
        .groupby(
            [
                "dataset",
                "distortion",
                "severity",
                "target_ssim",
                "candidate_index",
                "parameter",
                "subject_id",
            ],
            as_index=False,
        )
        .agg(
            mean_slice_ssim=(
                "ssim",
                "mean",
            )
        )
    )

    candidate_df = (
        subject_df
        .groupby(
            [
                "dataset",
                "distortion",
                "severity",
                "target_ssim",
                "candidate_index",
                "parameter",
            ],
            as_index=False,
        )
        .agg(
            n_subjects=(
                "subject_id",
                "nunique",
            ),

            median_subject_ssim=(
                "mean_slice_ssim",
                "median",
            ),

            mean_subject_ssim=(
                "mean_slice_ssim",
                "mean",
            ),

            sd_subject_ssim=(
                "mean_slice_ssim",
                "std",
            ),

            q1_subject_ssim=(
                "mean_slice_ssim",
                lambda x:
                    float(
                        np.percentile(
                            x,
                            25,
                        )
                    ),
            ),

            q3_subject_ssim=(
                "mean_slice_ssim",
                lambda x:
                    float(
                        np.percentile(
                            x,
                            75,
                        )
                    ),
            ),
        )
    )

    candidate_df[
        "absolute_target_error"
    ] = np.abs(
        candidate_df[
            "median_subject_ssim"
        ]
        - target
    )

    best = (
        candidate_df
        .sort_values(
            [
                "absolute_target_error",
                "candidate_index",
            ]
        )
        .iloc[0]
    )

    result = {
        "dataset":
            dataset_name,

        "distortion":
            distortion,

        "severity":
            severity,

        "target_ssim":
            target,

        "selected_candidate_index":
            int(
                best[
                    "candidate_index"
                ]
            ),

        "selected_parameter":
            best[
                "parameter"
            ],

        "achieved_median_subject_ssim":
            float(
                best[
                    "median_subject_ssim"
                ]
            ),

        "achieved_mean_subject_ssim":
            float(
                best[
                    "mean_subject_ssim"
                ]
            ),

        "achieved_sd_subject_ssim":
            float(
                best[
                    "sd_subject_ssim"
                ]
            ),

        "absolute_target_error":
            float(
                best[
                    "absolute_target_error"
                ]
            ),

        "within_tolerance":
            bool(
                best[
                    "absolute_target_error"
                ]
                <= TARGET_TOLERANCE
            ),

        "n_subjects":
            int(
                best[
                    "n_subjects"
                ]
            ),
    }

    print()
    print(
        "Selected:",
        result[
            "selected_parameter"
        ],
        "| achieved median SSIM =",
        f"{result['achieved_median_subject_ssim']:.5f}",
        "| error =",
        f"{result['absolute_target_error']:.5f}",
        "| within tolerance =",
        result[
            "within_tolerance"
        ],
    )

    return candidate_df, result


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not PASS2_PARAMETERS.exists():
        raise FileNotFoundError(
            f"Pass-2 table not found:\n"
            f"{PASS2_PARAMETERS}"
        )

    pass2 = pd.read_csv(
        PASS2_PARAMETERS
    )

    expected_rows = 30

    if len(pass2) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows in Pass-2 table, "
            f"found {len(pass2)}."
        )

    all_candidate_results = []
    replacements = []

    for (        dataset_name,
        distortion,
        severity
    ), candidate_grid in REFINEMENT_GRIDS.items():

        candidate_df, result = (
            calibrate_condition(
                dataset_name=dataset_name,
                distortion=distortion,
                severity=severity,
                candidate_grid=candidate_grid,
            )
        )

        all_candidate_results.append(
            candidate_df
        )

        replacements.append(
            result
        )

    pass3_refined = pd.DataFrame(
        replacements
    )

    pass3_candidates = pd.concat(
        all_candidate_results,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Merge: preserve all successful Pass-2 rows and replace
    # only the six targeted rows.
    # --------------------------------------------------------

    final_df = pass2.copy()

    for _, new_row in pass3_refined.iterrows():

        mask = (
            (final_df["dataset"] == new_row["dataset"])
            &
            (final_df["distortion"] == new_row["distortion"])
            &
            (final_df["severity"] == new_row["severity"])
        )

        if mask.sum() != 1:
            raise RuntimeError(
                "Expected exactly one Pass-2 row for "
                f"{new_row['dataset']} / "
                f"{new_row['distortion']} / "
                f"{new_row['severity']}, "
                f"found {mask.sum()}."
            )

        # Update only fields recalculated in Pass 3.
        for col in [
            "target_ssim",
            "selected_candidate_index",
            "selected_parameter",
            "achieved_median_subject_ssim",
            "achieved_mean_subject_ssim",
            "achieved_sd_subject_ssim",
            "absolute_target_error",
            "within_tolerance",
            "n_subjects",
        ]:
            final_df.loc[
                mask,
                col,
            ] = new_row[col]

        # Pass-2 candidate-range diagnostics are no longer
        # meaningful for these refined grids.
        if (
            "candidate_min_median_ssim"
            in final_df.columns
        ):
            sub = pass3_candidates[
                (pass3_candidates["dataset"] == new_row["dataset"])
                &
                (pass3_candidates["distortion"] == new_row["distortion"])
                &
                (pass3_candidates["severity"] == new_row["severity"])
            ]

            final_df.loc[
                mask,
                "candidate_min_median_ssim",
            ] = sub[
                "median_subject_ssim"
            ].min()

            final_df.loc[
                mask,
                "candidate_max_median_ssim",
            ] = sub[
                "median_subject_ssim"
            ].max()

        if (
            "target_inside_candidate_ssim_range"
            in final_df.columns
        ):
            mn = float(
                final_df.loc[
                    mask,
                    "candidate_min_median_ssim",
                ].iloc[0]
            )

            mx = float(
                final_df.loc[
                    mask,
                    "candidate_max_median_ssim",
                ].iloc[0]
            )

            target = float(
                new_row[
                    "target_ssim"
                ]
            )

            final_df.loc[
                mask,
                "target_inside_candidate_ssim_range",
            ] = (
                mn <= target <= mx
            )

    # Ensure boolean field is clean.
    final_df[
        "within_tolerance"
    ] = (
        final_df[
            "absolute_target_error"
        ]
        <= TARGET_TOLERANCE
    )

    # Sort cleanly.
    distortion_order = {
        "motion": 0,
        "rician_noise": 1,
        "rotation": 2,
        "gaussian_blur": 3,
        "brightness_contrast": 4,
    }

    severity_order = {
        "mild": 0,
        "moderate": 1,
        "severe": 2,
    }

    final_df["_dist_order"] = (
        final_df[
            "distortion"
        ].map(
            distortion_order
        )
    )

    final_df["_sev_order"] = (
        final_df[
            "severity"
        ].map(
            severity_order
        )
    )

    final_df = (
        final_df
        .sort_values(
            [
                "dataset",
                "_dist_order",
                "_sev_order",
            ]
        )
        .drop(
            columns=[
                "_dist_order",
                "_sev_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    pass3_candidates.to_csv(
        OUTPUT_ROOT
        / "calibration_candidate_summary_subject_level_pass3.csv",
        index=False,
    )

    pass3_refined.to_csv(
        OUTPUT_ROOT
        / "calibrated_parameters_pass3_refined_only.csv",
        index=False,
    )

    final_df.to_csv(
        OUTPUT_ROOT
        / "calibrated_parameters_FINAL.csv",
        index=False,
    )

    remaining = final_df[
        ~final_df[
            "within_tolerance"
        ]
    ].copy()

    remaining.to_csv(
        OUTPUT_ROOT
        / "calibration_conditions_outside_tolerance_FINAL.csv",
        index=False,
    )

    config = {
        "global_seed":
            GLOBAL_SEED,

        "target_tolerance":
            TARGET_TOLERANCE,

        "targets":
            TARGETS,

        "pass2_source":
            str(
                PASS2_PARAMETERS
            ),

        "refined_conditions":
            [
                {
                    "dataset": k[0],
                    "distortion": k[1],
                    "severity": k[2],
                    "candidates": [
                        parameter_string(x)
                        for x in values
                    ],
                }
                for k, values
                in REFINEMENT_GRIDS.items()
            ],

        "rule": (
            "Only Pass-2 conditions outside ±0.02 SSIM tolerance "
            "were recalibrated. All Pass-2 conditions already within "
            "tolerance were preserved unchanged."
        ),
    }

    with open(
        OUTPUT_ROOT
        / "calibration_config_pass3.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("PASS-3 REFINED CONDITIONS")
    print("=" * 100)

    print(
        pass3_refined[
            [
                "dataset",
                "distortion",
                "severity",
                "target_ssim",
                "selected_parameter",
                "achieved_median_subject_ssim",
                "absolute_target_error",
                "within_tolerance",
            ]
        ]
        .round(5)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("FINAL CALIBRATION STATUS")
    print("=" * 100)

    n_good = int(
        final_df[
            "within_tolerance"
        ].sum()
    )

    print(
        f"Within tolerance: "
        f"{n_good}/{len(final_df)}"
    )

    if remaining.empty:

        print(
            "SUCCESS: all 30 dataset × distortion × severity "
            "conditions are within ±0.02 SSIM."
        )

        print(
            "The calibration table is ready for review before "
            "generating the held-out calibrated test distortions."
        )

    else:

        print(
            f"{len(remaining)} condition(s) still remain outside "
            f"±{TARGET_TOLERANCE:.2f} SSIM."
        )

        print()
        print(
            remaining[
                [
                    "dataset",
                    "distortion",
                    "severity",
                    "target_ssim",
                    "selected_parameter",
                    "achieved_median_subject_ssim",
                    "absolute_target_error",
                ]
            ]
            .round(5)
            .to_string(
                index=False
            )
        )

        print()
        print(
            "Do NOT generate the calibrated test set yet."
        )

    print()
    print(
        f"Results saved to:\n"
        f"{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()