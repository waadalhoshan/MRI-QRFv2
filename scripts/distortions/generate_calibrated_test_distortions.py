#!/usr/bin/env python3
"""
MRI-QRF calibrated held-out test distortion generator

Purpose
-------
Generate the FINAL dataset-calibrated test corruptions using the frozen
calibrated_parameters_FINAL.csv table.

Important:
- Uses CLEAN HELD-OUT TEST IMAGES ONLY.
- Does NOT retrain any model.
- Does NOT recalibrate parameters on the test set.
- Preserves the original fixed-corruption folders.
- Uses the same corruption mechanics / deterministic randomization as the
  calibration passes, so the frozen parameters are applied consistently.
- Saves a manifest and a post-hoc test-set SSIM summary for verification only.

Expected clean inputs:
    D:\OASIS_1\clean_128\test
    D:\OASIS_2\clean_128\test

Frozen calibration table:
    D:\MRI-QRF_calibrated\results\calibration_pass3\
        calibrated_parameters_FINAL.csv

Outputs:
    D:\OASIS_1\distortions_128_calibrated
    D:\OASIS_2\distortions_128_calibrated

Output structure:
    distortions_128_calibrated\
        motion\
            mild\
            moderate\
            severe\
        rician_noise\
            ...
        rotation\
            ...
        gaussian_blur\
            ...
        brightness_contrast\
            ...

The class/subject/image structure under the clean test root is preserved.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
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
    "OASIS_1": {
        "clean_test": Path(r"D:\OASIS_1\clean_128\test"),
        "output": Path(r"D:\OASIS_1\distortions_128_calibrated"),
    },
    "OASIS_2": {
        "clean_test": Path(r"D:\OASIS_2\clean_128\test"),
        "output": Path(r"D:\OASIS_2\distortions_128_calibrated"),
    },
}

CALIBRATION_TABLE = Path(
    r"D:\MRI-QRF_calibrated\results\calibration_pass3"
    r"\calibrated_parameters_FINAL.csv"
)

RESULT_ROOT = Path(
    r"D:\MRI-QRF_calibrated\results\generated_test_distortions"
)

GLOBAL_SEED = 42

DISTORTIONS = [
    "motion",
    "rician_noise",
    "rotation",
    "gaussian_blur",
    "brightness_contrast",
]

SEVERITIES = [
    "mild",
    "moderate",
    "severe",
]

EXPECTED_TARGETS = {
    "mild": 0.95,
    "moderate": 0.85,
    "severe": 0.70,
}

# Safety switch:
# False = stop if output folders already contain PNGs.
# True  = delete and regenerate the calibrated distortion folders.
OVERWRITE_EXISTING = False


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


def save_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    out = np.rint(
        np.clip(arr, 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    Image.fromarray(
        out,
        mode="L",
    ).save(
        path,
        "PNG",
    )


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


def get_subject_id(
    path: Path,
    root: Path,
) -> str:
    rel = path.relative_to(root)
    parts = rel.parts

    # Expected:
    # <class>/<subject>/<slice>.png
    if len(parts) < 3:
        raise RuntimeError(
            f"Unexpected test path structure under {root}: {rel}"
        )

    return parts[1]


def parse_parameter(
    distortion: str,
    raw_value,
):
    """
    Convert selected_parameter from CSV back to the correct Python type.
    """
    text = str(raw_value).strip()

    if distortion in {
        "motion",
        "brightness_contrast",
    }:
        value = ast.literal_eval(text)

        if not (
            isinstance(value, tuple)
            and len(value) == 2
        ):
            raise ValueError(
                f"Expected tuple parameter for {distortion}, got {text}"
            )

        if distortion == "motion":
            return (
                float(value[0]),
                int(round(float(value[1]))),
            )

        return (
            float(value[0]),
            float(value[1]),
        )

    return float(text)


# ============================================================
# CORRUPTION FUNCTIONS
# ============================================================

def apply_rician(
    x: np.ndarray,
    sigma: float,
    dataset: str,
    rel: str,
) -> np.ndarray:
    """
    Same common-random-number design as calibration:
    one deterministic pair of standard-normal fields per image.
    """
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


def apply_rotation(
    x: np.ndarray,
    magnitude_deg: float,
    dataset: str,
    rel: str,
) -> np.ndarray:
    """
    One deterministic +/- direction per image, shared conceptually
    across severity levels.
    """
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


def apply_gaussian_blur(
    x: np.ndarray,
    sigma: float,
) -> np.ndarray:

    y = gaussian_filter(
        x,
        sigma=sigma,
        mode="nearest",
    )

    return np.clip(
        y,
        0.0,
        1.0,
    )


def brightness_contrast_signs(
    dataset: str,
    rel: str,
) -> tuple[float, float]:
    """
    Deterministic balanced four-way sign design:
        (+brightness, +contrast)
        (+brightness, -contrast)
        (-brightness, +contrast)
        (-brightness, -contrast)
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


def apply_brightness_contrast(
    x: np.ndarray,
    bmag: float,
    cmag: float,
    dataset: str,
    rel: str,
) -> np.ndarray:

    bsign, csign = (
        brightness_contrast_signs(
            dataset,
            rel,
        )
    )

    b = bsign * bmag
    c = csign * cmag

    y = (
        (x - 0.5)
        * (1.0 + c)
        + 0.5
        + b
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
    """
    Same nested motion randomization as calibration:
    - deterministic random row ordering
    - deterministic U(-1,1) displacement values
    - stronger candidates use prefixes/scaling of same structure
    """
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


def apply_motion(
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


def apply_corruption(
    x: np.ndarray,
    distortion: str,
    parameter,
    dataset: str,
    rel: str,
) -> np.ndarray:

    if distortion == "motion":

        fraction, max_disp = parameter

        return apply_motion(
            x=x,
            fraction=float(fraction),
            max_disp=int(max_disp),
            dataset=dataset,
            rel=rel,
        )

    if distortion == "rician_noise":

        return apply_rician(
            x=x,
            sigma=float(parameter),
            dataset=dataset,
            rel=rel,
        )

    if distortion == "rotation":

        return apply_rotation(
            x=x,
            magnitude_deg=float(parameter),
            dataset=dataset,
            rel=rel,
        )

    if distortion == "gaussian_blur":

        return apply_gaussian_blur(
            x=x,
            sigma=float(parameter),
        )

    if distortion == "brightness_contrast":

        bmag, cmag = parameter

        return apply_brightness_contrast(
            x=x,
            bmag=float(bmag),
            cmag=float(cmag),
            dataset=dataset,
            rel=rel,
        )

    raise ValueError(
        f"Unknown distortion: {distortion}"
    )


# ============================================================
# CALIBRATION TABLE VALIDATION
# ============================================================

def load_and_validate_calibration_table() -> pd.DataFrame:

    if not CALIBRATION_TABLE.exists():
        raise FileNotFoundError(
            f"Calibration table not found:\n{CALIBRATION_TABLE}"
        )

    df = pd.read_csv(
        CALIBRATION_TABLE
    )

    required = {
        "dataset",
        "distortion",
        "severity",
        "target_ssim",
        "selected_parameter",
        "achieved_median_subject_ssim",
        "absolute_target_error",
        "within_tolerance",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Calibration table is missing columns: {sorted(missing)}"
        )

    expected_rows = (
        len(DATASETS)
        * len(DISTORTIONS)
        * len(SEVERITIES)
    )

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} calibration rows, found {len(df)}."
        )

    # Every expected dataset/distortion/severity combination must appear once.
    expected = {
        (dataset, distortion, severity)
        for dataset in DATASETS
        for distortion in DISTORTIONS
        for severity in SEVERITIES
    }

    observed = set(
        zip(
            df["dataset"],
            df["distortion"],
            df["severity"],
        )
    )

    if observed != expected:
        missing_conditions = sorted(
            expected - observed
        )
        extra_conditions = sorted(
            observed - expected
        )

        raise ValueError(
            "Calibration condition mismatch.\n"
            f"Missing: {missing_conditions}\n"
            f"Extra: {extra_conditions}"
        )

    duplicates = (
        df
        .groupby(
            [
                "dataset",
                "distortion",
                "severity",
            ]
        )
        .size()
    )

    if (duplicates != 1).any():
        raise ValueError(
            "Duplicate calibration conditions found."
        )

    # Verify targets.
    for _, row in df.iterrows():

        expected_target = (
            EXPECTED_TARGETS[
                row["severity"]
            ]
        )

        if not np.isclose(
            float(row["target_ssim"]),
            expected_target,
            atol=1e-12,
        ):
            raise ValueError(
                f"Unexpected target for "
                f"{row['dataset']} / "
                f"{row['distortion']} / "
                f"{row['severity']}: "
                f"{row['target_ssim']}"
            )

    # Require all frozen calibrations to be accepted.
    # Recompute from error instead of trusting CSV boolean parsing.
    bad = df[
        df["absolute_target_error"].astype(float)
        > 0.0200001
    ]

    if not bad.empty:
        raise RuntimeError(
            "Calibration table contains conditions outside ±0.02 SSIM. "
            "Do not generate test corruptions."
        )

    return df


def build_parameter_lookup(
    calibration_df: pd.DataFrame,
):
    lookup = {}

    for _, row in calibration_df.iterrows():

        key = (
            str(row["dataset"]),
            str(row["distortion"]),
            str(row["severity"]),
        )

        lookup[key] = {
            "parameter": parse_parameter(
                str(row["distortion"]),
                row["selected_parameter"],
            ),
            "parameter_text": str(
                row["selected_parameter"]
            ),
            "target_ssim": float(
                row["target_ssim"]
            ),
            "training_achieved_median_ssim": float(
                row["achieved_median_subject_ssim"]
            ),
        }

    return lookup


# ============================================================
# OUTPUT SAFETY
# ============================================================

def prepare_output_root(
    output_root: Path,
) -> None:

    if output_root.exists():

        existing_pngs = list(
            output_root.rglob("*.png")
        )

        if existing_pngs:

            if not OVERWRITE_EXISTING:
                raise RuntimeError(
                    f"Output folder already contains "
                    f"{len(existing_pngs)} PNG files:\n"
                    f"{output_root}\n\n"
                    f"Set OVERWRITE_EXISTING=True only if you "
                    f"intentionally want to delete and regenerate it."
                )

            shutil.rmtree(
                output_root
            )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# GENERATION
# ============================================================

def generate_dataset(
    dataset_name: str,
    clean_root: Path,
    output_root: Path,
    parameter_lookup: dict,
) -> pd.DataFrame:

    print()
    print("=" * 100)
    print(
        f"GENERATING CALIBRATED TEST DISTORTIONS: {dataset_name}"
    )
    print("=" * 100)

    if not clean_root.is_dir():
        raise FileNotFoundError(
            f"Clean test root not found:\n{clean_root}"
        )

    files = sorted(
        clean_root.rglob("*.png")
    )

    if not files:
        raise RuntimeError(
            f"No test PNGs found under:\n{clean_root}"
        )

    subject_ids = {
        get_subject_id(
            p,
            clean_root,
        )
        for p in files
    }

    print(
        f"Clean test images:   {len(files)}"
    )
    print(
        f"Clean test subjects: {len(subject_ids)}"
    )
    print(        f"Expected outputs:    "
        f"{len(files) * len(DISTORTIONS) * len(SEVERITIES)}"
    )

    prepare_output_root(
        output_root
    )

    rows = []

    for distortion in DISTORTIONS:

        print()
        print(
            f"Distortion: {distortion}"
        )

        for severity in SEVERITIES:

            info = parameter_lookup[
                (
                    dataset_name,
                    distortion,
                    severity,
                )
            ]

            parameter = info[
                "parameter"
            ]

            print(
                f"  {severity:8s} "
                f"parameter={info['parameter_text']} "
                f"| target SSIM={info['target_ssim']:.2f}"
            )

            for i, src in enumerate(files, start=1):

                rel_path = (
                    src.relative_to(
                        clean_root
                    )
                )

                rel = (
                    rel_path.as_posix()
                )

                subject_id = (
                    get_subject_id(
                        src,
                        clean_root,
                    )
                )

                x = load_image(
                    src
                )

                y = apply_corruption(
                    x=x,
                    distortion=distortion,
                    parameter=parameter,
                    dataset=dataset_name,
                    rel=rel,
                )

                dst = (
                    output_root
                    / distortion
                    / severity
                    / rel_path
                )

                save_image(
                    y,
                    dst,
                )

                # Read back saved PNG so SSIM reflects exactly the
                # quantized file that will be used by the models.
                y_saved = load_image(
                    dst
                )

                ssim_value = (
                    compute_ssim(
                        x,
                        y_saved,
                    )
                )

                rows.append({
                    "dataset":
                        dataset_name,

                    "distortion":
                        distortion,

                    "severity":
                        severity,

                    "target_ssim":
                        info[
                            "target_ssim"
                        ],

                    "selected_parameter":
                        info[
                            "parameter_text"
                        ],

                    "training_calibration_median_subject_ssim":
                        info[
                            "training_achieved_median_ssim"
                        ],

                    "subject_id":
                        subject_id,

                    "relative_source":
                        rel,

                    "source_path":
                        str(src),

                    "output_path":
                        str(dst),

                    "test_image_ssim":
                        ssim_value,
                })

    manifest = pd.DataFrame(
        rows
    )

    expected_count = (
        len(files)
        * len(DISTORTIONS)
        * len(SEVERITIES)
    )

    if len(manifest) != expected_count:
        raise RuntimeError(
            f"Generation count mismatch for {dataset_name}: "
            f"expected {expected_count}, got {len(manifest)}"
        )

    actual_pngs = len(
        list(
            output_root.rglob(
                "*.png"
            )
        )
    )

    if actual_pngs != expected_count:
        raise RuntimeError(
            f"Disk PNG count mismatch for {dataset_name}: "
            f"expected {expected_count}, found {actual_pngs}"
        )

    print()
    print(
        f"Generated successfully: {actual_pngs} PNGs"
    )

    return manifest


# ============================================================
# TEST-SET SSIM VERIFICATION
# ============================================================

def build_test_ssim_summary(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # First balance the 20 slices within each subject.
    subject_summary = (
        manifest
        .groupby(
            [
                "dataset",
                "distortion",
                "severity",
                "target_ssim",
                "selected_parameter",
                "training_calibration_median_subject_ssim",
                "subject_id",
            ],
            as_index=False,
        )
        .agg(
            mean_slice_ssim=(
                "test_image_ssim",
                "mean",
            ),
            n_slices=(
                "test_image_ssim",
                "size",
            ),
        )
    )

    # Then summarize across subjects exactly as in calibration.
    condition_summary = (
        subject_summary
        .groupby(
            [
                "dataset",
                "distortion",
                "severity",
                "target_ssim",
                "selected_parameter",
                "training_calibration_median_subject_ssim",
            ],
            as_index=False,
        )
        .agg(
            n_subjects=(
                "subject_id",
                "nunique",
            ),
            test_median_subject_ssim=(
                "mean_slice_ssim",
                "median",
            ),
            test_mean_subject_ssim=(
                "mean_slice_ssim",
                "mean",
            ),
            test_sd_subject_ssim=(
                "mean_slice_ssim",
                "std",
            ),
            test_q1_subject_ssim=(
                "mean_slice_ssim",
                lambda x:
                    float(
                        np.percentile(
                            x,
                            25,
                        )
                    ),
            ),
            test_q3_subject_ssim=(
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

    condition_summary[
        "test_absolute_target_error"
    ] = np.abs(
        condition_summary[
            "test_median_subject_ssim"
        ]
        - condition_summary[
            "target_ssim"
        ]
    )

    condition_summary[
        "training_to_test_median_ssim_shift"
    ] = (
        condition_summary[
            "test_median_subject_ssim"
        ]
        - condition_summary[
            "training_calibration_median_subject_ssim"
        ]
    )

    return (
        subject_summary,
        condition_summary,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration_df = (
        load_and_validate_calibration_table()
    )

    parameter_lookup = (
        build_parameter_lookup(
            calibration_df
        )
    )

    all_manifests = []

    for dataset_name, cfg in DATASETS.items():

        manifest = generate_dataset(
            dataset_name=dataset_name,
            clean_root=cfg["clean_test"],
            output_root=cfg["output"],
            parameter_lookup=parameter_lookup,
        )

        all_manifests.append(
            manifest
        )

        manifest.to_csv(
            RESULT_ROOT
            / f"{dataset_name}_calibrated_test_manifest.csv",
            index=False,
        )

    full_manifest = pd.concat(
        all_manifests,
        ignore_index=True,
    )

    full_manifest.to_csv(
        RESULT_ROOT
        / "calibrated_test_manifest_ALL.csv",
        index=False,
    )

    subject_ssim, condition_ssim = (
        build_test_ssim_summary(
            full_manifest
        )
    )

    subject_ssim.to_csv(
        RESULT_ROOT
        / "calibrated_test_ssim_subject_level.csv",
        index=False,
    )

    condition_ssim.to_csv(
        RESULT_ROOT
        / "calibrated_test_ssim_condition_summary.csv",
        index=False,
    )

    # Save a copy of the frozen calibration table used for generation.
    calibration_df.to_csv(
        RESULT_ROOT
        / "frozen_calibrated_parameters_used.csv",
        index=False,
    )

    run_config = {
        "global_seed":
            GLOBAL_SEED,

        "calibration_table":
            str(
                CALIBRATION_TABLE
            ),

        "datasets":
            DATASETS,

        "distortions":
            DISTORTIONS,

        "severities":
            SEVERITIES,

        "important_note": (
            "Test-set SSIM values are verification only. "
            "No parameter is recalibrated or changed using test data."
        ),
    }

    # Convert Path objects for JSON.
    serializable_config = {
        "global_seed":
            GLOBAL_SEED,

        "calibration_table":
            str(
                CALIBRATION_TABLE
            ),

        "datasets": {
            name: {
                key: str(value)
                for key, value
                in cfg.items()
            }
            for name, cfg
            in DATASETS.items()
        },

        "distortions":
            DISTORTIONS,

        "severities":
            SEVERITIES,

        "important_note": (
            "Test-set SSIM values are verification only. "
            "No parameter is recalibrated or changed using test data."
        ),
    }

    with open(
        RESULT_ROOT
        / "generation_config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            serializable_config,
            f,
            indent=2,
        )

    print()
    print("=" * 110)
    print(
        "CALIBRATED TEST-SET SSIM VERIFICATION"
    )
    print("=" * 110)

    display_cols = [
        "dataset",
        "distortion",
        "severity",
        "target_ssim",
        "selected_parameter",
        "training_calibration_median_subject_ssim",
        "test_median_subject_ssim",
        "test_absolute_target_error",
        "training_to_test_median_ssim_shift",
    ]

    print(
        condition_ssim[
            display_cols
        ]
        .round(5)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 110)
    print("DONE")
    print("=" * 110)
    print(
        "The frozen calibrated parameters were applied to the "
        "held-out test images without recalibration."
    )
    print()
    print(
        f"Verification outputs saved to:\n{RESULT_ROOT}"
    )


if __name__ == "__main__":
    main()