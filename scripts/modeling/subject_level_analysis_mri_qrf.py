"""
MRI-QRF subject-level analysis
==============================

Purpose
-------
Aggregate the already-saved slice-level predictions from the classical
ML experiments (32x32, 64x64, 128x128) to the SUBJECT level.

No model retraining is performed.

For each subject and condition:
    - Random Forest: average class probabilities across that subject's slices.
    - Linear SVM: average class decision scores across that subject's slices.
    - Predict the class with the largest mean score.

Then calculate:
    - Accuracy
    - Macro-F1
    - Weighted-F1
    - Balanced accuracy
    - Per-class precision / recall / F1 / support
    - Confusion matrices
    - Subject-level ARR
    - Subject-level DRI
    - Subject-level CD-DRI
    - Mean ± SD across the three training seeds
    - Resolution comparison (32 vs 64 vs 128)

Expected result roots
---------------------
D:\MRI_QRF_results\ML_32
D:\MRI_QRF_results\ML_64
D:\MRI_QRF_results\ML_128

Each should contain:
predictions/
    OASIS_1/
        RandomForest/
            seed_13/
                clean_test.csv
                motion__mild.csv
                ...
        LinearSVM/
            ...
    OASIS_2/
        ...

Output
------
D:\MRI_QRF_results\subject_level_analysis
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


# ============================================================
# CONFIGURATION
# ============================================================

RESULT_ROOTS = {
    32: Path(r"D:\MRI_QRF_results\ML_32"),
    64: Path(r"D:\MRI_QRF_results\ML_64"),
    128: Path(r"D:\MRI_QRF_results\ML_128"),
}

OUTPUT_ROOT = Path(
    r"D:\MRI_QRF_results\subject_level_analysis"
)

CLASS_NAMES = [
    "Non-Demented",
    "Very Mild Dementia",
    "Mild Dementia",
    "Moderate Dementia",
]

CLASS_INDICES = np.arange(len(CLASS_NAMES))

EXPECTED_DATASETS = [
    "OASIS_1",
    "OASIS_2",
]

EXPECTED_MODELS = [
    "RandomForest",
    "LinearSVM",
]

EXPECTED_SEEDS = [
    13,
    47,
    101,
]

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

EXPECTED_CORRUPTED_CONDITIONS = [
    f"{distortion}__{severity}"
    for distortion in DISTORTIONS
    for severity in SEVERITIES
]


# ============================================================
# HELPERS
# ============================================================

def normalize_token(text: str) -> str:
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(text).lower(),
    )


EXPECTED_SCORE_COLUMNS = [
    f"score_{normalize_token(name)}"
    for name in CLASS_NAMES
]


def parse_seed(seed_dir_name: str) -> int:
    match = re.fullmatch(
        r"seed_(\d+)",
        seed_dir_name,
    )

    if not match:
        raise ValueError(
            f"Unexpected seed directory: {seed_dir_name}"
        )

    return int(match.group(1))


def parse_condition(condition_name: str):
    if condition_name == "clean_test":
        return "clean", "clean"

    if "__" not in condition_name:
        return "", ""

    distortion, severity = condition_name.split(
        "__",
        1,
    )

    return distortion, severity


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=CLASS_INDICES,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=CLASS_INDICES,
            average="weighted",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
    }


def validate_prediction_file(
    df: pd.DataFrame,
    csv_path: Path,
):
    required = {
        "subject_id",
        "y_true",
        "y_pred",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: "
            f"{sorted(missing)}"
        )

    score_cols_present = [
        c for c in EXPECTED_SCORE_COLUMNS
        if c in df.columns
    ]

    if len(score_cols_present) != len(EXPECTED_SCORE_COLUMNS):
        raise ValueError(
            f"{csv_path} does not contain all four class score columns.\n"
            f"Expected: {EXPECTED_SCORE_COLUMNS}\n"
            f"Found: {score_cols_present}\n"
            "The subject-level analysis requires the saved RF probabilities "
            "or LinearSVM decision scores."
        )


# ============================================================
# SUBJECT-LEVEL AGGREGATION
# ============================================================

def aggregate_subject_predictions(
    slice_df: pd.DataFrame,
    resolution: int,
    dataset: str,
    model: str,
    seed: int,
    condition: str,
) -> pd.DataFrame:
    """
    Average the four class-score columns across all slices belonging
    to each subject, then argmax the mean score.
    """

    validate_prediction_file(
        slice_df,
        Path(
            f"{resolution}/{dataset}/{model}/seed_{seed}/{condition}"
        ),
    )

    rows = []

    for subject_id, group in slice_df.groupby(
        "subject_id",
        sort=True,
    ):
        unique_true = sorted(
            pd.unique(group["y_true"])
        )

        if len(unique_true) != 1:
            raise ValueError(
                f"Subject {subject_id} has multiple y_true labels "
                f"in {dataset}/{model}/seed_{seed}/{condition}: "
                f"{unique_true}"
            )

        y_true = int(unique_true[0])

        mean_scores = (
            group[EXPECTED_SCORE_COLUMNS]
            .mean(axis=0)
            .to_numpy(dtype=float)
        )

        if not np.all(np.isfinite(mean_scores)):
            raise ValueError(
                f"Non-finite mean class scores for subject "
                f"{subject_id} in {condition}."
            )

        y_pred = int(
            np.argmax(mean_scores)
        )

        row = {
            "resolution": resolution,
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "condition": condition,
            "subject_id": subject_id,
            "n_slices": len(group),
            "y_true": y_true,
            "true_class_name": CLASS_NAMES[y_true],
            "y_pred": y_pred,
            "pred_class_name": CLASS_NAMES[y_pred],
            "correct": int(
                y_true == y_pred
            ),
        }

        for class_idx, class_name in enumerate(
            CLASS_NAMES
        ):
            col = (
                f"mean_score_"
                f"{normalize_token(class_name)}"
            )
            row[col] = float(
                mean_scores[class_idx]
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# SAVE SUPPORTING OUTPUTS
# ============================================================

def save_confusion_matrix(
    subject_df: pd.DataFrame,
    output_path: Path,
):
    cm = confusion_matrix(
        subject_df["y_true"],
        subject_df["y_pred"],
        labels=CLASS_INDICES,
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    cm_df.index.name = "true_class"
    cm_df.columns.name = "predicted_class"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cm_df.to_csv(
        output_path,
    )


def get_per_class_metrics(
    subject_df: pd.DataFrame,
) -> pd.DataFrame:
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            subject_df["y_true"],
            subject_df["y_pred"],
            labels=CLASS_INDICES,
            zero_division=0,
        )
    )

    return pd.DataFrame({
        "class_index": CLASS_INDICES,
        "class_name": CLASS_NAMES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_subject_prediction_frames = []
    all_condition_metric_rows = []
    all_per_class_frames = []

    found_resolutions = []

    # --------------------------------------------------------
    # Read every saved prediction CSV.
    # --------------------------------------------------------

    for resolution, result_root in RESULT_ROOTS.items():

        predictions_root = (
            result_root
            / "predictions"
        )

        if not predictions_root.exists():
            print(
                f"SKIPPING {resolution}x{resolution}: "
                f"{predictions_root} does not exist."
            )
            continue

        found_resolutions.append(
            resolution
        )

        print("\n")
        print("=" * 80)
        print(
            f"RESOLUTION: "
            f"{resolution} x {resolution}"
        )
        print("=" * 80)

        csv_files = sorted(
            predictions_root.rglob("*.csv")
        )

        if not csv_files:
            print(
                f"No prediction CSVs found under "
                f"{predictions_root}"
            )
            continue

        for csv_path in csv_files:

            # Expected:
            # predictions/<dataset>/<model>/seed_N/<condition>.csv
            rel = csv_path.relative_to(
                predictions_root
            )

            parts = rel.parts

            if len(parts) != 4:
                print(
                    f"Skipping unexpected path: {rel}"
                )
                continue

            dataset = parts[0]
            model = parts[1]
            seed = parse_seed(parts[2])
            condition = csv_path.stem

            if condition == "clean_val":
                # This analysis is for held-out TEST robustness.
                continue

            distortion, severity = (
                parse_condition(condition)
            )

            print(
                f"{resolution:>3} | "
                f"{dataset:<7} | "
                f"{model:<12} | "
                f"seed={seed:<3} | "
                f"{condition}"
            )

            slice_df = pd.read_csv(
                csv_path
            )

            subject_df = (
                aggregate_subject_predictions(
                    slice_df=slice_df,
                    resolution=resolution,
                    dataset=dataset,
                    model=model,
                    seed=seed,
                    condition=condition,
                )
            )

            subject_df.insert(
                5,
                "distortion",
                distortion,
            )

            subject_df.insert(
                6,
                "severity",
                severity,
            )

            all_subject_prediction_frames.append(
                subject_df
            )

            metrics = compute_metrics(
                subject_df["y_true"].to_numpy(),
                subject_df["y_pred"].to_numpy(),
            )

            metric_row = {
                "resolution": resolution,
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "condition": condition,
                "distortion": distortion,
                "severity": severity,
                "n_subjects": len(subject_df),
                **metrics,
            }

            all_condition_metric_rows.append(
                metric_row
            )

            # Per-class metrics.
            per_class = get_per_class_metrics(
                subject_df
            )

            per_class.insert(
                0,
                "resolution",
                resolution,
            )

            per_class.insert(
                1,
                "dataset",
                dataset,
            )

            per_class.insert(
                2,
                "model",
                model,
            )

            per_class.insert(
                3,
                "seed",
                seed,
            )

            per_class.insert(
                4,
                "condition",
                condition,
            )

            per_class.insert(
                5,
                "distortion",
                distortion,
            )

            per_class.insert(
                6,
                "severity",
                severity,
            )

            all_per_class_frames.append(
                per_class
            )

            # Confusion matrix.
            cm_path = (
                OUTPUT_ROOT
                / "confusion_matrices"
                / f"{resolution}x{resolution}"
                / dataset
                / model
                / f"seed_{seed}"
                / f"{condition}.csv"
            )

            save_confusion_matrix(
                subject_df,
                cm_path,
            )

    if not all_condition_metric_rows:
        raise RuntimeError(
            "No prediction results were found. "
            "Check RESULT_ROOTS at the top of the script."
        )

    # --------------------------------------------------------
    # Combine and save raw subject-level predictions.
    # --------------------------------------------------------

    subject_predictions = pd.concat(
        all_subject_prediction_frames,
        ignore_index=True,
    )

    subject_predictions.to_csv(
        OUTPUT_ROOT
        / "subject_level_all_predictions.csv",
        index=False,
    )

    per_class_all = pd.concat(
        all_per_class_frames,
        ignore_index=True,
    )

    per_class_all.to_csv(
        OUTPUT_ROOT
        / "subject_level_per_class_metrics.csv",
        index=False,
    )

    condition_results = pd.DataFrame(
        all_condition_metric_rows
    ).sort_values(
        [
            "resolution",
            "dataset",
            "model",
            "seed",
            "condition",
        ]
    )

    condition_results.to_csv(
        OUTPUT_ROOT
        / "subject_level_condition_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Clean results only.
    # --------------------------------------------------------

    clean_results = (
        condition_results[
            condition_results["condition"]
            == "clean_test"
        ]
        .copy()
        .sort_values(
            [
                "resolution",
                "dataset",
                "model",
                "seed",
            ]
        )
    )

    clean_results.to_csv(
        OUTPUT_ROOT
        / "subject_level_clean_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # ARR.
    # --------------------------------------------------------

    clean_baseline = clean_results[
        [
            "resolution",
            "dataset",
            "model",
            "seed",
            "accuracy",
            "macro_f1",
        ]
    ].rename(
        columns={
            "accuracy":
                "clean_accuracy",
            "macro_f1":
                "clean_macro_f1",
        }
    )

    corrupted = (
        condition_results[
            condition_results["condition"]
            != "clean_test"
        ]
        .copy()
    )

    arr = corrupted.merge(
        clean_baseline,
        on=[
            "resolution",
            "dataset",
            "model",
            "seed",
        ],
        how="left",
        validate="many_to_one",
    )

    arr["ARR"] = np.where(
        arr["clean_accuracy"] > 0,
        (
            arr["accuracy"]
            / arr["clean_accuracy"]
        ) * 100.0,
        np.nan,
    )

    arr["delta_accuracy"] = (
        arr["accuracy"]
        - arr["clean_accuracy"]
    )

    arr["delta_macro_f1"] = (
        arr["macro_f1"]
        - arr["clean_macro_f1"]
    )

    arr.to_csv(
        OUTPUT_ROOT
        / "subject_level_ARR.csv",
        index=False,
    )

    # --------------------------------------------------------
    # DRI.
    # --------------------------------------------------------

    dri = (
        arr
        .groupby(
            [
                "resolution",
                "dataset",
                "model",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            DRI=("ARR", "mean"),
            mean_corrupted_accuracy=(
                "accuracy",
                "mean",
            ),
            mean_corrupted_macro_f1=(
                "macro_f1",
                "mean",
            ),
            n_conditions=(
                "condition",
                "nunique",
            ),
        )
    )

    dri.to_csv(
        OUTPUT_ROOT
        / "subject_level_DRI.csv",
        index=False,
    )

    # --------------------------------------------------------
    # CD-DRI.
    # --------------------------------------------------------

    cd_dri = (
        dri
        .groupby(
            [
                "resolution",
                "model",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            CD_DRI=("DRI", "mean"),
            datasets_included=(
                "dataset",
                "nunique",
            ),
        )
    )

    cd_dri.to_csv(
        OUTPUT_ROOT
        / "subject_level_CD_DRI.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Across-seed summaries.
    # --------------------------------------------------------

    clean_seed_summary = (
        clean_results
        .groupby(
            [
                "resolution",
                "dataset",
                "model",
            ],
            as_index=False,
        )
        .agg(
            clean_accuracy_mean=(
                "accuracy",
                "mean",
            ),
            clean_accuracy_sd=(
                "accuracy",
                "std",
            ),
            clean_macro_f1_mean=(
                "macro_f1",
                "mean",
            ),
            clean_macro_f1_sd=(
                "macro_f1",
                "std",
            ),
            clean_balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            clean_balanced_accuracy_sd=(
                "balanced_accuracy",
                "std",
            ),
        )
    )

    dri_seed_summary = (
        dri
        .groupby(
            [
                "resolution",
                "dataset",
                "model",
            ],
            as_index=False,
        )
        .agg(
            DRI_mean=(
                "DRI",
                "mean",
            ),
            DRI_sd=(
                "DRI",
                "std",
            ),
        )
    )

    cd_dri_seed_summary = (
        cd_dri
        .groupby(
            [
                "resolution",
                "model",
            ],
            as_index=False,
        )
        .agg(
            CD_DRI_mean=(
                "CD_DRI",
                "mean",
            ),
            CD_DRI_sd=(
                "CD_DRI",
                "std",
            ),
        )
    )

    # Merge clean + DRI for easy resolution comparison.
    resolution_comparison = (
        clean_seed_summary
        .merge(
            dri_seed_summary,
            on=[
                "resolution",
                "dataset",
                "model",
            ],
            how="left",
        )
        .sort_values(
            [
                "dataset",
                "model",
                "resolution",
            ]
        )
    )

    resolution_comparison.to_csv(
        OUTPUT_ROOT
        / "subject_level_resolution_comparison.csv",
        index=False,
    )

    clean_seed_summary.to_csv(
        OUTPUT_ROOT
        / "subject_level_clean_summary_across_seeds.csv",
        index=False,
    )

    dri_seed_summary.to_csv(
        OUTPUT_ROOT
        / "subject_level_DRI_summary_across_seeds.csv",
        index=False,
    )

    cd_dri_seed_summary.to_csv(
        OUTPUT_ROOT
        / "subject_level_CD_DRI_summary_across_seeds.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Condition-level summary across seeds.
    # --------------------------------------------------------

    condition_seed_summary = (
        condition_results
        .groupby(
            [
                "resolution",
                "dataset",
                "model",
                "condition",
                "distortion",
                "severity",
            ],
            as_index=False,
        )
        .agg(
            accuracy_mean=(
                "accuracy",
                "mean",
            ),
            accuracy_sd=(
                "accuracy",
                "std",
            ),
            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),
            macro_f1_sd=(
                "macro_f1",
                "std",
            ),
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_sd=(
                "balanced_accuracy",
                "std",
            ),
        )
    )

    condition_seed_summary.to_csv(
        OUTPUT_ROOT
        / "subject_level_condition_summary_across_seeds.csv",
        index=False,
    )

    arr_seed_summary = (
        arr
        .groupby(
            [
                "resolution",
                "dataset",
                "model",
                "condition",
                "distortion",
                "severity",
            ],
            as_index=False,
        )
        .agg(
            ARR_mean=(
                "ARR",
                "mean",
            ),
            ARR_sd=(
                "ARR",
                "std",
            ),
            delta_accuracy_mean=(
                "delta_accuracy",
                "mean",
            ),
            delta_accuracy_sd=(
                "delta_accuracy",
                "std",
            ),
            delta_macro_f1_mean=(
                "delta_macro_f1",
                "mean",
            ),
            delta_macro_f1_sd=(
                "delta_macro_f1",
                "std",
            ),
        )
    )

    arr_seed_summary.to_csv(
        OUTPUT_ROOT
        / "subject_level_ARR_summary_across_seeds.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Sanity checks.
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)

    for resolution in sorted(
        condition_results["resolution"].unique()
    ):
        for dataset in sorted(
            condition_results["dataset"].unique()
        ):
            subset = condition_results[
                (
                    condition_results["resolution"]
                    == resolution
                )
                &
                (
                    condition_results["dataset"]
                    == dataset
                )
            ]

            if subset.empty:
                continue

            subject_counts = sorted(
                subset["n_subjects"].unique()
            )

            print(
                f"{resolution}x{resolution} | "
                f"{dataset} | "
                f"subject counts seen: "
                f"{subject_counts}"
            )

    # Expected held-out subject counts:
    # OASIS-1 = 36
    # OASIS-2 = 23
    expected_subject_counts = {
        "OASIS_1": 36,
        "OASIS_2": 23,
    }

    for _, row in condition_results.iterrows():
        expected = expected_subject_counts.get(
            row["dataset"]
        )

        if (
            expected is not None
            and int(row["n_subjects"])
            != expected
        ):
            print(
                "WARNING: "
                f"{row['resolution']} | "
                f"{row['dataset']} | "
                f"{row['model']} | "
                f"seed={row['seed']} | "
                f"{row['condition']} has "
                f"{row['n_subjects']} subjects; "
                f"expected {expected}."
            )

    # Check 15 corruption conditions per DRI.
    bad_dri = dri[
        dri["n_conditions"]
        != 15
    ]

    if not bad_dri.empty:
        print(
            "\nWARNING: Some DRI rows do not "
            "contain exactly 15 corrupted conditions:"
        )

        print(
            bad_dri.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Console summary.
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("SUBJECT-LEVEL CLEAN PERFORMANCE")
    print("=" * 80)

    print(
        clean_seed_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print("SUBJECT-LEVEL DRI")
    print("=" * 80)

    print(
        dri_seed_summary
        .round(2)
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print("SUBJECT-LEVEL CD-DRI")
    print("=" * 80)

    print(
        cd_dri_seed_summary
        .round(2)
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Processed resolutions: "
        f"{sorted(found_resolutions)}"
    )

    print(
        f"Results saved to:\n"
        f"{OUTPUT_ROOT}"
    )

    print("\nMain files:")
    main_files = [
        "subject_level_clean_results.csv",
        "subject_level_condition_results.csv",
        "subject_level_ARR.csv",
        "subject_level_DRI.csv",
        "subject_level_CD_DRI.csv",
        "subject_level_resolution_comparison.csv",
        "subject_level_clean_summary_across_seeds.csv",
        "subject_level_DRI_summary_across_seeds.csv",
        "subject_level_CD_DRI_summary_across_seeds.csv",
        "subject_level_condition_summary_across_seeds.csv",
        "subject_level_ARR_summary_across_seeds.csv",
        "subject_level_per_class_metrics.csv",
        "subject_level_all_predictions.csv",
    ]

    for filename in main_files:
        print(
            f"  {filename}"
        )


if __name__ == "__main__":
    main()
