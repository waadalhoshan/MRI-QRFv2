#!/usr/bin/env python
"""
MRI-QRF: patient-clustered bootstrap for deep-learning subject-level predictions.

Computes paired changes from CLEAN for SimpleCNN and ResNet18:
  - Delta Accuracy
  - Delta Macro-F1
  - 95% patient-clustered bootstrap confidence intervals

The bootstrap resamples SUBJECTS with replacement. For each sampled subject,
the clean and corrupted predictions are kept paired.

Expected input columns:
    dataset, model, seed, condition, subject_id, true_label, pred_label
Optional probability columns are ignored.

Example from the MRI-QRFv2 repository root:
    python scripts/statistics/run_dl_bootstrap.py

Or specify paths explicitly:
    python scripts/statistics/run_dl_bootstrap.py \
        --input results/deep/DL_128/predictions_all_runs_subject_level.csv \
        --output results/deep/DL_128/patient_clustered_bootstrap_CIs_subject.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "seed",
    "condition",
    "subject_id",
    "true_label",
    "pred_label",
}

DEFAULT_INPUT = Path(
    "results/deep/DL_128/predictions_all_runs_subject_level.csv"
)
DEFAULT_OUTPUT = Path(
    "results/deep/DL_128/patient_clustered_bootstrap_CIs_subject.csv"
)

# Keep the four study classes fixed even when a bootstrap replicate contains
# no observations from one of the rare classes.
CLASS_LABELS = [0, 1, 2, 3]


def split_condition(condition: str) -> tuple[str, str]:
    """Return (distortion, severity) for a non-clean condition."""
    if condition == "clean":
        return "clean", "clean"

    if "__" not in condition:
        raise ValueError(
            f"Unexpected condition format: {condition!r}. "
            "Expected '<distortion>__<severity>'."
        )

    distortion, severity = condition.rsplit("__", 1)
    return distortion, severity


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return Accuracy and four-class Macro-F1."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )
    return float(acc), float(macro_f1)


def prepare_pair(
    clean: pd.DataFrame,
    corrupt: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    seed: int,
    condition: str,
) -> pd.DataFrame:
    """Create one paired clean/corrupted row per subject."""
    key = ["subject_id"]

    clean_cols = key + ["true_label", "pred_label"]
    corrupt_cols = key + ["true_label", "pred_label"]

    c0 = clean[clean_cols].rename(
        columns={
            "true_label": "true_label_clean",
            "pred_label": "pred_clean",
        }
    )
    c1 = corrupt[corrupt_cols].rename(
        columns={
            "true_label": "true_label_corrupt",
            "pred_label": "pred_corrupt",
        }
    )

    if c0["subject_id"].duplicated().any():
        raise ValueError(
            f"Duplicate clean subject rows for {dataset}/{model}/seed={seed}."
        )
    if c1["subject_id"].duplicated().any():
        raise ValueError(
            f"Duplicate corrupted subject rows for "
            f"{dataset}/{model}/seed={seed}/{condition}."
        )

    pair = c0.merge(c1, on="subject_id", how="inner", validate="one_to_one")

    clean_subjects = set(c0["subject_id"])
    corrupt_subjects = set(c1["subject_id"])
    if clean_subjects != corrupt_subjects:
        missing_from_corrupt = sorted(clean_subjects - corrupt_subjects)
        missing_from_clean = sorted(corrupt_subjects - clean_subjects)
        raise ValueError(
            f"Subject mismatch for {dataset}/{model}/seed={seed}/{condition}. "
            f"Missing from corrupt: {missing_from_corrupt[:5]}; "
            f"missing from clean: {missing_from_clean[:5]}"
        )

    if not np.array_equal(
        pair["true_label_clean"].to_numpy(),
        pair["true_label_corrupt"].to_numpy(),
    ):
        raise ValueError(
            f"True-label mismatch between clean and corrupted records for "
            f"{dataset}/{model}/seed={seed}/{condition}."
        )

    return pair


def bootstrap_paired_deltas(
    pair: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    Compute observed paired deltas and subject-clustered bootstrap CIs.

    Delta = corrupted metric - clean metric.
    """
    y_true = pair["true_label_clean"].to_numpy(dtype=int)
    pred_clean = pair["pred_clean"].to_numpy(dtype=int)
    pred_corrupt = pair["pred_corrupt"].to_numpy(dtype=int)

    clean_acc, clean_macro = metric_values(y_true, pred_clean)
    corrupt_acc, corrupt_macro = metric_values(y_true, pred_corrupt)

    observed_delta_acc = corrupt_acc - clean_acc
    observed_delta_macro = corrupt_macro - clean_macro

    n = len(pair)
    boot_delta_acc = np.empty(n_bootstrap, dtype=float)
    boot_delta_macro = np.empty(n_bootstrap, dtype=float)

    # Resample subject rows with replacement. Because each row contains both
    # clean and corrupted predictions, the clean/corrupted pairing is preserved.
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)

        yt = y_true[idx]
        pc = pred_clean[idx]
        pdg = pred_corrupt[idx]

        clean_acc_b, clean_macro_b = metric_values(yt, pc)
        corrupt_acc_b, corrupt_macro_b = metric_values(yt, pdg)

        boot_delta_acc[b] = corrupt_acc_b - clean_acc_b
        boot_delta_macro[b] = corrupt_macro_b - clean_macro_b

    acc_low, acc_high = np.percentile(boot_delta_acc, [2.5, 97.5])
    macro_low, macro_high = np.percentile(boot_delta_macro, [2.5, 97.5])

    return {
        "clean_accuracy": clean_acc,
        "corrupt_accuracy": corrupt_acc,
        "delta_accuracy_observed": observed_delta_acc,
        # These three names mirror the existing classical bootstrap file.
        "delta_accuracy_mean": float(np.mean(boot_delta_acc)),
        "delta_accuracy_ci_low": float(acc_low),
        "delta_accuracy_ci_high": float(acc_high),
        "clean_macro_f1": clean_macro,
        "corrupt_macro_f1": corrupt_macro,
        "delta_macro_f1_observed": observed_delta_macro,
        "delta_macro_f1_mean": float(np.mean(boot_delta_macro)),
        "delta_macro_f1_ci_low": float(macro_low),
        "delta_macro_f1_ci_high": float(macro_high),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Patient-clustered paired bootstrap for MRI-QRF "
            "deep-learning subject-level predictions."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input subject-level predictions CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=2000,
        help="Bootstrap replicates per model/dataset/seed/condition (default: 2000)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
        help="Master RNG seed for reproducible bootstrap sampling (default: 2026)",
    )
    args = parser.parse_args()

    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")

    df = pd.read_csv(args.input)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Input is missing required columns: " + ", ".join(sorted(missing))
        )

    # Normalize expected types.
    df = df.copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["true_label"] = pd.to_numeric(df["true_label"], errors="raise").astype(int)
    df["pred_label"] = pd.to_numeric(df["pred_label"], errors="raise").astype(int)
    df["condition"] = df["condition"].astype(str)
    df["subject_id"] = df["subject_id"].astype(str)

    unknown_labels = set(df["true_label"].unique()) - set(CLASS_LABELS)
    if unknown_labels:
        raise ValueError(f"Unexpected true labels: {sorted(unknown_labels)}")

    if "clean" not in set(df["condition"]):
        raise ValueError("No clean condition was found in the input.")

    # Stable ordering keeps the generated file reproducible.
    groups = (
        df[["dataset", "model", "seed"]]
        .drop_duplicates()
        .sort_values(["dataset", "model", "seed"])
        .itertuples(index=False, name=None)
    )

    master_rng = np.random.default_rng(args.bootstrap_seed)
    rows: list[dict] = []

    for dataset, model, seed in groups:
        subset = df[
            (df["dataset"] == dataset)
            & (df["model"] == model)
            & (df["seed"] == seed)
        ]

        clean = subset[subset["condition"] == "clean"]
        if clean.empty:
            raise ValueError(
                f"Missing clean rows for {dataset}/{model}/seed={seed}."
            )

        conditions = sorted(
            c for c in subset["condition"].unique() if c != "clean"
        )

        for condition in conditions:
            corrupt = subset[subset["condition"] == condition]

            pair = prepare_pair(
                clean,
                corrupt,
                dataset=dataset,
                model=model,
                seed=seed,
                condition=condition,
            )

            # Give each analysis cell its own deterministic child RNG.
            child_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
            rng = np.random.default_rng(child_seed)

            stats = bootstrap_paired_deltas(
                pair,
                n_bootstrap=args.replicates,
                rng=rng,
            )
            distortion, severity = split_condition(condition)

            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "condition": condition,
                    "distortion": distortion,
                    "severity": severity,
                    **stats,
                    "bootstrap_replicates": args.replicates,
                    "bootstrap_subjects": len(pair),
                    "bootstrap_seed_master": args.bootstrap_seed,
                }
            )

    out = pd.DataFrame(rows).sort_values(
        ["dataset", "model", "seed", "distortion", "severity"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Input:  {args.input}")
    print(f"Rows:   {len(df):,}")
    print(f"Output: {args.output}")
    print(f"Tests:  {len(out):,}")
    print(f"Bootstrap replicates per test: {args.replicates:,}")
    print("\nCompleted successfully.")


if __name__ == "__main__":
    main()
