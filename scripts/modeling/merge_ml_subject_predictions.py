from pathlib import Path
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

BASE = Path(r"D:\MRI-QRF_calibrated\results")

TASKS = {
    "binary": {
        "root": BASE / "ML_binary_128_calibrated",
        "output": "ML_binary_predictions_all_runs_subject_level.csv",
    },
    "multiclass": {
        "root": BASE / "ML_multiclass_128_calibrated",
        "output": "ML_multiclass_predictions_all_runs_subject_level.csv",
    },
}


# ============================================================
# HELPERS
# ============================================================

def normalize_condition_from_filename(stem: str) -> str:
    """
    Convert filenames such as:
        clean_test
        motion_mild
        gaussian_blur_moderate
        brightness_contrast_severe

    into the condition naming convention:
        clean_test
        motion__mild
        gaussian_blur__moderate
        brightness_contrast__severe
    """
    if stem == "clean_test":
        return "clean_test"

    severities = ["mild", "moderate", "severe"]

    for severity in severities:
        suffix = f"_{severity}"
        if stem.endswith(suffix):
            distortion = stem[: -len(suffix)]
            return f"{distortion}__{severity}"

    return stem


def infer_distortion_and_severity(condition: str):
    if condition == "clean_test":
        return "clean", "clean"

    if "__" in condition:
        distortion, severity = condition.split("__", 1)
        return distortion, severity

    return "", ""


def combine_task(task_name: str, result_root: Path, output_name: str):
    pred_root = result_root / "subject_predictions"

    if not pred_root.exists():
        raise FileNotFoundError(
            f"Missing subject_predictions folder:\n{pred_root}"
        )

    rows = []

    for dataset_dir in sorted(pred_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            model = model_dir.name

            for seed_dir in sorted(model_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue

                if not seed_dir.name.startswith("seed_"):
                    continue

                seed = int(seed_dir.name.replace("seed_", ""))

                for csv_file in sorted(seed_dir.glob("*.csv")):
                    df = pd.read_csv(csv_file)

                    # ----------------------------------------
                    # Metadata from folder structure
                    # ----------------------------------------
                    df["dataset"] = dataset
                    df["model"] = model
                    df["seed"] = seed
                    df["task"] = task_name

                    # ----------------------------------------
                    # Condition
                    # ----------------------------------------
                    filename_condition = normalize_condition_from_filename(
                        csv_file.stem
                    )

                    if "condition" not in df.columns:
                        df["condition"] = filename_condition

                    # Normalize condition naming consistently
                    df["condition"] = df["condition"].replace({
                        csv_file.stem: filename_condition
                    })

                    # ----------------------------------------
                    # Distortion / severity
                    # ----------------------------------------
                    if (
                        "distortion" not in df.columns
                        or "severity" not in df.columns
                    ):
                        ds = df["condition"].apply(
                            infer_distortion_and_severity
                        )

                        if "distortion" not in df.columns:
                            df["distortion"] = ds.apply(lambda x: x[0])

                        if "severity" not in df.columns:
                            df["severity"] = ds.apply(lambda x: x[1])

                    rows.append(df)

    if not rows:
        raise RuntimeError(
            f"No CSV prediction files found under:\n{pred_root}"
        )

    combined = pd.concat(rows, ignore_index=True)

    # ========================================================
    # COLUMN ORDER
    # ========================================================

    common_cols = [
        "dataset",
        "model",
        "task",
        "seed",
        "condition",
        "distortion",
        "severity",
        "subject_id",
        "y_true",
        "y_pred",
    ]

    # Keep score/probability columns automatically
    score_cols = [
        c for c in combined.columns
        if c.startswith("score_") or c.startswith("prob_")
    ]

    # Any extra useful columns are retained at the end
    used = set(common_cols + score_cols)

    extra_cols = [
        c for c in combined.columns
        if c not in used
    ]

    final_cols = (
        [c for c in common_cols if c in combined.columns]
        + sorted(score_cols)
        + extra_cols
    )

    combined = combined[final_cols]

    # ========================================================
    # SORT
    # ========================================================

    sort_cols = [
        c for c in [
            "dataset",
            "model",
            "seed",
            "condition",
            "subject_id",
        ]
        if c in combined.columns
    ]

    combined = (
        combined
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_path = result_root / output_name

    combined.to_csv(output_path, index=False)

    # ========================================================
    # VALIDATION SUMMARY
    # ========================================================

    print("=" * 80)
    print(f"TASK: {task_name}")
    print(f"Saved: {output_path}")
    print(f"Rows: {len(combined):,}")

    print("\nDatasets:")
    print(sorted(combined["dataset"].unique()))

    print("\nModels:")
    print(sorted(combined["model"].unique()))

    print("\nSeeds:")
    print(sorted(combined["seed"].unique()))

    print("\nConditions:")
    print(
        combined["condition"]
        .value_counts()
        .sort_index()
    )

    print("\nUnique runs:")
    print(
        combined[
            ["dataset", "model", "seed"]
        ]
        .drop_duplicates()
        .sort_values(
            ["dataset", "model", "seed"]
        )
        .to_string(index=False)
    )

    print()


# ============================================================
# RUN BOTH TASKS
# ============================================================

for task_name, cfg in TASKS.items():
    combine_task(
        task_name=task_name,
        result_root=cfg["root"],
        output_name=cfg["output"],
    )

print("DONE.")