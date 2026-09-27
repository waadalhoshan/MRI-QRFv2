"""
MRI-QRF classical ML pipeline
=============================

Models:
    1. Random Forest
    2. Linear SVM

Default image resolution:
    64 x 64 grayscale (flattened to 4096 features)

Datasets:
    D:\OASIS_1
    D:\OASIS_2

Expected structure (class-folder spelling may vary slightly):
    D:\OASIS_1\clean_64\train\<class>\<subject>\*.png
    D:\OASIS_1\clean_64\val\<class>\<subject>\*.png
    D:\OASIS_1\clean_64\test\<class>\<subject>\*.png

    D:\OASIS_1\distortions_64\<distortion>\<severity>\<class>\<subject>\*.png

and the same for OASIS_2.

Outputs:
    - trained model files
    - validation metrics
    - clean-test metrics
    - all 15 corrupted-test metrics
    - per-class precision / recall / F1 / support
    - confusion matrices
    - raw per-slice predictions
    - ARR for every corrupted condition
    - DRI per dataset/model/seed
    - CD-DRI across OASIS-1 and OASIS-2
    - mean/std summaries across the three training seeds
    - optional patient-clustered bootstrap CIs for clean-vs-corrupted
      changes in Accuracy and Macro-F1

Important:
    Training and validation use CLEAN data only.
    Corruptions are used only for held-out test inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    "OASIS_1": Path(r"D:\OASIS_1"),
    "OASIS_2": Path(r"D:\OASIS_2"),
}

RESOLUTION = 64

SEEDS = [13, 47, 101]

CLASS_NAMES = [
    "Non-Demented",
    "Very Mild Dementia",
    "Mild Dementia",
    "Moderate Dementia",
]

CLASS_TO_INDEX = {
    name: i for i, name in enumerate(CLASS_NAMES)
}

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

# Set False for the first fast run.
# Set True for the final analysis.
RUN_BOOTSTRAP = True
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 2026

OUTPUT_ROOT = Path(r"D:\MRI_QRF_results\ML_64")

# Image extensions accepted by the loader.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ============================================================
# CLASS-NAME NORMALIZATION
# ============================================================

def normalize_token(text: str) -> str:
    """Lowercase and remove spaces/punctuation for robust matching."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


CLASS_ALIASES = {
    # Non-Demented
    "nondemented": "Non-Demented",
    "nondementia": "Non-Demented",
    "normal": "Non-Demented",
    "cdr0": "Non-Demented",

    # Very Mild Dementia
    "verymilddementia": "Very Mild Dementia",
    "verymild": "Very Mild Dementia",
    "cdr05": "Very Mild Dementia",
    "cdr0p5": "Very Mild Dementia",

    # Mild Dementia
    "milddementia": "Mild Dementia",
    "mild": "Mild Dementia",
    "cdr1": "Mild Dementia",

    # Moderate Dementia
    "moderatedementia": "Moderate Dementia",
    "moderate": "Moderate Dementia",
    "cdr2": "Moderate Dementia",
}


def canonical_class_name(folder_name: str) -> str:
    key = normalize_token(folder_name)

    if key not in CLASS_ALIASES:
        raise ValueError(
            f"Unrecognized class folder '{folder_name}'. "
            f"Update CLASS_ALIASES if your folder naming differs."
        )

    return CLASS_ALIASES[key]


# ============================================================
# IMAGE / SAMPLE LOADING
# ============================================================

def list_images(root: Path) -> list[Path]:
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files)


def infer_metadata_from_relative_path(
    image_path: Path,
    root: Path,
) -> tuple[str, str, str]:
    """
    Expected path relative to root:
        <class>/<subject>/<image>.png

    Returns:
        canonical_class, subject_id, sample_key

    sample_key is kept independent of absolute disk location so that
    clean and corrupted copies can be paired.
    """
    rel = image_path.relative_to(root)
    parts = rel.parts

    if len(parts) < 2:
        raise ValueError(
            f"Expected at least class/image below {root}, got: {rel}"
        )

    class_name = canonical_class_name(parts[0])

    if len(parts) >= 3:
        subject_id = parts[1]
    else:
        # Fallback if there is no explicit subject subfolder.
        # OASIS subject IDs are usually embedded in filenames.
        m = re.search(r"(OAS\d+_\d+)", image_path.stem, flags=re.I)
        subject_id = m.group(1) if m else image_path.stem

    # Important: use canonical class name + the path below class.
    # This gives the same key for clean and distorted versions.
    below_class = Path(*parts[1:]).as_posix()
    sample_key = f"{class_name}/{below_class}"

    return class_name, subject_id, sample_key


def load_image_vector(path: Path, resolution: int) -> np.ndarray:
    """Load grayscale image as float32 [0,1] and flatten."""
    with Image.open(path) as img:
        img = img.convert("L")

        # Safety check: if a wrong-sized image sneaks in, resize consistently.
        if img.size != (resolution, resolution):
            img = img.resize(
                (resolution, resolution),
                resample=Image.Resampling.BILINEAR,
            )

        arr = np.asarray(img, dtype=np.float32) / 255.0

    return arr.reshape(-1)


def load_split(root: Path, resolution: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load one split/condition.

    Returns:
        X          shape (n_samples, resolution^2)
        y          integer labels
        metadata   one row per image
    """
    files = list_images(root)

    if not files:
        raise FileNotFoundError(f"No images found under: {root}")

    X = np.empty(
        (len(files), resolution * resolution),
        dtype=np.float32,
    )
    y = np.empty(len(files), dtype=np.int64)

    meta_rows = []

    for i, path in enumerate(files):
        class_name, subject_id, sample_key = infer_metadata_from_relative_path(
            path,
            root,
        )

        X[i] = load_image_vector(path, resolution)
        y[i] = CLASS_TO_INDEX[class_name]

        meta_rows.append({
            "file_path": str(path),
            "relative_path": path.relative_to(root).as_posix(),
            "sample_key": sample_key,
            "subject_id": subject_id,
            "class_name": class_name,
            "y_true": CLASS_TO_INDEX[class_name],
        })

    metadata = pd.DataFrame(meta_rows)

    return X, y, metadata


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(model_name: str, seed: int):
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "LinearSVM":
        return Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svm",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ])

    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    labels = np.arange(len(CLASS_NAMES))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
    }


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    labels = np.arange(len(CLASS_NAMES))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    return pd.DataFrame({
        "class_index": labels,
        "class_name": CLASS_NAMES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_csv: Path,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
    )

    df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    df.index.name = "true_class"
    df.columns.name = "predicted_class"

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(output_csv)


def get_decision_output(model, X: np.ndarray) -> np.ndarray:
    """
    Save decision scores when available.
    Random Forest probability output is also supported.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return np.empty((len(X), 0), dtype=np.float32)


# ============================================================
# PREDICTION / EVALUATION
# ============================================================

def evaluate_condition(
    model,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    seed: int,
    split_name: str,
    condition_name: str,
    distortion: str,
    severity: str,
    output_dir: Path,
) -> dict:
    y_pred = model.predict(X)

    metrics = compute_metrics(
        y,
        y_pred,
    )

    decision = get_decision_output(
        model,
        X,
    )

    pred_df = metadata.copy()

    pred_df.insert(0, "dataset", dataset_name)
    pred_df.insert(1, "model", model_name)
    pred_df.insert(2, "seed", seed)
    pred_df.insert(3, "split", split_name)
    pred_df.insert(4, "condition", condition_name)
    pred_df.insert(5, "distortion", distortion)
    pred_df.insert(6, "severity", severity)

    pred_df["y_pred"] = y_pred
    pred_df["pred_class_name"] = [
        CLASS_NAMES[int(v)] for v in y_pred
    ]
    pred_df["correct"] = (
        pred_df["y_true"].to_numpy() == y_pred
    ).astype(int)

    # Save class scores / probabilities.
    if decision.ndim == 2 and decision.shape[1] == len(CLASS_NAMES):
        for j, class_name in enumerate(CLASS_NAMES):
            safe = normalize_token(class_name)
            pred_df[f"score_{safe}"] = decision[:, j]

    pred_path = (
        output_dir
        / "predictions"
        / dataset_name
        / model_name
        / f"seed_{seed}"
        / f"{condition_name}.csv"
    )

    pred_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_df.to_csv(
        pred_path,
        index=False,
    )

    per_class = compute_per_class_metrics(
        y,
        y_pred,
    )

    per_class.insert(0, "dataset", dataset_name)
    per_class.insert(1, "model", model_name)
    per_class.insert(2, "seed", seed)
    per_class.insert(3, "split", split_name)
    per_class.insert(4, "condition", condition_name)
    per_class.insert(5, "distortion", distortion)
    per_class.insert(6, "severity", severity)

    per_class_path = (
        output_dir
        / "per_class_metrics"
        / dataset_name
        / model_name
        / f"seed_{seed}"
        / f"{condition_name}.csv"
    )

    per_class_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_class.to_csv(
        per_class_path,
        index=False,
    )

    cm_path = (
        output_dir
        / "confusion_matrices"
        / dataset_name
        / model_name
        / f"seed_{seed}"
        / f"{condition_name}.csv"
    )

    save_confusion_matrix(
        y,
        y_pred,
        cm_path,
    )

    row = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "split": split_name,
        "condition": condition_name,
        "distortion": distortion,
        "severity": severity,
        "n_images": len(y),
        "n_subjects": metadata["subject_id"].nunique(),
        **metrics,
    }

    return row


# ============================================================
# BOOTSTRAP
# ============================================================

def stable_seed(*parts) -> int:
    text = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def patient_cluster_bootstrap(
    clean_df: pd.DataFrame,
    corrupt_df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """
    Paired patient-clustered bootstrap.

    Clean and corrupted rows are first paired by sample_key.
    Subjects are then resampled with replacement.
    All slices belonging to a selected subject are included together.
    """

    clean = clean_df[
        ["sample_key", "subject_id", "y_true", "y_pred"]
    ].rename(
        columns={"y_pred": "clean_pred"}
    )

    corrupt = corrupt_df[
        ["sample_key", "subject_id", "y_true", "y_pred"]
    ].rename(
        columns={"y_pred": "corrupt_pred"}
    )

    paired = clean.merge(
        corrupt,
        on=["sample_key", "subject_id", "y_true"],
        how="inner",
        validate="one_to_one",
    )

    if len(paired) != len(clean) or len(paired) != len(corrupt):
        raise ValueError(
            "Clean/corrupted prediction files did not pair one-to-one. "
            "Check sample_key construction and folder structure."
        )

    subjects = paired["subject_id"].drop_duplicates().to_numpy()

    rng = np.random.default_rng(seed)

    delta_acc = np.empty(n_bootstrap, dtype=np.float64)
    delta_f1 = np.empty(n_bootstrap, dtype=np.float64)

    labels = np.arange(len(CLASS_NAMES))

    subject_groups = {
        s: paired.loc[paired["subject_id"] == s]
        for s in subjects
    }

    for b in range(n_bootstrap):
        sampled_subjects = rng.choice(
            subjects,
            size=len(subjects),
            replace=True,
        )

        blocks = [
            subject_groups[s]
            for s in sampled_subjects
        ]

        boot = pd.concat(
            blocks,
            ignore_index=True,
        )

        y_true = boot["y_true"].to_numpy()
        clean_pred = boot["clean_pred"].to_numpy()
        corrupt_pred = boot["corrupt_pred"].to_numpy()

        clean_acc = accuracy_score(y_true, clean_pred)
        corrupt_acc = accuracy_score(y_true, corrupt_pred)

        clean_f1 = f1_score(
            y_true,
            clean_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        corrupt_f1 = f1_score(
            y_true,
            corrupt_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        delta_acc[b] = corrupt_acc - clean_acc
        delta_f1[b] = corrupt_f1 - clean_f1

    return {
        "delta_accuracy_mean": float(delta_acc.mean()),
        "delta_accuracy_ci_low": float(np.quantile(delta_acc, 0.025)),
        "delta_accuracy_ci_high": float(np.quantile(delta_acc, 0.975)),
        "delta_macro_f1_mean": float(delta_f1.mean()),
        "delta_macro_f1_ci_low": float(np.quantile(delta_f1, 0.025)),
        "delta_macro_f1_ci_high": float(np.quantile(delta_f1, 0.975)),
        "bootstrap_replicates": n_bootstrap,
        "bootstrap_subjects": len(subjects),
    }


# ============================================================
# MAIN TRAINING / TEST LOOP
# ============================================================

def main():
    start_time = time.time()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metric_rows = []

    model_names = [
        "RandomForest",
        "LinearSVM",
    ]

    for dataset_name, dataset_root in DATASETS.items():

        clean_root = (
            dataset_root
            / f"clean_{RESOLUTION}"
        )

        distortions_root = (
            dataset_root
            / f"distortions_{RESOLUTION}"
        )

        train_root = clean_root / "train"
        val_root = clean_root / "val"
        test_root = clean_root / "test"

        print("\n")
        print("=" * 80)
        print(f"DATASET: {dataset_name}")
        print("=" * 80)

        print("Loading clean training data...")
        X_train, y_train, meta_train = load_split(
            train_root,
            RESOLUTION,
        )

        print(
            f"  train: {len(y_train)} images, "
            f"{meta_train['subject_id'].nunique()} subjects"
        )

        print("Loading clean validation data...")
        X_val, y_val, meta_val = load_split(
            val_root,
            RESOLUTION,
        )

        print(
            f"  val:   {len(y_val)} images, "
            f"{meta_val['subject_id'].nunique()} subjects"
        )

        print("Loading clean test data...")
        X_test, y_test, meta_test = load_split(
            test_root,
            RESOLUTION,
        )

        print(
            f"  test:  {len(y_test)} images, "
            f"{meta_test['subject_id'].nunique()} subjects"
        )

        # Basic leakage check.
        train_subjects = set(meta_train["subject_id"])
        val_subjects = set(meta_val["subject_id"])
        test_subjects = set(meta_test["subject_id"])

        if train_subjects & val_subjects:
            raise RuntimeError("Subject leakage: train and val overlap.")

        if train_subjects & test_subjects:
            raise RuntimeError("Subject leakage: train and test overlap.")

        if val_subjects & test_subjects:
            raise RuntimeError("Subject leakage: val and test overlap.")

        # Cache corrupted data once per dataset so it is not reloaded
        # separately for every seed/model.
        corruption_cache = {}

        print("Loading corrupted test conditions...")

        for distortion in DISTORTIONS:
            for severity in SEVERITIES:
                condition = f"{distortion}__{severity}"

                condition_root = (
                    distortions_root
                    / distortion
                    / severity
                )

                X_c, y_c, meta_c = load_split(
                    condition_root,
                    RESOLUTION,
                )

                if len(y_c) != len(y_test):
                    raise RuntimeError(
                        f"{dataset_name} {condition}: "
                        f"expected {len(y_test)} images but found {len(y_c)}."
                    )

                # Ensure exact same test slices are represented.
                clean_keys = set(meta_test["sample_key"])
                corrupt_keys = set(meta_c["sample_key"])

                if clean_keys != corrupt_keys:
                    missing = clean_keys - corrupt_keys
                    extra = corrupt_keys - clean_keys

                    raise RuntimeError(
                        f"Sample mismatch in {dataset_name} {condition}. "
                        f"Missing={len(missing)}, Extra={len(extra)}"
                    )

                corruption_cache[condition] = (
                    X_c,
                    y_c,
                    meta_c,
                    distortion,
                    severity,
                )

        print(
            f"  Loaded {len(corruption_cache)} corrupted conditions."
        )

        for model_name in model_names:

            for seed in SEEDS:

                print("\n" + "-" * 80)
                print(
                    f"{dataset_name} | {model_name} | seed={seed}"
                )
                print("-" * 80)

                model = build_model(
                    model_name,
                    seed,
                )

                print("Training...")
                model.fit(
                    X_train,
                    y_train,
                )

                model_path = (
                    OUTPUT_ROOT
                    / "models"
                    / dataset_name
                    / model_name
                    / f"seed_{seed}.joblib"
                )

                model_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                joblib.dump(
                    model,
                    model_path,
                )

                print("Validation...")
                val_row = evaluate_condition(
                    model=model,
                    X=X_val,
                    y=y_val,
                    metadata=meta_val,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed=seed,
                    split_name="val",
                    condition_name="clean_val",
                    distortion="clean",
                    severity="clean",
                    output_dir=OUTPUT_ROOT,
                )

                all_metric_rows.append(val_row)

                print(
                    f"  Val Accuracy={val_row['accuracy']:.4f} "
                    f"Macro-F1={val_row['macro_f1']:.4f}"
                )

                print("Clean test...")
                clean_row = evaluate_condition(
                    model=model,
                    X=X_test,
                    y=y_test,
                    metadata=meta_test,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed=seed,
                    split_name="test",
                    condition_name="clean_test",
                    distortion="clean",
                    severity="clean",
                    output_dir=OUTPUT_ROOT,
                )

                all_metric_rows.append(clean_row)

                print(
                    f"  Test Accuracy={clean_row['accuracy']:.4f} "
                    f"Macro-F1={clean_row['macro_f1']:.4f}"
                )

                print("Corrupted test conditions...")

                for condition, (
                    X_c,
                    y_c,
                    meta_c,
                    distortion,
                    severity,
                ) in corruption_cache.items():

                    row = evaluate_condition(
                        model=model,
                        X=X_c,
                        y=y_c,
                        metadata=meta_c,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        seed=seed,
                        split_name="test",
                        condition_name=condition,
                        distortion=distortion,
                        severity=severity,
                        output_dir=OUTPUT_ROOT,
                    )

                    all_metric_rows.append(row)

                    print(
                        f"  {condition:<32} "
                        f"Acc={row['accuracy']:.4f} "
                        f"Macro-F1={row['macro_f1']:.4f}"
                    )

                # Write an incremental metrics file after every fitted model.
                pd.DataFrame(all_metric_rows).to_csv(
                    OUTPUT_ROOT / "metrics_all_incremental.csv",
                    index=False,
                )

    # ========================================================
    # SUMMARY METRICS / ARR / DRI / CD-DRI
    # ========================================================

    metrics_df = pd.DataFrame(all_metric_rows)

    metrics_df.to_csv(
        OUTPUT_ROOT / "metrics_all.csv",
        index=False,
    )

    test_df = metrics_df[
        metrics_df["split"] == "test"
    ].copy()

    clean_test = test_df[
        test_df["condition"] == "clean_test"
    ][
        [
            "dataset",
            "model",
            "seed",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
        ]
    ].rename(
        columns={
            "accuracy": "clean_accuracy",
            "macro_f1": "clean_macro_f1",
            "weighted_f1": "clean_weighted_f1",
            "balanced_accuracy": "clean_balanced_accuracy",
        }
    )

    corrupted = test_df[
        test_df["condition"] != "clean_test"
    ].copy()

    corrupted = corrupted.merge(
        clean_test,
        on=["dataset", "model", "seed"],
        how="left",
        validate="many_to_one",
    )

    corrupted["ARR"] = np.where(
        corrupted["clean_accuracy"] > 0,
        (
            corrupted["accuracy"]
            / corrupted["clean_accuracy"]
        ) * 100.0,
        np.nan,
    )

    corrupted["delta_accuracy"] = (
        corrupted["accuracy"]
        - corrupted["clean_accuracy"]
    )

    corrupted["delta_macro_f1"] = (
        corrupted["macro_f1"]
        - corrupted["clean_macro_f1"]
    )

    corrupted.to_csv(
        OUTPUT_ROOT / "corrupted_metrics_with_ARR.csv",
        index=False,
    )

    # DRI = mean ARR across 15 corrupted conditions.
    dri = (
        corrupted
        .groupby(
            ["dataset", "model", "seed"],
            as_index=False,
        )
        .agg(
            DRI=("ARR", "mean"),
            mean_corrupted_accuracy=("accuracy", "mean"),
            mean_corrupted_macro_f1=("macro_f1", "mean"),
        )
    )

    dri.to_csv(
        OUTPUT_ROOT / "DRI_by_dataset_model_seed.csv",
        index=False,
    )

    # Cross-dataset DRI = mean of the two dataset-specific DRIs.
    cd_dri = (
        dri
        .groupby(
            ["model", "seed"],
            as_index=False,
        )
        .agg(
            CD_DRI=("DRI", "mean"),
            datasets_included=("dataset", "nunique"),
        )
    )

    cd_dri.to_csv(
        OUTPUT_ROOT / "CD_DRI_by_model_seed.csv",
        index=False,
    )

    # Mean ± SD across seeds for clean and corrupted condition results.
    condition_seed_summary = (
        test_df
        .groupby(
            [
                "dataset",
                "model",
                "condition",
                "distortion",
                "severity",
            ],
            as_index=False,
        )
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_sd=("weighted_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
        )
    )

    condition_seed_summary.to_csv(
        OUTPUT_ROOT / "condition_summary_across_seeds.csv",
        index=False,
    )

    arr_seed_summary = (
        corrupted
        .groupby(
            [
                "dataset",
                "model",
                "condition",
                "distortion",
                "severity",
            ],
            as_index=False,
        )
        .agg(
            ARR_mean=("ARR", "mean"),
            ARR_sd=("ARR", "std"),
            delta_accuracy_mean=("delta_accuracy", "mean"),
            delta_accuracy_sd=("delta_accuracy", "std"),
            delta_macro_f1_mean=("delta_macro_f1", "mean"),
            delta_macro_f1_sd=("delta_macro_f1", "std"),
        )
    )

    arr_seed_summary.to_csv(
        OUTPUT_ROOT / "ARR_summary_across_seeds.csv",
        index=False,
    )

    dri_seed_summary = (
        dri
        .groupby(
            ["dataset", "model"],
            as_index=False,
        )
        .agg(
            DRI_mean=("DRI", "mean"),
            DRI_sd=("DRI", "std"),
        )
    )

    dri_seed_summary.to_csv(
        OUTPUT_ROOT / "DRI_summary_across_seeds.csv",
        index=False,
    )

    cd_dri_seed_summary = (
        cd_dri
        .groupby(
            ["model"],
            as_index=False,
        )
        .agg(
            CD_DRI_mean=("CD_DRI", "mean"),
            CD_DRI_sd=("CD_DRI", "std"),
        )
    )

    cd_dri_seed_summary.to_csv(
        OUTPUT_ROOT / "CD_DRI_summary_across_seeds.csv",
        index=False,
    )

    # ========================================================
    # OPTIONAL PATIENT-CLUSTERED BOOTSTRAP
    # ========================================================

    if RUN_BOOTSTRAP:
        print("\n" + "=" * 80)
        print("PATIENT-CLUSTERED BOOTSTRAP")
        print("=" * 80)

        bootstrap_rows = []

        for dataset_name in DATASETS:
            for model_name in model_names:
                for seed in SEEDS:

                    clean_path = (
                        OUTPUT_ROOT
                        / "predictions"
                        / dataset_name
                        / model_name
                        / f"seed_{seed}"
                        / "clean_test.csv"
                    )

                    clean_pred = pd.read_csv(clean_path)

                    for distortion in DISTORTIONS:
                        for severity in SEVERITIES:

                            condition = f"{distortion}__{severity}"

                            corrupt_path = (
                                OUTPUT_ROOT
                                / "predictions"
                                / dataset_name
                                / model_name
                                / f"seed_{seed}"
                                / f"{condition}.csv"
                            )

                            corrupt_pred = pd.read_csv(
                                corrupt_path
                            )

                            boot_seed = stable_seed(
                                BOOTSTRAP_SEED,
                                dataset_name,
                                model_name,
                                seed,
                                condition,
                            )

                            boot = patient_cluster_bootstrap(
                                clean_pred,
                                corrupt_pred,
                                n_bootstrap=N_BOOTSTRAP,
                                seed=boot_seed,
                            )

                            bootstrap_rows.append({
                                "dataset": dataset_name,
                                "model": model_name,
                                "seed": seed,
                                "condition": condition,
                                "distortion": distortion,
                                "severity": severity,
                                **boot,
                            })

                            print(
                                f"{dataset_name} | "
                                f"{model_name} | "
                                f"seed={seed} | "
                                f"{condition}"
                            )

        bootstrap_df = pd.DataFrame(
            bootstrap_rows
        )

        bootstrap_df.to_csv(
            OUTPUT_ROOT
            / "patient_clustered_bootstrap_CIs.csv",
            index=False,
        )

    # ========================================================
    # RUN METADATA
    # ========================================================

    run_info = {
        "resolution": RESOLUTION,
        "seeds": SEEDS,
        "class_names": CLASS_NAMES,
        "models": model_names,
        "distortions": DISTORTIONS,
        "severities": SEVERITIES,
        "run_bootstrap": RUN_BOOTSTRAP,
        "bootstrap_replicates": (
            N_BOOTSTRAP if RUN_BOOTSTRAP else 0
        ),
        "random_forest": {
            "n_estimators": 300,
            "max_depth": 10,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
        "linear_svm": {
            "scaler": "StandardScaler",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
        },
        "training_protocol": (
            "Train and validation use clean images only. "
            "All distortions are test-time only."
        ),
    }

    with open(
        OUTPUT_ROOT / "run_configuration.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run_info,
            f,
            indent=2,
        )

    elapsed = time.time() - start_time

    print("\n")
    print("=" * 80)
    print("FINISHED")
    print("=" * 80)
    print(f"Results folder: {OUTPUT_ROOT}")
    print(f"Elapsed: {elapsed / 60:.1f} minutes")
    print("\nMain result files:")
    print("  metrics_all.csv")
    print("  corrupted_metrics_with_ARR.csv")
    print("  DRI_by_dataset_model_seed.csv")
    print("  CD_DRI_by_model_seed.csv")
    print("  condition_summary_across_seeds.csv")
    print("  ARR_summary_across_seeds.csv")
    print("  DRI_summary_across_seeds.csv")
    print("  CD_DRI_summary_across_seeds.csv")

    if RUN_BOOTSTRAP:
        print("  patient_clustered_bootstrap_CIs.csv")


if __name__ == "__main__":
    main()
