#!/usr/bin/env python3
"""Create deterministic subject-level train/val/test partitions for MRI-QRF."""

from __future__ import annotations
import argparse, random, shutil
from pathlib import Path
import pandas as pd

ALLOC = {
    "OASIS_1": {
        "Non_Demented": (95, 20, 20),
        "Very_Mild_Dementia": (49, 10, 11),
        "Mild_Dementia": (20, 4, 4),
        "Moderate_Dementia": (1, 0, 1),
    },
    "OASIS_2": {
        "Non_Demented": (51, 11, 11),
        "Very_Mild_Dementia": (37, 8, 8),
        "Mild_Dementia": (15, 3, 3),
        "Moderate_Dementia": (1, 1, 1),
    },
}
CLASS_ORDER = ["Non_Demented","Very_Mild_Dementia","Mild_Dementia","Moderate_Dementia"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["OASIS_1","OASIS_2"], required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split-root", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if not {"subject_id","class","organized_hdr","organized_img"}.issubset(df.columns):
        raise ValueError("Manifest must contain subject_id,class,organized_hdr,organized_img")

    records = []
    for ci, cls in enumerate(CLASS_ORDER):
        z = df[df["class"] == cls].drop_duplicates("subject_id").copy()
        ids = sorted(z["subject_id"].tolist())
        rng = random.Random(args.seed + 1009 * ci)
        rng.shuffle(ids)
        nt, nv, nte = ALLOC[args.dataset][cls]
        if len(ids) != nt + nv + nte:
            raise RuntimeError(f"{args.dataset} {cls}: found {len(ids)} subjects, expected {nt+nv+nte}")
        assignment = (
            [(sid,"train") for sid in ids[:nt]] +
            [(sid,"val") for sid in ids[nt:nt+nv]] +
            [(sid,"test") for sid in ids[nt+nv:]]
        )
        for sid, split in assignment:
            row = z[z["subject_id"] == sid].iloc[0]
            dest = args.split_root / split / cls / sid
            dest.mkdir(parents=True, exist_ok=True)
            hdr = Path(row["organized_hdr"]); img = Path(row["organized_img"])
            out_hdr = dest / hdr.name; out_img = dest / img.name
            shutil.copy2(hdr, out_hdr); shutil.copy2(img, out_img)
            records.append({
                "dataset": args.dataset,
                "subject_id": sid,
                "class": cls,
                "split": split,
                "source_hdr": str(hdr),
                "split_hdr": str(out_hdr),
            })

    out = pd.DataFrame(records)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_manifest, index=False)
    print(pd.crosstab(out["class"], out["split"]))
    # leakage check
    assert out.groupby("subject_id")["split"].nunique().max() == 1
    print(f"No subject leakage. Seed={args.seed}")

if __name__ == "__main__":
    main()
