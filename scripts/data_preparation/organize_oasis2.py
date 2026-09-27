#!/usr/bin/env python3
"""Build the OASIS-2 MRI-QRF cohort using the last available MRI visit per subject."""

from __future__ import annotations
import argparse, shutil
from pathlib import Path
import pandas as pd

CDR_TO_CLASS = {
    0.0: "Non_Demented",
    0.5: "Very_Mild_Dementia",
    1.0: "Mild_Dementia",
    2.0: "Moderate_Dementia",
}

def find_mpr1(raw_root: Path, session_id: str) -> Path | None:
    patterns = [
        f"**/{session_id}_mpr-1_anon.hdr",
        f"**/{session_id}*mpr-1*anon*.hdr",
        f"**/{session_id}*mpr-1*.hdr",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(raw_root.glob(pat))
        if candidates:
            break
    candidates = sorted(set(candidates), key=lambda p: ("/RAW/" not in p.as_posix().upper(), len(p.as_posix())))
    return candidates[0] if candidates else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_excel(args.metadata)
    req = {"Subject ID", "MRI ID", "Visit", "CDR"}
    if not req.issubset(df.columns):
        raise ValueError(f"Metadata must contain {req}; got {list(df.columns)}")

    df["CDR_num"] = pd.to_numeric(df["CDR"], errors="coerce")
    df = df[df["CDR_num"].isin(CDR_TO_CLASS.keys())].copy()
    df["Visit_num"] = pd.to_numeric(df["Visit"], errors="coerce")
    df["MRDelay_num"] = pd.to_numeric(df.get("MR Delay", 0), errors="coerce").fillna(0)
    df = df.sort_values(["Subject ID", "Visit_num", "MRDelay_num"])
    selected = df.groupby("Subject ID", as_index=False).tail(1)

    rows, missing = [], []
    for _, r in selected.iterrows():
        subject_id = str(r["Subject ID"]).strip()
        session_id = str(r["MRI ID"]).strip()
        cdr = float(r["CDR_num"])
        cls = CDR_TO_CLASS[cdr]
        hdr = find_mpr1(args.raw_root, session_id)
        if hdr is None:
            missing.append(session_id)
            continue
        img = hdr.with_suffix(".img")
        if not img.exists():
            missing.append(session_id + " [missing .img]")
            continue

        dest = args.output_root / cls / subject_id
        dest.mkdir(parents=True, exist_ok=True)
        out_hdr = dest / hdr.name
        out_img = dest / img.name
        shutil.copy2(hdr, out_hdr)
        shutil.copy2(img, out_img)

        rows.append({
            "subject_id": subject_id,
            "session_id": session_id,
            "visit": int(r["Visit_num"]) if pd.notna(r["Visit_num"]) else None,
            "cdr": cdr,
            "class": cls,
            "acquisition": "mpr-1",
            "original_hdr": str(hdr),
            "original_img": str(img),
            "organized_hdr": str(out_hdr),
            "organized_img": str(out_img),
        })

    out = pd.DataFrame(rows).sort_values(["class", "subject_id"])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.manifest, index=False)
    print(out["class"].value_counts().sort_index())
    print(f"Total selected: {len(out)}")
    print(f"Missing sessions: {len(missing)}")
    if missing:
        print("\n".join(missing[:20]))

if __name__ == "__main__":
    main()
