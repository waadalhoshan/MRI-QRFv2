#!/usr/bin/env python3
"""MRI-QRF canonical preprocessing: volume -> 20 central axial 128x128 PNGs."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image

CLASS_DIRS = ["Non_Demented","Very_Mild_Dementia","Mild_Dementia","Moderate_Dementia"]

def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vol = np.asarray(vol, dtype=np.float32)
    mask = np.isfinite(vol) & (vol != 0)
    out = np.zeros_like(vol, dtype=np.float32)
    if not np.any(mask):
        return out
    p1, p99 = np.percentile(vol[mask], [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        raise ValueError(f"Invalid percentiles p1={p1}, p99={p99}")
    clipped = np.clip(vol, p1, p99)
    out = (clipped - p1) / (p99 - p1)
    out[~mask] = 0.0
    return np.clip(out, 0.0, 1.0)

def choose_slices(nz: int, n: int = 20) -> np.ndarray:
    lo = int(np.floor(0.30 * nz))
    hi = int(np.ceil(0.70 * nz)) - 1
    idx = np.rint(np.linspace(lo, hi, n)).astype(int)
    idx = np.clip(idx, 0, nz-1)
    if len(np.unique(idx)) != n:
        raise RuntimeError(f"Could not choose {n} unique slices from depth {nz}")
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    rows, failures = [], []
    for split in ["train","val","test"]:
        split_dir = args.split_root / split
        for cls in CLASS_DIRS:
            cdir = split_dir / cls
            if not cdir.exists():
                continue
            for subject_dir in sorted([p for p in cdir.iterdir() if p.is_dir()]):
                hdrs = sorted(subject_dir.glob("*.hdr"))
                if len(hdrs) != 1:
                    failures.append((subject_dir.as_posix(), f"expected 1 hdr, found {len(hdrs)}"))
                    continue
                hdr = hdrs[0]
                try:
                    nii = nib.as_closest_canonical(nib.load(str(hdr)))
                    vol = np.squeeze(nii.get_fdata())
                    if vol.ndim != 3:
                        raise ValueError(f"expected 3D after squeeze, got {vol.shape}")
                    vol = normalize_volume(vol)
                    zs = choose_slices(vol.shape[2], 20)
                    out_dir = args.output_root / split / cls / subject_dir.name
                    out_dir.mkdir(parents=True, exist_ok=True)

                    for i, z in enumerate(zs, start=1):
                        sl = vol[:, :, int(z)]
                        im = Image.fromarray(np.rint(sl * 255).astype(np.uint8), mode="L")
                        im = im.resize((128,128), resample=Image.Resampling.BILINEAR)
                        out = out_dir / f"{subject_dir.name}_slice_{i:02d}_z{int(z):03d}.png"
                        im.save(out, format="PNG", optimize=False)
                        rows.append({
                            "dataset": args.dataset,
                            "split": split,
                            "class": cls,
                            "subject_id": subject_dir.name,
                            "slice_number": i,
                            "z_index": int(z),
                            "source_hdr": str(hdr),
                            "output_png": str(out),
                        })
                except Exception as e:
                    failures.append((subject_dir.as_posix(), repr(e)))

    m = pd.DataFrame(rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.manifest, index=False)
    print(pd.crosstab(m["split"], m["class"]))
    print(f"Images: {len(m)}")
    print(f"Unique subjects: {m['subject_id'].nunique()}")
    print(f"Failures: {len(failures)}")
    for f in failures[:20]:
        print(f)

if __name__ == "__main__":
    main()
