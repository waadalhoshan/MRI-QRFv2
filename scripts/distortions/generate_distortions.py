#!/usr/bin/env python3
"""Generate MRI-QRF test-only distortions at canonical 128x128 resolution."""

from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import rotate, gaussian_filter

PARAMS = {
    "motion": {
        "mild": (0.10, 1),
        "moderate": (0.20, 3),
        "severe": (0.40, 6),
    },
    "rician_noise": {"mild": 0.035, "moderate": 0.075, "severe": 0.15},
    "rotation": {"mild": 5.0, "moderate": 10.0, "severe": 15.0},
    "gaussian_blur": {"mild": 0.75, "moderate": 1.5, "severe": 2.5},
    "brightness_contrast": {
        "mild": (0.05, 0.10),
        "moderate": (0.10, 0.20),
        "severe": (0.15, 0.30),
    },
}

def deterministic_seed(global_seed: int, dataset: str, rel: str, distortion: str, severity: str) -> int:
    payload = f"{global_seed}|{dataset}|{rel}|{distortion}|{severity}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)

def motion_kspace(x: np.ndarray, fraction: float, max_disp: int, rng: np.random.Generator) -> np.ndarray:
    h, w = x.shape
    k = np.fft.fftshift(np.fft.fft2(x))
    n_lines = max(1, int(round(fraction * h)))
    lines = rng.choice(h, size=n_lines, replace=False)

    fx = np.fft.fftshift(np.fft.fftfreq(w))
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    for row in lines:
        dx = int(rng.integers(-max_disp, max_disp + 1))
        dy = int(rng.integers(-max_disp, max_disp + 1))
        # Translation theorem. For a selected ky line, ky is constant.
        phase = np.exp(-2j * np.pi * (fx * dx + fy[row] * dy))
        k[row, :] *= phase

    y = np.abs(np.fft.ifft2(np.fft.ifftshift(k)))
    return np.clip(y, 0.0, 1.0)

def apply(x: np.ndarray, distortion: str, severity: str, rng: np.random.Generator) -> np.ndarray:
    if distortion == "motion":
        frac, disp = PARAMS[distortion][severity]
        return motion_kspace(x, frac, disp, rng)
    if distortion == "rician_noise":
        sigma = PARAMS[distortion][severity]
        n1 = rng.normal(0, sigma, x.shape)
        n2 = rng.normal(0, sigma, x.shape)
        return np.clip(np.sqrt((x+n1)**2 + n2**2), 0, 1)
    if distortion == "rotation":
        mag = PARAMS[distortion][severity]
        angle = float(rng.choice([-1.0, 1.0])) * mag
        return np.clip(rotate(x, angle, reshape=False, order=1, mode="constant", cval=0.0, prefilter=False), 0, 1)
    if distortion == "gaussian_blur":
        sigma = PARAMS[distortion][severity]
        return np.clip(gaussian_filter(x, sigma=sigma, mode="nearest"), 0, 1)
    if distortion == "brightness_contrast":
        bmag, cmag = PARAMS[distortion][severity]
        b = float(rng.choice([-1.0, 1.0])) * bmag
        c = float(rng.choice([-1.0, 1.0])) * cmag
        y = (x - 0.5) * (1.0 + c) + 0.5 + b
        return np.clip(y, 0, 1)
    raise ValueError(distortion)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--clean-test-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--global-seed", type=int, default=42)
    ap.add_argument("--manifest", type=Path)
    args=ap.parse_args()

    images=sorted(args.clean_test_root.rglob("*.png"))
    if not images:
        raise RuntimeError(f"No PNGs under {args.clean_test_root}")

    rows=[]
    for src in images:
        rel=src.relative_to(args.clean_test_root)
        x=np.asarray(Image.open(src).convert("L"), dtype=np.float32)/255.0
        for distortion in PARAMS:
            for severity in ["mild","moderate","severe"]:
                seed=deterministic_seed(args.global_seed,args.dataset,rel.as_posix(),distortion,severity)
                rng=np.random.default_rng(seed)
                y=apply(x,distortion,severity,rng)
                dst=args.output_root/distortion/severity/rel
                dst.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.rint(y*255).astype(np.uint8),mode="L").save(dst,"PNG")
                rows.append({
                    "dataset":args.dataset,
                    "relative_source":rel.as_posix(),
                    "distortion":distortion,
                    "severity":severity,
                    "seed":seed,
                    "output":str(dst)
                })

    print(f"Source test PNGs: {len(images)}")
    print(f"Generated: {len(rows)}")
    if args.manifest:
        args.manifest.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(args.manifest,index=False)

if __name__=="__main__":
    main()
