#!/usr/bin/env python3
"""Validate MRI-QRF prepared clean/distorted counts and basic hierarchy."""

from __future__ import annotations
import argparse
from pathlib import Path

EXPECTED = {
    "OASIS_1": {"train":3300, "val":680, "test":720, "distorted":10800},
    "OASIS_2": {"train":2080, "val":460, "test":460, "distorted":6900},
}
DISTORTIONS=["motion","rician_noise","rotation","gaussian_blur","brightness_contrast"]
SEVERITIES=["mild","moderate","severe"]

def check(name: str, root: Path):
    print(f"\n{name}: {root}")
    clean = root/"clean_128"
    for split in ["train","val","test"]:
        n = len(list((clean/split).rglob("*.png")))
        exp = EXPECTED[name][split]
        print(f"  clean/{split}: {n} (expected {exp})")
        if n != exp:
            raise RuntimeError(f"{name} {split} count mismatch")
    per = EXPECTED[name]["test"]
    total = 0
    for d in DISTORTIONS:
        for s in SEVERITIES:
            n = len(list((root/"distortions_128"/d/s).rglob("*.png")))
            if n != per:
                raise RuntimeError(f"{name} {d}/{s}: {n}, expected {per}")
            total += n
    print(f"  distorted total: {total} (expected {EXPECTED[name]['distorted']})")
    if total != EXPECTED[name]["distorted"]:
        raise RuntimeError("distorted total mismatch")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--oasis1-root", type=Path, required=True)
    ap.add_argument("--oasis2-root", type=Path, required=True)
    a=ap.parse_args()
    check("OASIS_1", a.oasis1_root)
    check("OASIS_2", a.oasis2_root)
    print("\nValidation passed.")

if __name__=="__main__":
    main()
