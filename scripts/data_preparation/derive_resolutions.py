#!/usr/bin/env python3
"""Derive lower-resolution MRI-QRF copies by bilinear resizing from canonical 128x128 PNGs."""

from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--size", type=int, choices=[32,64], required=True)
    args = ap.parse_args()

    files = sorted(args.source.rglob("*.png"))
    if not files:
        raise RuntimeError(f"No PNGs under {args.source}")
    for i, src in enumerate(files, 1):
        rel = src.relative_to(args.source)
        dst = args.output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im.convert("L").resize((args.size,args.size), Image.Resampling.BILINEAR).save(dst, "PNG")
        if i % 2000 == 0:
            print(f"{i}/{len(files)}")
    print(f"Created {len(files)} images at {args.size}x{args.size}: {args.output}")

if __name__ == "__main__":
    main()
