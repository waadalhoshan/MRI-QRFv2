#!/usr/bin/env python3
"""Prepare a blinded MRI-QRF readability-study package.

The original manuscript used 160 unique source slices:
- 10 clean
- 10 from each of 15 corrupted conditions

This script expects clean and distorted test roots plus a candidate manifest.
It enforces source-slice uniqueness, shuffles the final set, renames images
IMG_001...IMG_160, enlarges to 512x512, and writes a private answer key.

For exact manuscript replication, preserve the original private answer key.
"""

from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

DIST=["motion","rician_noise","rotation","gaussian_blur","brightness_contrast"]
SEV=["mild","moderate","severe"]

def source_key(path: str) -> str:
    # filename is preserved across clean/corrupted copies in the MRI-QRF layout
    return Path(path).name

def choose_group(candidates: pd.DataFrame, used: set, n: int, rng) -> pd.DataFrame:
    z=candidates[~candidates["source_key"].isin(used)].copy()
    if len(z)<n:
        raise RuntimeError(f"Only {len(z)} unused candidates remain; need {n}")
    # Prefer approximately equal datasets and classes using greedy rare-cell sampling.
    z["cell"]=z["dataset"].astype(str)+"||"+z["class"].astype(str)
    chosen=[]
    target_dataset={"OASIS_1":n//2,"OASIS_2":n-n//2}
    ds_count={k:0 for k in target_dataset}
    class_count={}
    order=list(z.index); rng.shuffle(order)
    for _ in range(n):
        best=None; best_score=None
        for idx in order:
            if idx in chosen: continue
            r=z.loc[idx]
            ds=r["dataset"]; cls=r["class"]
            ds_pen=ds_count.get(ds,0)/max(target_dataset.get(ds,1),1)
            cls_pen=class_count.get(cls,0)
            score=(ds_pen,cls_pen,rng.random())
            if best_score is None or score<best_score:
                best_score=score; best=idx
        chosen.append(best)
        rr=z.loc[best]
        ds_count[rr["dataset"]]=ds_count.get(rr["dataset"],0)+1
        class_count[rr["class"]]=class_count.get(rr["class"],0)+1
    return z.loc[chosen].copy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidates",type=Path,required=True,
        help="CSV: dataset,class,condition,path; condition=clean or distortion__severity")
    ap.add_argument("--output-root",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=2026)
    a=ap.parse_args()

    d=pd.read_csv(a.candidates)
    req={"dataset","class","condition","path"}
    if not req.issubset(d.columns): raise ValueError(f"Missing {req-set(d.columns)}")
    d["source_key"]=d["path"].map(source_key)
    rng=np.random.default_rng(a.seed)
    used=set(); groups=[]

    conditions=["clean"]+[f"{x}__{s}" for x in DIST for s in SEV]
    for cond in conditions:
        z=d[d["condition"]==cond]
        g=choose_group(z,used,10,rng)
        used.update(g["source_key"])
        g["study_group"]=cond
        groups.append(g)

    sel=pd.concat(groups,ignore_index=True)
    if sel["source_key"].duplicated().any():
        raise RuntimeError("Source-slice duplication detected")
    sel=sel.sample(frac=1,random_state=a.seed).reset_index(drop=True)

    image_dir=a.output_root/"images_for_raters"
    key_dir=a.output_root/"KEY_DO_NOT_SHARE"
    image_dir.mkdir(parents=True,exist_ok=True); key_dir.mkdir(parents=True,exist_ok=True)

    ids=[]
    for i,r in sel.iterrows():
        iid=f"IMG_{i+1:03d}"
        dst=image_dir/f"{iid}.png"
        with Image.open(r["path"]) as im:
            im.convert("L").resize((512,512),Image.Resampling.BILINEAR).save(dst,"PNG")
        ids.append(iid)
    sel.insert(0,"image_id",ids)

    def expected(cond):
        if cond=="clean": return "Readable"
        sev=cond.split("__",1)[1]
        return {"mild":"Readable","moderate":"Borderline","severe":"Unreadable"}[sev]
    sel["expected_readability"]=sel["study_group"].map(expected)
    sel.to_csv(key_dir/"answer_key_DO_NOT_SHARE.csv",index=False)

    summary=(sel.groupby(["study_group","dataset","class"]).size()
             .reset_index(name="n"))
    summary.to_csv(key_dir/"selection_summary.csv",index=False)
    print(f"Prepared {len(sel)} blinded images in {image_dir}")
    print("KEEP KEY_DO_NOT_SHARE PRIVATE.")

if __name__=="__main__":
    main()
