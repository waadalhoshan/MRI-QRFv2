#!/usr/bin/env python3
"""Prepare subject-level prediction rows for the MRI-QRF logistic mixed-effects model."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, nargs="+", required=True,
                    help="CSV(s) with dataset,model,seed,condition,subject_id,true_label,pred_label")
    ap.add_argument("--output", type=Path, required=True)
    a=ap.parse_args()

    frames=[pd.read_csv(p) for p in a.input]
    df=pd.concat(frames,ignore_index=True)
    req={"dataset","model","seed","condition","subject_id","true_label","pred_label"}
    missing=req-set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out=df[list(req)].copy()
    out["correct"]=(out["true_label"].astype(int)==out["pred_label"].astype(int)).astype(int)
    out["patient_key"]=out["dataset"].astype(str)+"::"+out["subject_id"].astype(str)
    out["training_seed"]=out["seed"].astype(str)
    out["condition"]=out["condition"].replace({"clean_test":"clean"})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    df2=df[["dataset","model","seed","condition","subject_id","true_label","pred_label"]].copy()
    df2["correct"]=(df2["true_label"].astype(int)==df2["pred_label"].astype(int)).astype(int)
    df2["patient_key"]=df2["dataset"].astype(str)+"::"+df2["subject_id"].astype(str)
    df2["training_seed"]=df2["seed"].astype(str)
    df2["condition"]=df2["condition"].replace({"clean_test":"clean"})
    df2.to_csv(a.output,index=False)
    print(f"Wrote {len(df2)} rows to {a.output}")

if __name__=="__main__":
    main()
