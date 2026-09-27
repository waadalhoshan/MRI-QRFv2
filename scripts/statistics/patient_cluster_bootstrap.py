#!/usr/bin/env python3
"""Patient-clustered bootstrap CIs for clean-vs-corrupted ΔAccuracy and ΔMacro-F1.

Input rows must be subject-level predictions:
dataset, model, seed, condition, subject_id, true_label, pred_label

Within each bootstrap replicate subjects are resampled, paired conditions are
retained, metric deltas are computed per seed, and then averaged across seeds.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

LABELS=[0,1,2,3]

def metric(y,p):
    return (
        accuracy_score(y,p),
        f1_score(y,p,labels=LABELS,average="macro",zero_division=0)
    )

def bootstrap_group(z: pd.DataFrame, n_boot: int, rng: np.random.Generator):
    subjects=np.array(sorted(z["subject_id"].astype(str).unique()))
    conditions=[c for c in z["condition"].unique() if c not in ("clean","clean_test")]
    seeds=sorted(z["seed"].unique())
    rows=[]
    for condition in conditions:
        da=[]; df1=[]
        for _ in range(n_boot):
            sampled=rng.choice(subjects,size=len(subjects),replace=True)
            seed_da=[]; seed_df1=[]
            for seed in seeds:
                zz=z[z["seed"]==seed]
                # Duplicate sampled patients with a bootstrap-instance key.
                pieces=[]
                for j,sid in enumerate(sampled):
                    p=zz[zz["subject_id"].astype(str)==str(sid)].copy()
                    p["_boot_subject"]=f"{j}::{sid}"
                    pieces.append(p)
                b=pd.concat(pieces,ignore_index=True)

                clean=b[b["condition"].isin(["clean","clean_test"])]
                corr=b[b["condition"]==condition]
                if len(clean)==0 or len(corr)==0:
                    continue
                a0,f0=metric(clean["true_label"],clean["pred_label"])
                a1,f1=metric(corr["true_label"],corr["pred_label"])
                seed_da.append(a1-a0); seed_df1.append(f1-f0)
            if seed_da:
                da.append(np.mean(seed_da)); df1.append(np.mean(seed_df1))
        rows.append({
            "condition":condition,
            "delta_accuracy_mean":float(np.mean(da)),
            "delta_accuracy_ci_low":float(np.quantile(da,0.025)),
            "delta_accuracy_ci_high":float(np.quantile(da,0.975)),
            "delta_macro_f1_mean":float(np.mean(df1)),
            "delta_macro_f1_ci_low":float(np.quantile(df1,0.025)),
            "delta_macro_f1_ci_high":float(np.quantile(df1,0.975)),
            "bootstrap_resamples":n_boot,
        })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    a=ap.parse_args()
    d=pd.concat([pd.read_csv(p) for p in a.input],ignore_index=True)
    req={"dataset","model","seed","condition","subject_id","true_label","pred_label"}
    if not req.issubset(d.columns):
        raise ValueError(f"Missing: {sorted(req-set(d.columns))}")
    rng=np.random.default_rng(a.seed)
    outs=[]
    for (ds,mdl),z in d.groupby(["dataset","model"]):
        o=bootstrap_group(z,a.n_boot,rng)
        o.insert(0,"model",mdl); o.insert(0,"dataset",ds)
        outs.append(o)
    out=pd.concat(outs,ignore_index=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output,index=False)
    print(f"Wrote {a.output}")

if __name__=="__main__":
    main()
