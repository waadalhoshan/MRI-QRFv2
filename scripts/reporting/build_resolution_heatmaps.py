#!/usr/bin/env python3
"""Create classical input-resolution ARR heatmaps from subject-level ARR summary."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ORDER=["motion","rician_noise","rotation","gaussian_blur","brightness_contrast"]
LABELS=["Motion","Rician noise","Rotation","Gaussian blur","Brightness/contrast"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args()
    d=pd.read_csv(a.input)
    req={"resolution","dataset","model","distortion","ARR_mean"}
    if not req.issubset(d.columns):
        raise ValueError(f"Missing columns: {sorted(req-set(d.columns))}")

    agg=d.groupby(["resolution","model","distortion"],as_index=False)["ARR_mean"].mean()
    a.output_dir.mkdir(parents=True,exist_ok=True)

    for model in sorted(agg["model"].unique()):
        p=(agg[agg["model"]==model]
           .pivot(index="resolution",columns="distortion",values="ARR_mean")
           .reindex(index=[32,64,128],columns=ORDER))
        fig,ax=plt.subplots(figsize=(8.5,4.2))
        im=ax.imshow(p.to_numpy(),aspect="auto")
        ax.set_xticks(range(len(ORDER)),LABELS,rotation=20,ha="right")
        ax.set_yticks(range(3),["32×32","64×64","128×128"])
        ax.set_xlabel("Distortion type"); ax.set_ylabel("Input resolution")
        ax.set_title(f"{model}: mean subject-level ARR")
        for i in range(3):
            for j in range(len(ORDER)):
                v=p.to_numpy()[i,j]
                if np.isfinite(v): ax.text(j,i,f"{v:.1f}",ha="center",va="center")
        cb=fig.colorbar(im,ax=ax); cb.set_label("Mean ARR (%)")
        fig.tight_layout()
        safe=model.lower().replace(" ","_")
        fig.savefig(a.output_dir/f"{safe}_resolution_arr_heatmap.png",dpi=300,bbox_inches="tight")
        fig.savefig(a.output_dir/f"{safe}_resolution_arr_heatmap.pdf",bbox_inches="tight")
        plt.close(fig)

if __name__=="__main__":
    main()
