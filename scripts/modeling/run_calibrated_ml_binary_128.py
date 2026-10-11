from __future__ import annotations
import re
from pathlib import Path
import joblib, numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

DATASETS={"OASIS_1":Path(r"D:\OASIS_1"),"OASIS_2":Path(r"D:\OASIS_2")}
RESOLUTION=128
SEEDS=[13,47,101]
MODELS=["RandomForest","LinearSVM"]
DISTORTIONS=["motion","rician_noise","rotation","gaussian_blur","brightness_contrast"]
SEVERITIES=["mild","moderate","severe"]
ALIASES={
"nondemented":"Non-Demented","nondementia":"Non-Demented","normal":"Non-Demented","cdr0":"Non-Demented",
"verymilddementia":"Very Mild Dementia","verymild":"Very Mild Dementia","cdr05":"Very Mild Dementia","cdr0p5":"Very Mild Dementia",
"milddementia":"Mild Dementia","mild":"Mild Dementia","cdr1":"Mild Dementia",
"moderatedementia":"Moderate Dementia","moderate":"Moderate Dementia","cdr2":"Moderate Dementia"}

def norm(s): return re.sub(r"[^a-z0-9]","",str(s).lower())
def canonical(s):
    k=norm(s)
    if k not in ALIASES: raise ValueError(f"Unknown class folder: {s}")
    return ALIASES[k]

def load_split(root,class_to_y):
    xs,ys,rows=[],[],[]
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        cname=canonical(cdir.name); y=int(class_to_y(cname))
        for subj in sorted(p for p in cdir.iterdir() if p.is_dir()):
            for f in sorted(subj.rglob("*.png")):
                a=np.asarray(Image.open(f).convert("L"),dtype=np.float32)/255.0
                if a.shape!=(128,128): raise ValueError(f"{f}: {a.shape}")
                xs.append(a.reshape(-1)); ys.append(y)
                rows.append({"subject_id":subj.name,"relative_path":f.relative_to(root).as_posix(),
                             "path":str(f),"y_true":y,"original_class_name":cname})
    if not xs: raise RuntimeError(f"No PNGs under {root}")
    return np.stack(xs).astype(np.float32),np.asarray(ys),pd.DataFrame(rows)

def get_scores(model,X,nclasses,binary):
    if hasattr(model,"predict_proba"): s=np.asarray(model.predict_proba(X))
    else: s=np.asarray(model.decision_function(X))
    if binary and s.ndim==1: s=np.column_stack([-s,s])
    if s.ndim!=2 or s.shape[1]!=nclasses: raise RuntimeError(f"Bad score shape {s.shape}")
    return s

def metric(y,p,nc):
    lab=np.arange(nc)
    return {"accuracy":accuracy_score(y,p),
            "macro_f1":f1_score(y,p,labels=lab,average="macro",zero_division=0),
            "weighted_f1":f1_score(y,p,labels=lab,average="weighted",zero_division=0),
            "balanced_accuracy":balanced_accuracy_score(y,p)}

def evaluate(model,X,y,meta,class_names,ds,mn,seed,cond,d,sev,out,binary):
    p=np.asarray(model.predict(X),dtype=int); s=get_scores(model,X,len(class_names),binary)
    df=meta.copy()
    for k,v in [("dataset",ds),("model",mn),("seed",seed),("condition",cond),("distortion",d),("severity",sev)]: df[k]=v
    df["y_pred"]=p; df["correct"]=(df.y_true.to_numpy()==p).astype(int)
    scorecols=[]
    for j,c in enumerate(class_names):
        sc=f"score_{norm(c)}"; scorecols.append(sc); df[sc]=s[:,j]
    pp=out/"predictions"/ds/mn/f"seed_{seed}"/f"{cond}.csv"; pp.parent.mkdir(parents=True,exist_ok=True); df.to_csv(pp,index=False)

    sr=[]
    for sid,g in df.groupby("subject_id"):
        truth=g.y_true.unique()
        if len(truth)!=1: raise RuntimeError(f"Multiple labels for {sid}")
        ms=g[scorecols].mean().to_numpy()
        r={"dataset":ds,"model":mn,"seed":seed,"condition":cond,"distortion":d,"severity":sev,
           "subject_id":sid,"y_true":int(truth[0]),"y_pred":int(np.argmax(ms))}
        r.update({scorecols[i]:float(ms[i]) for i in range(len(scorecols))}); sr.append(r)
    sdf=pd.DataFrame(sr)
    sp=out/"subject_predictions"/ds/mn/f"seed_{seed}"/f"{cond}.csv"; sp.parent.mkdir(parents=True,exist_ok=True); sdf.to_csv(sp,index=False)
    a={"dataset":ds,"model":mn,"seed":seed,"condition":cond,"distortion":d,"severity":sev,"level":"slice","n":len(y),**metric(y,p,len(class_names))}
    b={"dataset":ds,"model":mn,"seed":seed,"condition":cond,"distortion":d,"severity":sev,"level":"subject","n":len(sdf),**metric(sdf.y_true.to_numpy(),sdf.y_pred.to_numpy(),len(class_names))}
    return a,b

def finish(rows,out):
    z=pd.DataFrame(rows); z.to_csv(out/"metrics_all.csv",index=False)
    for level in ["slice","subject"]:
        q=z[z.level==level].copy()
        clean=q[q.condition=="clean_test"][["dataset","model","seed","accuracy"]].rename(columns={"accuracy":"clean_accuracy"})
        cor=q[q.condition!="clean_test"].merge(clean,on=["dataset","model","seed"],how="left",validate="many_to_one")
        cor["ARR"]=100*cor.accuracy/cor.clean_accuracy; cor["delta_accuracy"]=cor.accuracy-cor.clean_accuracy
        cor.to_csv(out/f"{level}_ARR_all.csv",index=False)
        cor.groupby(["dataset","model","seed"],as_index=False).ARR.mean().rename(columns={"ARR":"DRI"}).to_csv(out/f"{level}_DRI.csv",index=False)
        cor.groupby(["dataset","model","condition","distortion","severity"],as_index=False).agg(
            accuracy_mean=("accuracy","mean"),accuracy_sd=("accuracy","std"),
            ARR_mean=("ARR","mean"),ARR_sd=("ARR","std"),
            delta_accuracy_mean=("delta_accuracy","mean"),delta_accuracy_sd=("delta_accuracy","std")
        ).to_csv(out/f"{level}_condition_summary_across_seeds.csv",index=False)

CLASS_NAMES=['CDR_0', 'CDR_GT_0']
MODEL_ROOT=Path(r"D:\MRI-QRF_binary\results\ML_128\models")
OUT=Path(r"D:\MRI-QRF_calibrated\results\ML_binary_128_calibrated")
def class_to_y(c): return 0 if c=="Non-Demented" else 1

def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=[]
    for ds,root in DATASETS.items():
        X0,y0,m0=load_split(root/"clean_128"/"test",class_to_y)
        cache={}
        for d in DISTORTIONS:
            for sev in SEVERITIES:
                cond=f"{d}__{sev}"
                X,y,m=load_split(root/"distortions_128_calibrated"/d/sev,class_to_y)
                if set(m.relative_path)!=set(m0.relative_path): raise RuntimeError(f"Sample mismatch: {ds} {cond}")
                cache[cond]=(X,y,m,d,sev)
        for mn in MODELS:
            for seed in SEEDS:
                mp=MODEL_ROOT/ds/mn/f"seed_{seed}.joblib"
                if not mp.exists(): raise FileNotFoundError(mp)
                print("Loading",mp); model=joblib.load(mp)
                a,b=evaluate(model,X0,y0,m0,CLASS_NAMES,ds,mn,seed,"clean_test","clean","clean",OUT,True); rows += [a,b]
                for cond,(X,y,m,d,sev) in cache.items():
                    a,b=evaluate(model,X,y,m,CLASS_NAMES,ds,mn,seed,cond,d,sev,OUT,True); rows += [a,b]
                    print(ds,mn,seed,cond,"subject acc=",round(b["accuracy"],4))
    finish(rows,OUT); print("DONE ->",OUT)
if __name__=="__main__": main()