#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

TASK = "multiclass"  # multiclass or binary
SEEDS = [13, 47, 101]
DISTORTIONS = ["motion","rician_noise","rotation","gaussian_blur","brightness_contrast"]
SEVERITIES = ["mild","moderate","severe"]
DATASETS = {"OASIS_1": Path("/workspace/OASIS_1"), "OASIS_2": Path("/workspace/OASIS_2")}
DISTORTION_DIR = "distortions_128_calibrated"
OUT = Path("/workspace/MRI_QRF/results/DL_multiclass_128")
SOURCE_CLASSES = ["Non-Demented","Very Mild Dementia","Mild Dementia","Moderate Dementia"]
ALIASES = {
 "Non-Demented":["Non-Demented","Non_Demented","NonDemented","non_demented"],
 "Very Mild Dementia":["Very Mild Dementia","Very_Mild_Dementia","VeryMildDementia","very_mild_dementia","Very_Mild"],
 "Mild Dementia":["Mild Dementia","Mild_Dementia","MildDementia","mild_dementia"],
 "Moderate Dementia":["Moderate Dementia","Moderate_Dementia","ModerateDementia","moderate_dementia"],
}
EXPECTED_IMAGES = {"OASIS_1":{"train":3300,"val":680,"test":720},"OASIS_2":{"train":2080,"val":460,"test":460}}
EXPECTED_SUBJECTS = {"OASIS_1":{"train":165,"val":34,"test":36},"OASIS_2":{"train":104,"val":23,"test":23}}
BATCH, MAX_EPOCHS, PATIENCE = 32, 20, 5
N_CLASSES = 4 if TASK == "multiclass" else 2

def seed_all(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def class_dir(parent, cname):
    for a in ALIASES[cname]:
        p = parent / a
        if p.is_dir(): return p
    return None

def index_images(root, require_all_source_classes=False):
    if not root.is_dir(): raise FileNotFoundError(f"Missing directory: {root}")
    rows, missing = [], []
    for source_y, cname in enumerate(SOURCE_CLASSES):
        d = class_dir(root, cname)
        if d is None:
            missing.append(cname); continue
        y = source_y if TASK == "multiclass" else (0 if source_y == 0 else 1)
        label_name = cname if TASK == "multiclass" else ("CDR_0" if y == 0 else "CDR_GT_0")
        subdirs = sorted(p for p in d.iterdir() if p.is_dir())
        if subdirs:
            for subj in subdirs:
                for f in sorted(subj.rglob("*.png")):
                    rows.append((str(f), y, label_name, subj.name, cname))
        else:
            for f in sorted(d.glob("*.png")):
                sid = f.stem.split("_slice")[0]
                rows.append((str(f), y, label_name, sid, cname))
    if require_all_source_classes and missing:
        raise FileNotFoundError(f"{root} missing source class folder(s): {', '.join(missing)}")
    if not rows: raise RuntimeError(f"No PNGs found under {root}")
    return rows

class ImgDS(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.tf = transforms.Compose([transforms.Grayscale(1),transforms.Resize((128,128)),transforms.ToTensor()])
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p,y,_,sid,_ = self.rows[i]
        return self.tf(Image.open(p).convert("L")), y, p, sid

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1,bias=False),nn.BatchNorm2d(128),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Dropout(0.30),nn.Linear(128,N_CLASSES))
    def forward(self,x): return self.net(x)

def make_model(name):
    if name == "SimpleCNN": return SimpleCNN()
    if name == "ResNet18":
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(1,64,7,stride=2,padding=3,bias=False)
        m.fc = nn.Linear(m.fc.in_features,N_CLASSES)
        return m
    raise ValueError(name)

def make_opt(name,m):
    if name == "SimpleCNN": return torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    return torch.optim.SGD(m.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-4)

def calc_metrics(y,p):
    labels = list(range(N_CLASSES))
    return dict(
        accuracy=accuracy_score(y,p),
        macro_f1=f1_score(y,p,labels=labels,average="macro",zero_division=0),
        weighted_f1=f1_score(y,p,labels=labels,average="weighted",zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y,p))

def class_weights(rows,device):
    cnt = np.bincount([r[1] for r in rows],minlength=N_CLASSES).astype(float)
    if (cnt == 0).any(): raise RuntimeError(f"Zero-count training class: {cnt}")
    w = np.sqrt(cnt.sum()/cnt); w /= w.mean()
    return torch.tensor(w,dtype=torch.float32,device=device), cnt.astype(int)

def make_loader(rows,shuffle,workers,seed=None):
    gen = None
    if seed is not None:
        gen = torch.Generator(); gen.manual_seed(seed)
    return DataLoader(ImgDS(rows),batch_size=BATCH,shuffle=shuffle,num_workers=workers,
                      pin_memory=True,persistent_workers=workers>0,generator=gen)

@torch.no_grad()
def evaluate(m,dl,device,dataset,model_name,seed,condition,amp_dtype):
    m.eval(); yt=[]; yp=[]; out=[]
    for x,y,paths,sids in dl:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with torch.autocast(device_type="cuda",dtype=amp_dtype,enabled=True): logits=m(x)
        prob=torch.softmax(logits.float(),1); pred=prob.argmax(1)
        ya=y.cpu().numpy(); pa=pred.cpu().numpy(); pr=prob.cpu().numpy()
        yt.extend(ya.tolist()); yp.extend(pa.tolist())
        for i in range(len(ya)):
            r={"dataset":dataset,"model":model_name,"task":TASK,"seed":seed,"condition":condition,
               "image_path":paths[i],"subject_id":sids[i],"true_label":int(ya[i]),"pred_label":int(pa[i])}
            r.update({f"prob_{j}":float(pr[i,j]) for j in range(N_CLASSES)})
            out.append(r)
    return calc_metrics(yt,yp), pd.DataFrame(out)

def subject_metrics(pred):
    probs=[f"prob_{i}" for i in range(N_CLASSES)]
    g=pred.groupby(["dataset","model","task","seed","condition","subject_id","true_label"],as_index=False)[probs].mean()
    g["pred_label"]=g[probs].to_numpy().argmax(1)
    rows=[]
    for cond,z in g.groupby("condition"):
        r={"dataset":z.dataset.iloc[0],"model":z.model.iloc[0],"task":TASK,"seed":int(z.seed.iloc[0]),"condition":cond}
        if cond=="clean": r.update(distortion="clean",severity="clean")
        else:
            d,s=cond.split("__",1); r.update(distortion=d,severity=s)
        r.update(calc_metrics(z.true_label,z.pred_label)); rows.append(r)
    return g,pd.DataFrame(rows)

def validate_layout(names):
    print(f"Validating {TASK} layout...")
    for name in names:
        root=DATASETS[name]; clean={}
        for split in ["train","val","test"]:
            rows=index_images(root/"clean_128"/split,require_all_source_classes=(split=="train")); clean[split]=rows
            nsub=len({r[3] for r in rows})
            print(f"{name} clean {split}: images={len(rows)} subjects={nsub}")
            if len(rows)!=EXPECTED_IMAGES[name][split]: raise RuntimeError(f"{name} {split}: wrong image count")
            if nsub!=EXPECTED_SUBJECTS[name][split]: raise RuntimeError(f"{name} {split}: wrong subject count")
            counts=np.bincount([r[1] for r in rows],minlength=N_CLASSES)
            print("  task slice counts:",counts.tolist())
        test_keys={(r[3],Path(r[0]).name) for r in clean["test"]}
        for d in DISTORTIONS:
            for s in SEVERITIES:
                rr=index_images(root/DISTORTION_DIR/d/s)
                keys={(r[3],Path(r[0]).name) for r in rr}
                if len(rr)!=len(clean["test"]): raise RuntimeError(f"{name} {d}/{s}: wrong image count")
                if keys!=test_keys: raise RuntimeError(f"{name} {d}/{s}: samples differ from clean test")
        print(f"{name}: all 15 calibrated conditions OK")

def train_run(dataset,model_name,seed,device,workers):
    seed_all(seed); root=DATASETS[dataset]
    tr=index_images(root/"clean_128"/"train",True); va=index_images(root/"clean_128"/"val"); te=index_images(root/"clean_128"/"test")
    tl=make_loader(tr,True,workers,seed); vl=make_loader(va,False,workers); cleanl=make_loader(te,False,workers)
    m=make_model(model_name).to(device); opt=make_opt(model_name,m); cw,cnt=class_weights(tr,device); lossfn=nn.CrossEntropyLoss(weight=cw)
    amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler("cuda",enabled=(amp_dtype==torch.float16))
    rd=OUT/dataset/model_name/f"seed_{seed}"; rd.mkdir(parents=True,exist_ok=True)
    with open(rd/"config.json","w") as f:
        json.dump({"dataset":dataset,"model":model_name,"task":TASK,"seed":seed,"batch_size":BATCH,
                   "max_epochs":MAX_EPOCHS,"patience":PATIENCE,"distortion_dir":DISTORTION_DIR,
                   "train_class_counts":cnt.tolist(),"class_weights":cw.detach().cpu().tolist(),
                   "gpu":torch.cuda.get_device_name(0),"amp_dtype":str(amp_dtype)},f,indent=2)
    best=-1.; bad=0; hist=[]; best_epoch=0
    for ep in range(1,MAX_EPOCHS+1):
        m.train(); total=0.; n=0
        for x,y,_,_ in tl:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=amp_dtype): logits=m(x); loss=lossfn(logits,y)
            if scaler.is_enabled(): scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else: loss.backward(); opt.step()
            total+=loss.item()*len(y); n+=len(y)
        vm,_=evaluate(m,vl,device,dataset,model_name,seed,"validation_clean",amp_dtype)
        hist.append({"epoch":ep,"train_loss":total/n,**{f"val_{k}":v for k,v in vm.items()}})
        print(f"[{dataset} {TASK} {model_name} seed={seed}] epoch={ep:02d} loss={total/n:.4f} valMacroF1={vm['macro_f1']:.4f}")
        if vm["macro_f1"]>best+1e-12:
            best=vm["macro_f1"]; best_epoch=ep; bad=0; torch.save(m.state_dict(),rd/"best_model.pt")
        else: bad+=1
        if bad>=PATIENCE: break
    pd.DataFrame(hist).to_csv(rd/"training_history.csv",index=False)
    m.load_state_dict(torch.load(rd/"best_model.pt",map_location=device))
    metric_rows=[]; preds=[]
    cm,p=evaluate(m,cleanl,device,dataset,model_name,seed,"clean",amp_dtype)
    metric_rows.append({"dataset":dataset,"model":model_name,"task":TASK,"seed":seed,"condition":"clean","distortion":"clean","severity":"clean",**cm}); preds.append(p)
    for d in DISTORTIONS:
        for s in SEVERITIES:
            cond=f"{d}__{s}"; rr=index_images(root/DISTORTION_DIR/d/s)
            mm,pp=evaluate(m,make_loader(rr,False,workers),device,dataset,model_name,seed,cond,amp_dtype)
            metric_rows.append({"dataset":dataset,"model":model_name,"task":TASK,"seed":seed,"condition":cond,"distortion":d,"severity":s,**mm}); preds.append(pp)
    md=pd.DataFrame(metric_rows); pdx=pd.concat(preds,ignore_index=True)
    clean_acc=float(md.loc[md.condition=="clean","accuracy"].iloc[0])
    md["ARR"]=np.where(md.condition=="clean",np.nan,100*md.accuracy/clean_acc)
    md["delta_accuracy"]=md.accuracy-clean_acc
    subpred,subm=subject_metrics(pdx); sclean=float(subm.loc[subm.condition=="clean","accuracy"].iloc[0])
    subm["ARR"]=np.where(subm.condition=="clean",np.nan,100*subm.accuracy/sclean)
    subm["delta_accuracy"]=subm.accuracy-sclean
    md.to_csv(rd/"metrics_slice_level.csv",index=False); pdx.to_csv(rd/"predictions_slice_level.csv",index=False)
    subm.to_csv(rd/"metrics_subject_level.csv",index=False); subpred.to_csv(rd/"predictions_subject_level.csv",index=False)
    summary={"dataset":dataset,"model":model_name,"task":TASK,"seed":seed,"best_epoch":best_epoch,"best_val_macro_f1":best,
             "clean_accuracy_slice":clean_acc,"DRI_slice":float(md.loc[md.condition!="clean","ARR"].mean()),
             "clean_accuracy_subject":sclean,"DRI_subject":float(subm.loc[subm.condition!="clean","ARR"].mean())}
    with open(rd/"run_summary.json","w") as f: json.dump(summary,f,indent=2)
    return md,subm,pdx,subpred,summary

def save_global(ms,ss,ps,sps,sums):
    OUT.mkdir(parents=True,exist_ok=True)
    M=pd.concat(ms,ignore_index=True); S=pd.concat(ss,ignore_index=True); P=pd.concat(ps,ignore_index=True); SP=pd.concat(sps,ignore_index=True)
    M.to_csv(OUT/"metrics_all_runs_slice_level.csv",index=False); S.to_csv(OUT/"metrics_all_runs_subject_level.csv",index=False)
    P.to_csv(OUT/"predictions_all_runs_slice_level.csv",index=False); SP.to_csv(OUT/"predictions_all_runs_subject_level.csv",index=False)
    pd.DataFrame(sums).to_csv(OUT/"run_summaries.csv",index=False)
    dri=S[S.condition!="clean"].groupby(["dataset","model","seed"],as_index=False).ARR.mean().rename(columns={"ARR":"DRI"})
    dri.to_csv(OUT/"DRI_by_dataset_model_seed_subject.csv",index=False)
    cd=dri.groupby(["model","seed"],as_index=False).DRI.mean().rename(columns={"DRI":"CD_DRI"}); cd.to_csv(OUT/"CD_DRI_by_model_seed_subject.csv",index=False)
    dri.groupby(["dataset","model"]).DRI.agg(["mean","std"]).reset_index().rename(columns={"mean":"DRI_mean","std":"DRI_sd"}).to_csv(OUT/"DRI_summary_across_seeds_subject.csv",index=False)
    cd.groupby("model").CD_DRI.agg(["mean","std"]).reset_index().rename(columns={"mean":"CD_DRI_mean","std":"CD_DRI_sd"}).to_csv(OUT/"CD_DRI_summary_across_seeds_subject.csv",index=False)
    cols=["accuracy","macro_f1","weighted_f1","balanced_accuracy","ARR","delta_accuracy"]
    q=S.groupby(["dataset","model","condition","distortion","severity"])[cols].agg(["mean","std"])
    q.columns=[f"{a}_{b}" for a,b in q.columns]; q.reset_index().to_csv(OUT/"condition_summary_across_seeds_subject.csv",index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--models",nargs="+",default=["SimpleCNN","ResNet18"]); ap.add_argument("--datasets",nargs="+",default=["OASIS_1","OASIS_2"]); ap.add_argument("--seeds",nargs="+",type=int,default=SEEDS); ap.add_argument("--num-workers",type=int,default=8); ap.add_argument("--validate-only",action="store_true"); a=ap.parse_args()
    validate_layout(a.datasets)
    if a.validate_only: return
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU not available")
    print("GPU:",torch.cuda.get_device_name(0)); device=torch.device("cuda"); OUT.mkdir(parents=True,exist_ok=True)
    ms=[]; ss=[]; ps=[]; sps=[]; sums=[]; total=len(a.datasets)*len(a.models)*len(a.seeds); i=0
    for d in a.datasets:
        for m in a.models:
            for seed in a.seeds:
                i+=1; print("="*80); print(f"RUN {i}/{total}: {d} | {TASK} | {m} | seed {seed}"); t=time.time()
                M,S,P,SP,su=train_run(d,m,seed,device,a.num_workers); su["elapsed_minutes"]=(time.time()-t)/60
                ms.append(M); ss.append(S); ps.append(P); sps.append(SP); sums.append(su); save_global(ms,ss,ps,sps,sums)
                print(f"Done in {su['elapsed_minutes']:.2f} min | subject DRI={su['DRI_subject']:.2f}")
    print(f"All {TASK} runs complete. Results: {OUT}")

if __name__=="__main__": main()