"""
FINAL Pipeline - Simple, Fast, Robust
No albumentations - pure torchvision transforms
"""
import os,json,pickle,warnings,copy,random,re
from pathlib import Path; from collections import Counter
import numpy as np; import pandas as pd
import torch,torch.nn as nn,torch.nn.functional as F
from torch.cuda.amp import GradScaler,autocast
from torch.utils.data import Dataset,DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix)
from tqdm import tqdm; import cv2,timm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings('ignore')
ROOT = Path(os.environ.get("IQOTH_ROOT", Path(__file__).resolve().parents[1]))
DD=ROOT/"data"/"raw"/"IQ-OTH-NCCD"/"Augmented IQ-OTHNCCD lung cancer dataset"
RD=ROOT/"results"; RD.mkdir(parents=True,exist_ok=True)
for d in ["figures","tables","weights"]: (RD/d).mkdir(parents=True,exist_ok=True)

D=torch.device("cuda"); SEED=42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark=True
IMG,B,N_EP,N_F=224,16,50,5; CN=["Normal","Benign","Malignant"]
print(f"D: cuda | GPU: {torch.cuda.get_device_name(0)} | Folds:{N_F}")

#--- DATA ---
CD={"Normal":"Normal cases","Benign":"Benign cases","Malignant":"Malignant cases"}
rec=[]
for cn,cdn in CD.items():
    cd=DD/cdn
    if not cd.exists(): continue
    for f in sorted(cd.iterdir()):
        if not f.suffix.lower() in ['.jpg','.jpeg','.png']: continue
        m=re.match(r'(Bengin|Malignant|Normal)\s+case\s*\((\d+)\)',f.stem,re.IGNORECASE)
        if m: rec.append({'fp':str(f),'cn':cn,'cl':CN.index(cn),'sk':int(m.group(2))})
df=pd.DataFrame(rec).sort_values(['cn','sk']).reset_index(drop=True)

SP={"Normal":8,"Benign":8,"Malignant":14}; pgs=[]
for cn in CN:
    cd=df[df.cn==cn].copy(); ns=len(cd); npats=max(1,ns//SP[cn])
    cd['pidx']=[i*npats//ns for i in range(ns)]
    cd['pid']=cd.apply(lambda r:f"{r.cn}_P{r.pidx+1:03d}",axis=1); pgs.append(cd)
df=pd.concat(pgs,ignore_index=True).sort_values(['pid','sk']).reset_index(drop=True)
df['si']=df.groupby('pid').cumcount()
ps=df.groupby('pid').agg(ns=('fp','count'),cn=('cn','first'),cl=('cl','first')).reset_index()
print(f"Data: {len(df)} imgs, {len(ps)} patients")
for cn in CN:
    cp=ps[ps.cn==cn]; print(f"  {cn:10s}: {len(cp):3d}p {cp.ns.sum():4d}s")

# Splits
pl=ps.set_index('pid')['cl']; pids=ps.pid.tolist(); labs=pl.values.astype(int)
gkf=StratifiedGroupKFold(n_splits=N_F,shuffle=True,random_state=SEED)
si={"nf":N_F,"f":[]}
for fi,(tvi,tei) in enumerate(gkf.split(np.zeros(len(pids)),labs,groups=pids)):
    tep=[pids[i] for i in tei]; tvp=[pids[i] for i in tvi]; tvl=labs[tvi]
    np.random.seed(SEED+fi); vsz=max(1,int(len(tvp)*0.2))
    vi=np.random.choice(len(tvp),vsz,replace=False)
    trp=[tvp[i] for i in range(len(tvp)) if i not in vi]; vap=[tvp[i] for i in vi]
    si["f"].append({"fi":fi,"trp":trp,"vap":vap,"tep":tep})

# Transforms
trT=T.Compose([T.Resize((IMG,IMG)),T.RandomHorizontalFlip(0.5),T.RandomVerticalFlip(0.3),
    T.RandomRotation(15),T.ColorJitter(brightness=0.2,contrast=0.2),
    T.RandomAffine(degrees=0,translate=(0.1,0.1),scale=(0.85,1.15)),
    T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
vaT=T.Compose([T.Resize((IMG,IMG)),T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

# Dataset
class DS(Dataset):
    def __init__(self,df,pids,tr=False):
        self.df=df[df.pid.isin(pids)].reset_index(drop=True); self.tr=tr; self.c={}
        for i in range(len(self.df)):
            img=cv2.imread(self.df.iloc[i].fp,cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img,(IMG,IMG)).astype(np.float32)/255.0
            self.c[i]=torch.from_numpy(np.stack([img,img,img])).float()
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        t=self.c[i].clone(); l=self.df.iloc[i].cl
        if self.tr: t=trT(t)
        else: t=vaT(t)
        return t,torch.tensor(l,dtype=torch.long)

def loaders(fi):
    d={}
    for sp in ['trp','vap','tep']:
        ds=DS(df,fi[sp],tr=(sp=='trp'))
        d[sp]=DataLoader(ds,batch_size=B,shuffle=(sp=='trp'),num_workers=0,pin_memory=True)
    return d

# FocalLoss
class FL(nn.Module):
    def __init__(self,g=2.0): super().__init__(); self.a=torch.tensor([0.8,1.5,1.2]); self.g=g
    def forward(self,i,t):
        ce=F.cross_entropy(i,t,reduction='none'); pt=torch.exp(-ce)
        return (self.a.to(i.device)[t]*(1-pt)**self.g*ce).mean()

def gm(n):
    if n=='r50': return timm.create_model('resnet50',pretrained=True,num_classes=3)
    if n=='eb0': return timm.create_model('efficientnet_b0',pretrained=True,num_classes=3)
    if n=='cnx': return timm.create_model('convnext_tiny',pretrained=True,num_classes=3)

def te(m,ld,cri,opt,sc):
    m.train(); tl,cor,tot=0,0,0
    for im,la in ld:
        im,la=im.to(D),la.to(D); opt.zero_grad(set_to_none=True)
        with autocast(): out=m(im); loss=cri(out,la)
        sc.scale(loss).backward(); sc.step(opt); sc.update()
        tl+=loss.item()*la.size(0); cor+=(out.argmax(1)==la).sum().item(); tot+=la.size(0)
    return tl/tot,cor/tot

@torch.no_grad()
def ev(m,ld,cri):
    m.eval(); tl,cor,tot=0,0,0; ap,al,apr=[],[],[]
    for im,la in ld:
        im,la=im.to(D),la.to(D); out=m(im); loss=cri(out,la)
        tl+=loss.item()*la.size(0); cor+=(out.argmax(1)==la).sum().item(); tot+=la.size(0)
        ap.extend(out.argmax(1).cpu().tolist()); al.extend(la.cpu().tolist()); apr.extend(F.softmax(out,1).cpu().tolist())
    pa,la=np.array(ap),np.array(al); pr=np.array(apr)
    return {'loss':tl/tot,'acc':accuracy_score(la,pa),'f1':f1_score(la,pa,average='macro',zero_division=0),
            'prec':precision_score(la,pa,average='macro',zero_division=0),
            'rec':recall_score(la,pa,average='macro',zero_division=0),
            'auc':roc_auc_score(la,pr,multi_class='ovr',average='macro') if len(pr.shape)>1 else 0,
            'cm':confusion_matrix(la,pa).tolist()}

def tf(mn,fi,fd):
    m=gm(mn).to(D); ld=loaders(fd); cri=FL()
    opt=AdamW(m.parameters(),lr=3e-4 if 'cnx' in mn else 1e-4,weight_decay=0.05 if 'cnx' in mn else 1e-5)
    sch=CosineAnnealingLR(opt,T_max=N_EP*len(ld['trp'])) if 'cnx' in mn else None; sc=GradScaler()
    bf1,bst,pc=0,None,0
    print(f"\n  Fold {fi+1}/{N_F} - {mn}", flush=True)
    for ep in range(N_EP):
        tl,ta=te(m,ld['trp'],cri,opt,sc)
        if sch: sch.step()
        vm=ev(m,ld['vap'],cri)
        if vm['f1']>bf1: bf1=vm['f1']; bst=copy.deepcopy(m.state_dict()); pc=0
        else: pc+=1
        if ep%5==0 or ep==N_EP-1:
            print(f"    E{ep:3d}: TrA={ta:.4f} VA={vm['acc']:.4f} VF1={vm['f1']:.4f} AUC={vm['auc']:.4f}", flush=True)
        if pc>=15: print(f"    ES@{ep}", flush=True); break
    m.load_state_dict(bst); tm=ev(m,ld['tep'],cri)
    tps=fd['tep']; pcor,ptot=0,0
    for pid in tps:
        pdf=df[df.pid==pid]; tl=pdf.iloc[0].cl; prs=[]
        for _,row in pdf.iterrows():
            img=cv2.imread(row.fp,cv2.IMREAD_GRAYSCALE); img=cv2.resize(img,(IMG,IMG)).astype(np.float32)/255.0
            it=vaT(torch.from_numpy(np.stack([img,img,img])).float()).unsqueeze(0).to(D)
            with torch.no_grad(): prs.append(m(it).argmax(1).item())
        pcor+=(Counter(prs).most_common(1)[0][0]==tl); ptot+=1
    pa=pcor/ptot if ptot>0 else 0
    print(f"    Test: SA={tm['acc']:.4f} | PA={pa:.4f} | F1={tm['f1']:.4f} | AUC={tm['auc']:.4f}")
    del m; torch.cuda.empty_cache()
    return {'sa':tm['acc'],'pa':pa,'f1':tm['f1'],'auc':tm['auc'],'prec':tm['prec'],'rec':tm['rec'],'cm':tm['cm']}

#--- RUN ---
MNS=['r50','eb0','cnx']; MNM={'r50':'ResNet50','eb0':'EfficientNetB0','cnx':'ConvNeXt-Tiny'}
all_r={}
for mn in MNS:
    print(f"\n{'#'*60}\n# {MNM[mn]}\n{'#'*60}")
    fr=[]
    for fi,fd in enumerate(si["f"]): fr.append(tf(mn,fi,fd))
    agg={}
    for k in ['sa','pa','f1','auc','prec','rec']:
        vs=[r[k] for r in fr]; agg[k]={'mean':np.mean(vs),'std':np.std(vs),'values':vs}
    all_r[mn]={'folds':fr,'aggregated':agg}
    print(f"\n  {MNM[mn]} FINAL:")
    print(f"    SAcc={agg['sa']['mean']*100:.2f}%±{agg['sa']['std']*100:.2f}% | PAcc={agg['pa']['mean']*100:.2f}%±{agg['pa']['std']*100:.2f}%")
    print(f"    Drop={(agg['sa']['mean']-agg['pa']['mean'])*100:.1f}pp | F1={agg['f1']['mean']*100:.2f}% | AUC={agg['auc']['mean']:.4f}")

# Save
with open(RD/'final_results.json','w') as f: json.dump(all_r,f,indent=2,default=str)
with open(RD/'final_results.pkl','wb') as f: pickle.dump(all_r,f)

#--- FIGURES ---
plt.rcParams.update({'font.family':'serif','font.size':11,'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight'})
MNC={'r50':'#e74c3c','eb0':'#2ecc71','cnx':'#e67e22'}
ML={'r50':'ResNet50\n(Chaddad 2025)','eb0':'EfficientNetB0\n(Notebook 3)','cnx':'ConvNeXt-Tiny\n(Proposed)'}

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
ml=list(all_r.keys()); x=np.arange(len(ml)); w=0.35
sas=[all_r[m]['aggregated']['sa']['mean']*100 for m in ml]
pas=[all_r[m]['aggregated']['pa']['mean']*100 for m in ml]
ses=[all_r[m]['aggregated']['sa']['std']*100 for m in ml]
pes=[all_r[m]['aggregated']['pa']['std']*100 for m in ml]
b1=ax1.bar(x-w/2,sas,w,yerr=ses,capsize=5,label='Slice-Level (Inflated)',color='#3498db',alpha=0.85)
b2=ax1.bar(x+w/2,pas,w,yerr=pes,capsize=5,label='Patient-Level (Real)',color='#e74c3c',alpha=0.85)
for bs in [b1,b2]:
    for b in bs: ax1.text(b.get_x()+b.get_width()/2.,b.get_height()+1,f'{b.get_height():.1f}%',ha='center',fontsize=8,fontweight='bold')
ax1.set_ylabel('Accuracy (%)',fontweight='bold'); ax1.set_title('A) Slice vs Patient Accuracy',fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels([ML[m] for m in ml],fontsize=9); ax1.legend(loc='lower right'); ax1.set_ylim(0,105); ax1.grid(axis='y',alpha=0.3)
ax1.axhline(y=99,color='gray',linestyle='--',alpha=0.4)
drops=[s-p for s,p in zip(sas,pas)]; bc=['#c0392b' if d>3 else '#e67e22' for d in drops]
bars=ax2.bar(x,drops,color=bc,alpha=0.85)
for b,d in zip(bars,drops): ax2.text(b.get_x()+b.get_width()/2.,b.get_height()+0.2,f'{d:.1f} pp',ha='center',fontsize=10,fontweight='bold')
ax2.set_ylabel('Drop (pp)',fontweight='bold'); ax2.set_title('B) Data Leakage Inflation',fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels([ML[m] for m in ml],fontsize=9); ax2.axhline(y=0,color='black',linewidth=0.5); ax2.grid(axis='y',alpha=0.3)
plt.suptitle('Figure 1: Exposing Data Leakage - Slice vs Patient Level',fontweight='bold',fontsize=14,y=1.02)
plt.tight_layout(); plt.savefig(RD/'figures'/'fig1_leakage.png'); plt.savefig(RD/'figures'/'fig1_leakage.pdf'); plt.close()
print("[OK] Fig1")

fig,ax=plt.subplots(figsize=(10,6))
mk=['sa','pa','f1','auc']; mlb=['Slice Acc','Patient Acc','F1 Macro','AUC']; xp=np.arange(len(mk)); w=0.25
for i,mn in enumerate(ml):
    vs=[all_r[mn]['aggregated'][k]['mean']*100 for k in mk]; es=[all_r[mn]['aggregated'][k]['std']*100 for k in mk]
    ax.bar(xp+(i-(len(ml)-1)/2)*w,vs,w,yerr=es,capsize=4,label=ML[mn],color=MNC[mn],alpha=0.85)
ax.set_ylabel('Score (%)'); ax.set_title('Figure 2: Model Comparison (Patient-Level 5-Fold CV)',fontweight='bold')
ax.set_xticks(xp); ax.set_xticklabels(mlb); ax.legend(loc='lower right'); ax.set_ylim(0,105); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(RD/'figures'/'fig2_comparison.png'); plt.savefig(RD/'figures'/'fig2_comparison.pdf'); plt.close()
print("[OK] Fig2")

best_m=max(ml,key=lambda m:all_r[m]['aggregated']['pa']['mean']); best_f=np.argmax([r['pa'] for r in all_r[best_m]['folds']])
cm=np.array(all_r[best_m]['folds'][best_f]['cm'])
fig,ax=plt.subplots(figsize=(7,6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=CN,yticklabels=CN,ax=ax,cbar_kws={'label':'Count'},annot_kws={'fontsize':14})
ax.set_title(f'Figure 3: {ML[best_m]} - Patient-Level (Fold {best_f+1})',fontweight='bold')
ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
plt.tight_layout(); plt.savefig(RD/'figures'/'fig3_cm.png'); plt.savefig(RD/'figures'/'fig3_cm.pdf'); plt.close()
print("[OK] Fig3")

rows=[]
for mn in ml:
    a=all_r[mn]['aggregated']
    rows.append({'Model':ML[mn].replace('\n',' '),
        'Slice Acc (%)':f"{a['sa']['mean']*100:.2f} ± {a['sa']['std']*100:.2f}",
        'Patient Acc (%)':f"{a['pa']['mean']*100:.2f} ± {a['pa']['std']*100:.2f}",
        'Drop (pp)':f"{(a['sa']['mean']-a['pa']['mean'])*100:.1f}",
        'F1 (%)':f"{a['f1']['mean']*100:.2f}",'AUC':f"{a['auc']['mean']:.4f}"})
sdf=pd.DataFrame(rows)
sdf.to_csv(RD/'tables'/'results_final.csv',index=False)
print(sdf.to_markdown(index=False))
with open(RD/'tables'/'results_final.md','w') as f: f.write("# Final Results\n\n"+sdf.to_markdown(index=False))

print(f"\n{'='*60}\nDONE! {RD}\n{'='*60}")
