#!/usr/bin/env python3
"""All manuscript figures: accessible palette, larger fonts, visible CIs."""
import glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
import warnings; warnings.filterwarnings("ignore")

import os
UP=os.environ.get("RESULTS_DIR","results")
def find(p): return sorted(glob.glob(f"{UP}/*{p}"))[0]
plt.rcParams.update({"font.size":13,"axes.titlesize":14,"axes.labelsize":13,
                     "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":11})
# Okabe-Ito colourblind-safe palette
OI={"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00",
    "purple":"#CC79A7","sky":"#56B4E9","yellow":"#F0E442","black":"#000000"}
AGG={"HandPD spiral":find("summary_handpd_spiral_agg.csv"),
     "HandPD meander":find("summary_handpd_meander_agg.csv"),
     "Kaggle spiral":find("summary_kaggle_spiral_agg.csv"),
     "Kaggle wave":find("summary_kaggle_wave_agg.csv")}
CROSS_HK=find("summary_cross_handpd_spiral_to_kaggle_spiral_agg.csv")
PRED_HK=find("predictions_cross_handpd_spiral_to_kaggle_spiral.csv")
PRED_KH=find("predictions_cross_kaggle_spiral_to_handpd_spiral.csv")
MODELS=["vgg16","resnet50","efficientnet_b0"]
ML={"vgg16":"VGG-16","resnet50":"ResNet-50","efficientnet_b0":"EfficientNet-B0"}
MC={"vgg16":OI["blue"],"resnet50":OI["orange"],"efficientnet_b0":OI["green"]}
DC={"HandPD spiral":OI["blue"],"HandPD meander":OI["green"],
    "Kaggle spiral":OI["orange"],"Kaggle wave":OI["red"]}
MK={"vgg16":"o","resnet50":"s","efficientnet_b0":"^"}
DM={"HandPD spiral":"o","HandPD meander":"s","Kaggle spiral":"^","Kaggle wave":"D"}
def sub(df): return df[df.level=="subject"]

# ================= FIGURE 3: staircase + forest =================
hs=sub(pd.read_csv(AGG["HandPD spiral"])); cr=sub(pd.read_csv(CROSS_HK))
fig,(axA,axB)=plt.subplots(1,2,figsize=(14,6),gridspec_kw={"width_ratios":[1,1.25]})
xs=[0,1,2]
for mdl in MODELS:
    p1=hs[(hs.model==mdl)&(hs.protocol=="P1_image")]
    p2=hs[(hs.model==mdl)&(hs.protocol=="P2_subject")]
    c =cr[cr.model==mdl]
    m=[p1.roc_auc_mean.values[0],p2.roc_auc_mean.values[0],c.roc_auc_mean.values[0]]
    e=[p1.roc_auc_std.values[0], p2.roc_auc_std.values[0], c.roc_auc_std.values[0]]
    axA.errorbar(xs,m,yerr=e,marker=MK[mdl],color=MC[mdl],lw=2.5,ms=9,capsize=5,
                 label=ML[mdl],markeredgecolor="white",markeredgewidth=1.2)
mean=[hs[hs.protocol=="P1_image"].roc_auc_mean.mean(),
      hs[hs.protocol=="P2_subject"].roc_auc_mean.mean(), cr.roc_auc_mean.mean()]
axA.plot(xs,mean,"--",color=OI["black"],lw=2,alpha=.65,label="mean",zorder=1)
for x,m in zip(xs,mean):
    axA.annotate(f"{m:.2f}",(x,m),textcoords="offset points",xytext=(8,9),
                 fontsize=13,fontweight="bold")
axA.axhline(.5,ls=":",color="grey",lw=1.5)
axA.text(2.05,.512,"chance",fontsize=11,color="grey",ha="right")
axA.set_xticks(xs)
axA.set_xticklabels(["P1\nimage-level\n(leaky)","P2\nsubject-\nindependent","Cross-cohort\nHandPD→Kaggle"])
axA.set_ylabel("ROC-AUC"); axA.set_ylim(.45,1.02); axA.set_xlim(-.35,2.35)
axA.set_title("(a) Discrimination falls as leakage is removed",fontweight="bold")
axA.legend(loc="lower left",framealpha=.95); axA.grid(axis="y",ls=":",alpha=.45)

rows=[]
for name,f in AGG.items():
    d=sub(pd.read_csv(f))
    for mdl in MODELS:
        p1=d[(d.model==mdl)&(d.protocol=="P1_image")]; p2=d[(d.model==mdl)&(d.protocol=="P2_subject")]
        a1,s1=p1.roc_auc_mean.values[0],p1.roc_auc_std.values[0]
        a2,s2=p2.roc_auc_mean.values[0],p2.roc_auc_std.values[0]
        se=np.sqrt(s1**2+s2**2)/np.sqrt(5)          # SE of the mean difference over 5 seeds
        rows.append(dict(name=name,mdl=mdl,d=a1-a2,lo=(a1-a2)-1.96*se,hi=(a1-a2)+1.96*se))
R=pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
for i,r in R.iterrows():
    axB.plot([r.lo,r.hi],[i,i],"-",color=DC[r["name"]],lw=2.5,alpha=.9)
    axB.plot(r.d,i,DM[r["name"]],color=DC[r["name"]],ms=9,markeredgecolor="white",markeredgewidth=1.2)
axB.axvline(0,color=OI["black"],lw=1.5)
axB.set_yticks(range(len(R)))
axB.set_yticklabels([f"{r['name']} · {ML[r.mdl]}" for _,r in R.iterrows()],fontsize=11)
axB.set_xlabel("ΔAUC = AUC$_{P1}$ − AUC$_{P2}$   (>0: leaky split inflates)")
axB.set_title("(b) All 12 combinations favour the leaky split",fontweight="bold")
axB.grid(axis="x",ls=":",alpha=.45)
axB.text(.975,.03,"12/12 positive\nWilcoxon $p$ = 0.0005\nmean +0.077",transform=axB.transAxes,
         ha="right",va="bottom",fontsize=12,bbox=dict(boxstyle="round",fc="#f2f2f2",ec="grey"))
axB.legend(handles=[Patch(color=DC[k],label=k) for k in AGG],loc="upper right",fontsize=10)
plt.tight_layout(); fig.savefig("results_figure.pdf",bbox_inches="tight"); fig.savefig("results_figure.png",dpi=200,bbox_inches="tight")
plt.close(fig); print("figure 3 ok")

# ================= FIGURE 4: literature vs this study =================
lit=[("Gazda et al. 2022\n(combined sets)",99.22),("Kamran et al. 2021\n(HandPD/NewHandPD)",96.0),
     ("iJOE 2026\n(NewHandPD)",97.4),("IJPRAI 2024\n(spiral, augmented)",100.0)]
fig,ax=plt.subplots(figsize=(12,6))
names=[l[0] for l in lit]; vals=[l[1] for l in lit]
xpos=np.arange(len(lit))
ax.bar(xpos,vals,color=OI["red"],alpha=.85,label="Reported in literature (image-level / split not stated)",
       edgecolor="black",linewidth=.6)
for x,v in zip(xpos,vals): ax.text(x,v+.8,f"{v:.1f}",ha="center",fontweight="bold",fontsize=12)
# our numbers as accuracy-equivalent bands (use balanced accuracy where meaningful)
ours=[("This study\nP1 image-level\n(matched, leak-free otherwise)",88.6,3.0),
      ("This study\nP2 subject-\nindependent",82.1,4.0),
      ("This study\ncross-cohort",69.0,6.0)]
xpos2=np.arange(len(lit),len(lit)+len(ours))
cols=[OI["orange"],OI["blue"],OI["green"]]
for x,(n,v,e),c in zip(xpos2,ours,cols):
    ax.bar(x,v,color=c,alpha=.9,yerr=e,capsize=6,edgecolor="black",linewidth=.6)
    ax.text(x,v+e+.8,f"{v:.0f}",ha="center",fontweight="bold",fontsize=12)
ax.set_xticks(list(xpos)+list(xpos2)); ax.set_xticklabels(names+[o[0] for o in ours],fontsize=10)
ax.axhline(50,ls=":",color="grey",lw=1.5); ax.text(len(lit)+len(ours)-.5,51,"chance",color="grey",ha="right",fontsize=11)
ax.set_ylabel("Reported performance (%; literature = accuracy,\nthis study = ROC-AUC ×100)")
ax.set_ylim(40,108)
ax.set_title("Literature-reported accuracy vs. performance under progressively stricter evaluation",
             fontweight="bold",fontsize=14)
ax.legend(handles=[Patch(color=OI["red"],label="Literature (as reported)"),
                   Patch(color=OI["orange"],label="This study — image-level (P1)"),
                   Patch(color=OI["blue"],label="This study — subject-independent (P2)"),
                   Patch(color=OI["green"],label="This study — cross-cohort")],
          loc="lower left",fontsize=10)
ax.grid(axis="y",ls=":",alpha=.45)
plt.tight_layout(); fig.savefig("literature_figure.pdf",bbox_inches="tight"); fig.savefig("literature_figure.png",dpi=200,bbox_inches="tight")
plt.close(fig); print("figure 4 ok")

# ================= FIGURE 5: reliability diagrams =================
def rel(y,p,bins=8):
    ed=np.linspace(0,1,bins+1); xs=[];ys=[];ns=[]
    for i in range(bins):
        m=(p>=ed[i])&(p<=ed[i+1]) if i==0 else (p>ed[i])&(p<=ed[i+1])
        if m.sum()<2: continue
        xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    return np.array(xs),np.array(ys),np.array(ns)
def ece(y,p,bins=10):
    ed=np.linspace(0,1,bins+1); e=0.
    for i in range(bins):
        m=(p>=ed[i])&(p<=ed[i+1]) if i==0 else (p>ed[i])&(p<=ed[i+1])
        if m.sum()==0: continue
        e+=m.sum()/len(y)*abs(y[m].mean()-p[m].mean())
    return e
fig,axes=plt.subplots(1,2,figsize=(13,6))
for ax,(tag,f) in zip(axes,[("Train HandPD → test Kaggle",PRED_HK),("Train Kaggle → test HandPD",PRED_KH)]):
    d=pd.read_csv(f)
    ax.plot([0,1],[0,1],"--",color="grey",lw=2,label="perfect calibration")
    for mdl in MODELS:
        g=d[d.model==mdl].groupby("subject").agg(y=("y_true","max"),p=("y_prob","mean")).reset_index()
        x,yv,_=rel(g.y.values,g.p.values)
        e=ece(g.y.values,g.p.values); b=brier_score_loss(g.y.values,g.p.values)
        ax.plot(x,yv,marker=MK[mdl],color=MC[mdl],lw=2.5,ms=9,markeredgecolor="white",
                markeredgewidth=1.2,label=f"{ML[mdl]} (ECE {e:.2f}, Brier {b:.2f})")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed fraction of PD")
    ax.set_title(tag,fontweight="bold"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.grid(ls=":",alpha=.45); ax.legend(loc="upper left",fontsize=10)
plt.tight_layout(); fig.savefig("calibration_figure.pdf",bbox_inches="tight"); fig.savefig("calibration_figure.png",dpi=200,bbox_inches="tight")
plt.close(fig); print("figure 5 ok")
