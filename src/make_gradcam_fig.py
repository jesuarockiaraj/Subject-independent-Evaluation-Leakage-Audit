#!/usr/bin/env python3
"""Quantitative Grad-CAM figure: on-trace mass and identity gap, per seed."""
import glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
UP=os.environ.get("RESULTS_DIR","results")
d=pd.read_csv(sorted(glob.glob(f"{UP}/*gradcam_metrics_handpd_spiral.csv"))[0])
plt.rcParams.update({"font.size":13,"axes.titlesize":14,"axes.labelsize":13,
                     "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":11})
OI={"p1":"#D55E00","p2":"#0072B2"}
MODELS=["vgg16","resnet50","efficientnet_b0"]
ML={"vgg16":"VGG-16","resnet50":"ResNet-50","efficientnet_b0":"EfficientNet-B0"}
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.6))
w=0.34; x=np.arange(len(MODELS))
for ax,col,title,ylab in [
    (ax1,"on_trace_mass","(a) On-trace saliency mass","fraction of saliency on the drawn stroke"),
    (ax2,"identity_gap","(b) Identity gap (within − between subject)","saliency-correlation gap")]:
    for k,(proto,lab) in enumerate([("P1_image","P1 image-level"),("P2_subject","P2 subject-independent")]):
        m=[d[(d.model==mm)&(d.protocol==proto)][col].mean() for mm in MODELS]
        s=[d[(d.model==mm)&(d.protocol==proto)][col].std()  for mm in MODELS]
        c=OI["p1" if k==0 else "p2"]
        ax.bar(x+(k-0.5)*w,m,w,yerr=s,capsize=5,color=c,alpha=.85,edgecolor="black",linewidth=.6,label=lab)
        for i,mm in enumerate(MODELS):   # per-seed points
            v=d[(d.model==mm)&(d.protocol==proto)][col].values
            ax.scatter(np.full_like(v,x[i]+(k-0.5)*w,dtype=float),v,color="black",s=22,zorder=5,alpha=.75)
    ax.set_xticks(x); ax.set_xticklabels([ML[m2] for m2 in MODELS])
    ax.set_title(title,fontweight="bold"); ax.set_ylabel(ylab)
    ax.grid(axis="y",ls=":",alpha=.45); ax.legend()
ax2.axhline(0,color="black",lw=1)
ax1.annotate("hypothesis predicted\nP1 < P2 (not observed)",xy=(.5,.93),xycoords="axes fraction",
             ha="center",fontsize=11,bbox=dict(boxstyle="round",fc="#f7f7f7",ec="grey"))
ax2.annotate("hypothesis predicted\nP1 > P2 (not observed)",xy=(.5,.93),xycoords="axes fraction",
             ha="center",fontsize=11,bbox=dict(boxstyle="round",fc="#f7f7f7",ec="grey"))
plt.tight_layout()
for e in ("pdf","png"): fig.savefig(f"gradcam_quant.{e}",dpi=200,bbox_inches="tight")
print("gradcam_quant ok")
