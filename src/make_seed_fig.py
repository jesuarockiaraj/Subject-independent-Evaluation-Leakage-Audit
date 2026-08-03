#!/usr/bin/env python3
"""Seed-variability figure: per-cell ROC-AUC mean +/- SD across 5 seeds, P1 vs P2.
Built from *_agg.csv (mean, std). Honest depiction of run-to-run spread; NOT a
synthetic boxplot (raw per-seed values were not available in the provided files)."""
import glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import os
UP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_baseline")
def find(p): return sorted(glob.glob(f"{UP}/*{p}"))[0]
plt.rcParams.update({"font.size":13,"axes.titlesize":14,"axes.labelsize":13,
                     "xtick.labelsize":11,"ytick.labelsize":12,"legend.fontsize":11})
OI={"p1":"#D55E00","p2":"#0072B2"}
AGG={"HandPD\nspiral":find("summary_handpd_spiral_agg.csv"),
     "HandPD\nmeander":find("summary_handpd_meander_agg.csv"),
     "Kaggle\nspiral":find("summary_kaggle_spiral_agg.csv"),
     "Kaggle\nwave":find("summary_kaggle_wave_agg.csv")}
MODELS=["vgg16","resnet50","efficientnet_b0"]
ML={"vgg16":"VGG-16","resnet50":"ResNet-50","efficientnet_b0":"EfficientNet-B0"}
def sub(df): return df[df.level=="subject"]

fig,axes=plt.subplots(1,3,figsize=(15,5.4),sharey=True)
for ax,mdl in zip(axes,MODELS):
    xs=np.arange(len(AGG)); w=0.3
    for k,(proto,c,lab) in enumerate([("P1_image",OI["p1"],"P1 image-level"),
                                       ("P2_subject",OI["p2"],"P2 subject-independent")]):
        means,stds=[],[]
        for name,f in AGG.items():
            d=sub(pd.read_csv(f)); r=d[(d.model==mdl)&(d.protocol==proto)]
            means.append(r.roc_auc_mean.values[0]); stds.append(r.roc_auc_std.values[0])
        means=np.array(means); stds=np.array(stds)
        pos=xs+(k-0.5)*w
        # mean+/-SD as a thick bar + whisker (honest spread over 5 seeds)
        ax.bar(pos,means,w,color=c,alpha=.35,edgecolor=c,linewidth=1.2,zorder=1)
        ax.errorbar(pos,means,yerr=stds,fmt="o",color=c,ms=7,capsize=5,lw=2,
                    markeredgecolor="white",markeredgewidth=1,zorder=3,label=lab)
    ax.axhline(0.5,ls=":",color="grey",lw=1.3)
    ax.set_title(ML[mdl],fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(list(AGG.keys()))
    ax.set_ylim(0.45,1.0); ax.grid(axis="y",ls=":",alpha=.4)
axes[0].set_ylabel("ROC-AUC (mean $\\pm$ SD over 5 seeds)")
axes[0].legend(loc="lower left",fontsize=10,framealpha=.95)
axes[2].text(0.98,0.02,"whisker = across-seed SD",transform=axes[2].transAxes,
             ha="right",va="bottom",fontsize=9,style="italic",color="#555")
plt.suptitle("Run-to-run variability: subject-independent (P2) vs image-level (P1) across 5 seeds",
             fontsize=14,fontweight="bold",y=1.02)
plt.tight_layout()
for e in ("pdf","png"): fig.savefig(f"seed_variability.{e}",dpi=300,bbox_inches="tight")
print("seed_variability figure written")
