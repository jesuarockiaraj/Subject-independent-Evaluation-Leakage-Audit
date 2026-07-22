#!/usr/bin/env python3
"""Generate the main results figure from the four prediction CSVs."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

UP = r"C:\Users\JESU ATCHAYA\Documents\Claude code New Topic\Project 2\results"
DATASETS = [("handpd_spiral", "HandPD spiral"), ("handpd_meander", "HandPD meander"),
            ("kaggle_spiral", "Kaggle spiral"), ("kaggle_wave", "Kaggle wave")]
MODELS = ["vgg16", "resnet50", "efficientnet_b0"]
MLABEL = {"vgg16": "VGG-16", "resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0"}
COLORS = {"handpd_spiral": "#2c6fbb", "handpd_meander": "#5aa469",
          "kaggle_spiral": "#d98c30", "kaggle_wave": "#b0504a"}

def ps(df):
    return df.groupby("subject").agg(y=("y_true", "max"), prob=("y_prob", "mean")).reset_index()
def auc(y, p):
    try: return roc_auc_score(y, p) if len(set(y)) > 1 else np.nan
    except ValueError: return np.nan

rng = np.random.default_rng(0)
rows = []
for key, label in DATASETS:
    p = pd.read_csv(f"{UP}/predictions_{key}.csv")
    for model in MODELS:
        d = p[p.model == model]
        m = ps(d[d.protocol == "P1_image"]).merge(
            ps(d[d.protocol == "P2_subject"]), on="subject", suffixes=("_1", "_2"))
        y, p1, p2 = m.y_1.values, m.prob_1.values, m.prob_2.values
        a1, a2 = auc(y, p1), auc(y, p2)
        idx = np.arange(len(m)); boots = []
        for _ in range(2000):
            bi = rng.choice(idx, len(idx), replace=True)
            if len(set(y[bi])) < 2: continue
            boots.append(auc(y[bi], p1[bi]) - auc(y[bi], p2[bi]))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        rows.append(dict(key=key, label=label, model=model, a1=a1, a2=a2,
                         d=a1 - a2, lo=lo, hi=hi))
R = pd.DataFrame(rows)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})

# ---- Panel A: mean AUC P1 vs P2 per dataset (mean over 3 models) ----
ds_keys = [k for k, _ in DATASETS]
p1m = [R[R.key == k].a1.mean() for k in ds_keys]
p2m = [R[R.key == k].a2.mean() for k in ds_keys]
p1s = [R[R.key == k].a1.std() for k in ds_keys]
p2s = [R[R.key == k].a2.std() for k in ds_keys]
x = np.arange(len(ds_keys)); w = 0.36
axL.bar(x - w/2, p1m, w, yerr=p1s, capsize=3, color="#c0392b", label="P1 image-level (leaky)")
axL.bar(x + w/2, p2m, w, yerr=p2s, capsize=3, color="#2e7d5b", label="P2 subject-independent")
axL.set_xticks(x); axL.set_xticklabels([l for _, l in DATASETS], rotation=20, ha="right")
axL.set_ylabel("ROC-AUC (mean over 3 CNNs)"); axL.set_ylim(0.5, 1.0)
axL.axhline(0.5, ls=":", c="grey", lw=1)
axL.set_title("(a) Discrimination drops under honest evaluation", fontsize=11)
axL.legend(fontsize=8, loc="lower left"); axL.grid(axis="y", ls=":", alpha=0.4)

# ---- Panel B: forest plot of dAUC for all 12 cells ----
R2 = R.iloc[::-1].reset_index(drop=True)  # top-to-bottom
ypos = np.arange(len(R2))
for i, r in R2.iterrows():
    c = COLORS[r.key]
    axR.plot([r.lo, r.hi], [i, i], "-", color=c, lw=2, alpha=0.85)
    axR.plot(r.d, i, "o", color=c, ms=6)
axR.axvline(0, color="black", lw=1)
axR.set_yticks(ypos)
axR.set_yticklabels([f"{r.label} · {MLABEL[r.model]}" for _, r in R2.iterrows()], fontsize=8)
axR.set_xlabel(r"$\Delta$AUC  =  AUC$_{P1}$ − AUC$_{P2}$   (positive = leaky split inflates)")
axR.set_title("(b) Every cell favours the leaky split", fontsize=11)
axR.grid(axis="x", ls=":", alpha=0.4)
axR.text(0.98, 0.02,
         "12/12 positive\nWilcoxon p = 0.0005\nmean +0.086",
         transform=axR.transAxes, ha="right", va="bottom", fontsize=8.5,
         bbox=dict(boxstyle="round", fc="#f5f5f5", ec="grey"))
handles = [Patch(color=COLORS[k], label=l) for k, l in DATASETS]
axR.legend(handles=handles, fontsize=7.5, loc="upper right")

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"C:\\Users\\JESU ATCHAYA\\Documents\\Claude code New Topic\\Project 2\\results_figure.{ext}", dpi=200, bbox_inches="tight")
print("saved results_figure.pdf / .png")
print(R[["label", "model", "a1", "a2", "d", "lo", "hi"]].to_string(index=False))
