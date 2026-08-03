#!/usr/bin/env python3
"""ROC-AUC comparison grid: models (rows) x datasets (columns), mean +/- SD
over 5 seeds. P1 (image-level, leaky) vs P2 (subject-independent, leakage-free)
vs cross-cohort (external validation), where cross-cohort data exists.

Reads aggregated seed summaries from results_baseline/ (protocol-split P1/P2
files + the two cross-cohort files) and renders a 3x4 panel grid matching the
requested reference layout, at 300 dpi.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "results_baseline")

# Okabe-Ito colorblind-safe categorical triple (blue / orange / green)
COL = {"p1": "#0072B2", "p2": "#E69F00", "cross": "#009E73"}

MODELS = [("vgg16", "VGG-16"), ("resnet50", "ResNet-50"),
          ("efficientnet_b0", "EfficientNet-B0")]

DATASETS = ["HandPD Spiral", "HandPD Meander", "Kaggle Spiral", "Kaggle Wave"]

WITHIN_FILES = {
    "HandPD Spiral": "summary_handpd_spiral_agg.csv",
    "HandPD Meander": "summary_handpd_meander_agg.csv",
    "Kaggle Spiral": "summary_kaggle_spiral_agg.csv",
    "Kaggle Wave": "summary_kaggle_wave_agg.csv",
}
# cross-cohort file that applies to each dataset column (None where not run)
CROSS_FILES = {
    "HandPD Spiral": "summary_cross_kaggle_spiral_to_handpd_spiral_agg.csv",
    "HandPD Meander": None,
    "Kaggle Spiral": "summary_cross_handpd_spiral_to_kaggle_spiral_agg.csv",
    "Kaggle Wave": None,
}


def load_within(fname):
    d = pd.read_csv(os.path.join(BASE, fname))
    return d[d.level == "subject"]


def load_cross(fname):
    if fname is None:
        return None
    d = pd.read_csv(os.path.join(BASE, fname))
    return d[d.level == "subject"]


WITHIN = {name: load_within(f) for name, f in WITHIN_FILES.items()}
CROSS = {name: load_cross(f) for name, f in CROSS_FILES.items()}

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 13,
                     "xtick.labelsize": 10, "ytick.labelsize": 11,
                     "legend.fontsize": 12})

fig, axes = plt.subplots(3, 4, figsize=(15.5, 12.2), sharey=True)
plt.subplots_adjust(hspace=0.65, wspace=0.12, top=0.88, bottom=0.09,
                    left=0.055, right=0.985)

for r, (mkey, mlabel) in enumerate(MODELS):
    for c, dname in enumerate(DATASETS):
        ax = axes[r, c]
        w = WITHIN[dname]
        p1 = w[(w.model == mkey) & (w.protocol == "P1_image")]
        p2 = w[(w.model == mkey) & (w.protocol == "P2_subject")]
        means = [p1.roc_auc_mean.values[0], p2.roc_auc_mean.values[0]]
        stds = [p1.roc_auc_std.values[0], p2.roc_auc_std.values[0]]
        colors = [COL["p1"], COL["p2"]]

        cdf = CROSS[dname]
        if cdf is not None:
            crow = cdf[cdf.model == mkey]
            means.append(crow.roc_auc_mean.values[0])
            stds.append(crow.roc_auc_std.values[0])
            colors.append(COL["cross"])

        n = len(means)
        xs = np.arange(n)
        bars = ax.bar(xs, means, width=0.62, color=colors, edgecolor="black",
                      linewidth=0.9, zorder=2)
        ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor="#333333",
                   elinewidth=1.4, capsize=4, capthick=1.4, zorder=3)
        for x, m, s in zip(xs, means, stds):
            ax.text(x, m + s + 0.015, f"{m:.3f}", ha="center", va="bottom",
                   fontsize=10.5, fontweight="bold")

        ax.set_title(dname, fontsize=12.5, pad=8)
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_xticks([])
        ax.set_ylim(0.4, 1.02)
        ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if c == 0:
            ax.set_ylabel("ROC-AUC")
            ax.set_yticks(np.arange(0.4, 1.01, 0.1))

    # per-row bold model header, centered above that row's 4 panels
    left_box = axes[r, 0].get_position()
    right_box = axes[r, -1].get_position()
    x_center = (left_box.x0 + right_box.x1) / 2
    y_top = left_box.y1 + 0.028
    fig.text(x_center, y_top, f"({chr(65 + r)}) {mlabel}", ha="center",
             va="bottom", fontsize=15, fontweight="bold")

legend_handles = [
    Patch(facecolor=COL["p1"], edgecolor="black", label="P1: Image-level (leaky)"),
    Patch(facecolor=COL["p2"], edgecolor="black", label="P2: Subject-independent (leakage-free)"),
    Patch(facecolor=COL["cross"], edgecolor="black", label="Cross-cohort (external validation)"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=3,
          bbox_to_anchor=(0.5, 0.985), frameon=True, framealpha=0.95,
          fontsize=12.5)

fig.suptitle("ROC-AUC Comparison with Error Bars (mean $\\pm$ SD over 5 seeds)",
            fontsize=18, fontweight="bold", y=1.015)

fig.text(0.055, 0.035,
        "P1: Image-level random k-fold CV (same subject may appear in train and test)   |   "
        "P2: Subject-grouped k-fold CV / LOSO (no subject overlap)",
        fontsize=10.5, ha="left", va="center")
fig.text(0.055, 0.015,
        "Cross-cohort: Train on one cohort, test on the other (HandPD $\\leftrightarrow$ Kaggle)",
        fontsize=10.5, ha="left", va="center")
fig.text(0.985, 0.015, "Error bars: mean $\\pm$ SD over 5 random seeds",
        fontsize=10.5, ha="right", va="center")

for ext in ("png", "pdf"):
    fig.savefig(f"seed_variability_grid.{ext}", dpi=300, bbox_inches="tight")
print("seed_variability_grid figure written")
