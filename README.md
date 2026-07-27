# Subject-Independent Evaluation of Deep Learning for Parkinson's Disease Detection from Handwriting Images

Code and results for the paper **"Subject-independent evaluation of deep learning for
Parkinson's disease detection from handwriting images: how much accuracy is real?"**

Deep learning on hand-drawn spirals, meanders and waves is routinely reported to detect
Parkinson's disease (PD) with 95–99% accuracy. These numbers are typically obtained by
splitting **images** at random. Because each participant contributes several drawings, this
lets drawings from the same person fall on both sides of the split, so a model can succeed by
re-identifying individuals rather than recognising disease. This repository contains a
pipeline that evaluates the same models under two protocols that are **identical except for
the split unit** — image-level (P1) versus subject-independent (P2) — plus cross-cohort
external validation, calibration and explainability analyses, and the leakage audit.

**Key finding.** Across all 12 architecture×dataset combinations, subject-independent
evaluation reduced ROC-AUC (mean −0.077; Wilcoxon *p* = 0.0005), and cross-cohort validation
reduced it further to a mean of ≈0.68 — above chance, but far below the reported 95–99%. The
most widely used benchmark's standard split is entirely subject-leaky (every test subject also
appears in training), so results on it are inflated by construction.

---

## Repository layout

```
.
├── README.md
├── LICENSE                     # add before making public (MIT/Apache-2.0 recommended)
├── requirements.txt
├── .gitignore
├── run_all.sh                  # reproduces every table and figure
├── src/
│   ├── pd_loso_pipeline.py     # main pipeline: loader, P1/P2 protocols, metrics,
│   │                           #   bootstrap CIs, cross-dataset mode, leakage probe
│   ├── gradcam_analysis.py     # Grad-CAM explainability + runtime/cost profiling
│   ├── make_figures_v2.py      # generates results/calibration/literature figures
│   └── make_gradcam_fig.py     # generates the quantitative Grad-CAM figure
├── results/                    # all CSVs behind the paper's numbers (see mapping below)
│   ├── summary_*_agg.csv        #   mean ± SD over seeds  →  the values in the paper's tables
│   ├── summary_*.csv            #   per-seed metrics + subject-level bootstrap CIs
│   ├── predictions_*.csv        #   raw per-image predictions (recompute any metric from these)
│   ├── gradcam_metrics_*.csv    #   on-trace saliency mass + identity-gap statistics
│   ├── runtime_*.csv            #   wall-clock time, peak GPU memory, model size
│   └── *.png                    #   Grad-CAM panels
└── docs/
    └── data_audit.md           # dataset provenance, NewHandPD exclusion, Kaggle leakage
```

---

## Datasets (download separately — not redistributed here)

The datasets are **not** included in this repository; download them from their original
sources and place them under a local `data/` directory (git-ignored).

| Dataset | Source | Notes |
|---|---|---|
| HandPD / NewHandPD (spiral, meander) | Botucatu/UNESP dataset page (Pereira et al.) | Subject ID is encoded in the filename (e.g. `0092-3.jpg` → subject `0092`). |
| Kaggle "Parkinson's Drawings" (spiral, wave) | Kaggle (Zham et al.) | Subject ID recoverable from filename (e.g. `V03PE05.png` → volunteer `V03`). Its provided train/test split is subject-leaky. |

Expected structure (paths are passed on the command line, so exact names can differ):

```
data/
├── HandPD/           # spiral + meander class folders
└── Kaggle/           # Parkinson's Drawings spiral + wave
```

> **NewHandPD is deliberately excluded** from the paper: the redistribution available to us was
> heavily augmented (parsed as 8 subjects × 30 images rather than the expected 66 subjects × 4)
> and lacked recoverable subject identifiers, which would reintroduce the very leakage the study
> measures. See `docs/data_audit.md`.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended. Experiments were run on Python 3.12 with PyTorch 2.x + CUDA.
For GPU acceleration, install the CUDA build of PyTorch matching your driver from
<https://pytorch.org/get-started/locally/>.

---

## Reproducing the results

Every result comes from a single configurable script. `run_all.sh` chains the full set;
individual commands:

```bash
# Within-cohort: P1 (image-level) vs P2 (subject-independent), 5 seeds, per dataset/task
python src/pd_loso_pipeline.py --dataset handpd --root data/HandPD --task spiral  --seeds 1 2 3 4 5
python src/pd_loso_pipeline.py --dataset handpd --root data/HandPD --task meander --seeds 1 2 3 4 5
python src/pd_loso_pipeline.py --dataset kaggle --root data/Kaggle --task spiral  --seeds 1 2 3 4 5
python src/pd_loso_pipeline.py --dataset kaggle --root data/Kaggle --task wave    --seeds 1 2 3 4 5

# Cross-cohort external validation (train on one cohort, test on the other)
python src/pd_loso_pipeline.py --dataset handpd --root data/HandPD --task spiral \
       --test-root data/Kaggle --test-task spiral --test-name kaggle --seeds 1 2 3 4 5
python src/pd_loso_pipeline.py --dataset kaggle --root data/Kaggle --task spiral \
       --test-root data/HandPD --test-task spiral --test-name handpd --seeds 1 2 3 4 5

# Explainability (Grad-CAM) + runtime/cost profiling
python src/gradcam_analysis.py --dataset handpd --root data/HandPD --task spiral --seeds 1 2 3

# Figures
python src/make_figures_v2.py
python src/make_gradcam_fig.py
```

Fixed hyperparameters (identical across every architecture, dataset and protocol): input
224×224, ImageNet normalisation; augmentation applied to training folds only; AdamW, lr 1e-4,
weight decay 1e-4, batch size 16, ≤40 epochs with early stopping on validation ROC-AUC
(patience 8); K = 10 folds (auto-reduced for small cohorts); seeds {1,2,3,4,5}; bootstrap
B = 2000 over subjects.

---

## Which file backs which result

Every number in the paper can be traced to a file in `results/`. The `*_agg.csv` files hold the
mean ± SD values printed in the tables; `predictions_*.csv` hold raw per-image outputs from
which **any** metric can be recomputed independently.

| Paper item | File(s) in `results/` |
|---|---|
| Table 3 — main AUC (P1 vs P2) | `summary_handpd_spiral_agg.csv`, `summary_handpd_meander_agg.csv`, `summary_kaggle_spiral_agg.csv`, `summary_kaggle_wave_agg.csv` |
| Table 4 / Fig — clinical metrics (Kaggle) | `summary_kaggle_spiral_agg.csv`, `summary_kaggle_wave_agg.csv` |
| Table 5 — cross-cohort external validation | `summary_cross_handpd_spiral_to_kaggle_spiral_agg.csv`, `summary_cross_kaggle_spiral_to_handpd_spiral_agg.csv` |
| Calibration (Brier/ECE/PR-AUC) | computed from `predictions_cross_*.csv` |
| Table 8 / Figs — Grad-CAM (null result) | `gradcam_metrics_handpd_spiral_agg.csv`, `gradcam_panel_handpd_spiral.png` |
| Table 9 — runtime / cost | `runtime_handpd_spiral.csv` |
| Kaggle leakage audit (25/25, 24/24 overlap) | `provided_split_leakage` rows in `summary_kaggle_spiral.csv`, `summary_kaggle_wave.csv` |
| Per-seed values + bootstrap CIs | `summary_*.csv` (non-agg) |
| Raw per-image predictions | `predictions_*.csv` |

All released CSVs use five seeds and are consistent with the reported tables.

---

## Citation

If you use this code or the released results, please cite the paper (details to follow on
publication). A `CITATION.cff` will be added with the final reference.

## License

Released under the terms in `LICENSE` (add an MIT or Apache-2.0 license before making the
repository public).
