#!/usr/bin/env bash
# Reproduce every table and figure in the paper.
# Prerequisite: datasets downloaded under ./data (see README). Adjust paths as needed.
set -euo pipefail

HANDPD="data/HandPD"
KAGGLE="data/Kaggle"
SEEDS="1 2 3 4 5"

echo "== Within-cohort: P1 (image-level) vs P2 (subject-independent) =="
python src/pd_loso_pipeline.py --dataset handpd --root "$HANDPD" --task spiral  --seeds $SEEDS
python src/pd_loso_pipeline.py --dataset handpd --root "$HANDPD" --task meander --seeds $SEEDS
python src/pd_loso_pipeline.py --dataset kaggle --root "$KAGGLE" --task spiral  --seeds $SEEDS
python src/pd_loso_pipeline.py --dataset kaggle --root "$KAGGLE" --task wave    --seeds $SEEDS

echo "== Cross-cohort external validation (both directions) =="
python src/pd_loso_pipeline.py --dataset handpd --root "$HANDPD" --task spiral \
       --test-root "$KAGGLE" --test-task spiral --test-name kaggle --seeds $SEEDS
python src/pd_loso_pipeline.py --dataset kaggle --root "$KAGGLE" --task spiral \
       --test-root "$HANDPD" --test-task spiral --test-name handpd --seeds $SEEDS

echo "== Explainability (Grad-CAM) + runtime/cost profiling =="
python src/gradcam_analysis.py --dataset handpd --root "$HANDPD" --task spiral --seeds 1 2 3

echo "== Figures =="
python src/make_figures_v2.py
python src/make_gradcam_fig.py

echo "== Done. See results/ for CSVs and figures. =="
