#!/usr/bin/env python3
"""
pd_loso_pipeline.py
===================
Subject-independent evaluation of deep learning for Parkinson's disease (PD)
detection from handwriting images.

This single script reproduces the paper's central experiment: it runs three
ImageNet-pretrained backbones (VGG-16, ResNet-50, EfficientNet-B0) on the same
data under TWO protocols that differ ONLY in the unit of the cross-validation
split:

    P1  image-level random k-fold      -> the field's usual (leaky) practice
    P2  subject-grouped k-fold / LOSO  -> honest, generalises to new people

It computes accuracy, balanced accuracy, sensitivity, specificity, F1, ROC-AUC
and MCC with SUBJECT-LEVEL bootstrap confidence intervals, plus a leakage probe
(same-subject-in-train sensitivity, and a label-permutation floor). Everything
is written to CSV so the manuscript tables can be filled directly.

--------------------------------------------------------------------------------
WHY THIS FILE EXISTS
--------------------------------------------------------------------------------
The paper's thesis is that reported 95-99% accuracies are inflated by image-level
splitting. Producing that evidence REQUIRES running the models on the real data;
the numbers must be measured, not assumed. Run this to generate them.

--------------------------------------------------------------------------------
EXPECTED DATA LAYOUT
--------------------------------------------------------------------------------
HandPD / NewHandPD keep subject identity in the FILENAME, e.g. `sp1-P3.jpg`
(spiral #1 from Patient 3) or `me2-H5.jpg` (meander #2 from Healthy #5). We parse
the trailing H<k>/P<k> token as the subject id and the H/P letter as the label.

    <HANDPD_ROOT>/**/sp*-*.jpg        spiral images (subject id in name)
    <HANDPD_ROOT>/**/me*-*.jpg        meander images

The Kaggle "Parkinson's Drawings" set does NOT publish subject ids and is laid
out by class folder:

    <KAGGLE_ROOT>/spiral/training/healthy/*.png
    <KAGGLE_ROOT>/spiral/training/parkinson/*.png
    <KAGGLE_ROOT>/spiral/testing/healthy/*.png
    <KAGGLE_ROOT>/spiral/testing/parkinson/*.png
    <KAGGLE_ROOT>/wave/...

For Kaggle we can therefore ONLY run P1; the script prints the provenance warning
and refuses P2 (this is itself a finding, see paper Section 6.3).

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    pip install torch torchvision scikit-learn pillow numpy pandas

    # HandPD spiral, all three models, both protocols:
    python pd_loso_pipeline.py --dataset handpd --root /path/HandPD --task spiral

    # NewHandPD meander:
    python pd_loso_pipeline.py --dataset newhandpd --root /path/NewHandPD --task meander

    # Kaggle spiral (P1 only, by construction):
    python pd_loso_pipeline.py --dataset kaggle --root /path/kaggle --task spiral

Add --quick for a fast smoke test (few epochs, 3 folds). Results -> ./results/.
"""

from __future__ import annotations
import argparse
import itertools
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# --- heavy deps imported lazily so the file at least imports without a GPU env
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision
    from torchvision import transforms
    from PIL import Image
    _TORCH_OK = True
except Exception as _e:  # pragma: no cover
    _TORCH_OK = False
    _IMPORT_ERR = _e
    Dataset = object  # fallback so the module still imports (loader can be used/tested)

from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    LeaveOneGroupOut,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    confusion_matrix,
)

# Small subject counts make bootstrap/LOSO folds occasionally single-class; sklearn
# emits a warning per occurrence and floods the console. These are expected and handled
# (AUC guarded, confusion_matrix uses explicit labels), so silence just those messages.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found")


# ==============================================================================
# Config
# ==============================================================================
@dataclass
class Config:
    dataset: str
    root: str
    task: str = "spiral"          # spiral | meander | wave
    models: tuple = ("vgg16", "resnet50", "efficientnet_b0")
    img_size: int = 224
    batch_size: int = 8
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-4
    n_folds: int = 10
    patience: int = 8             # early stopping
    seed: int = 1234
    n_bootstrap: int = 2000
    quick: bool = False
    device: str = "cuda" if _TORCH_OK and torch.cuda.is_available() else "cpu"
    out_dir: str = "results"

    def finalize(self):
        if self.quick:
            self.epochs = 3
            self.n_folds = 3
            self.n_bootstrap = 200
        return self


# ==============================================================================
# Data loading  (the crux: recover the SUBJECT id so we can group by it)
# ==============================================================================
@dataclass
class Sample:
    path: str
    label: int              # 0 = healthy, 1 = PD
    subject: Optional[str]  # namespaced subject id, e.g. "PD:0092", "HC:V01"
    task: str
    provided_split: Optional[str] = None  # "training"/"testing" if the dataset ships one


def _is_junk(p: Path) -> bool:
    """Skip macOS resource forks / __MACOSX / hidden files that unzip leaves behind."""
    return p.name.startswith("._") or "__MACOSX" in p.parts or p.name == ".DS_Store"


def _label_from_path(p: Path) -> Optional[int]:
    """
    Infer label from the CLOSEST class-indicating folder to the file (deepest-first),
    so a dataset whose ROOT folder is named e.g. 'Parkinson_s_drawings' does not make
    every file look like PD. Folder tokens: control/healthy -> 0, patient/parkinson -> 1.
    """
    for seg in reversed([part.lower() for part in p.parts]):
        if "control" in seg or "healthy" in seg:
            return 0
        if "patient" in seg or "parkinson" in seg:
            return 1
    return None


# subject-id parsers, tried in order
_RE_KAGGLE = re.compile(r"(V\d+)", re.IGNORECASE)                 # V01HE01 -> V01
_RE_CLASSIC = re.compile(r"([HP])_?(\d+)", re.IGNORECASE)         # sp1-P3  -> P3
_RE_NUMERIC = re.compile(r"^(\d+)")                              # 0092-3  -> 0092


def _subject_from_name(stem: str) -> Optional[str]:
    m = _RE_KAGGLE.search(stem)
    if m:
        return m.group(1).upper()
    m = _RE_NUMERIC.match(stem)
    if m:
        return m.group(1)
    m = _RE_CLASSIC.search(stem)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2))}"
    return None


def discover_samples(root: str, task: str) -> list[Sample]:
    """
    Robust loader for all supported layouts:
      * HandPD/NewHandPD  '0092-3.jpg' or 'sp1-P3.jpg', label from folder or H/P token
      * Kaggle            'V01HE01.png', label from folder, subject V01 (namespaced)
    Subject ids are namespaced by label ('PD:'/'HC:') so a patient V01 and a healthy
    V01 (or numeric-id collisions across classes) never merge.
    Deduplicates the Kaggle nested 'drawings/' copy by basename.
    """
    # 1) gather every image file under root (minus macOS junk)
    all_imgs = [p for p in Path(root).rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not _is_junk(p)]
    if not all_imgs:
        top = [d.name for d in Path(root).iterdir()] if Path(root).exists() else []
        raise RuntimeError(
            f"No image files (.jpg/.png) found under:\n  {root}\n"
            f"Path exists: {Path(root).exists()}. Top-level entries: {top}\n"
            "Check the --root path (use quotes on Windows for paths with spaces).")

    # Does the folder tree label the task anywhere (Spiral_HandPD/, /spiral/, /wave/ ..)?
    tree_joined = " ".join("/".join(x.lower() for x in p.parts) for p in all_imgs)
    has_task_markers = any(t in tree_joined for t in ("spiral", "meander", "wave"))

    candidates = []
    for p in all_imgs:
        parts_low = [x.lower() for x in p.parts]
        # keep only files for the requested task IF the tree distinguishes tasks;
        # if it doesn't (flattened folder), treat everything as the requested task.
        if has_task_markers and task not in "/".join(parts_low):
            continue
        candidates.append((p, "drawings" in parts_low))

    if has_task_markers and not candidates:
        found = sorted({t for t in ("spiral", "meander", "wave") if t in tree_joined})
        raise RuntimeError(
            f"Found {len(all_imgs)} images but none for task='{task}'. "
            f"The tree only contains task(s): {found}. Re-run with --task {found[0]}.")
    if not has_task_markers:
        warnings.warn(f"No spiral/meander/wave folder markers under {root}; "
                      f"treating all {len(all_imgs)} images as task='{task}'.")

    # 2) dedupe by basename, preferring the non-'drawings' (top-level) copy
    by_name: dict[str, Path] = {}
    for p, in_mirror in sorted(candidates, key=lambda t: t[1]):  # False (top) first
        by_name.setdefault(p.name, p)

    samples: list[Sample] = []
    n_no_label = n_no_subject = 0
    for p in by_name.values():
        parts_low = [x.lower() for x in p.parts]
        label = _label_from_path(p)
        subj_raw = _subject_from_name(p.stem)
        # classic HandPD encodes label in the H/P token when folder is uninformative
        if label is None and subj_raw and subj_raw[0] in "HP" and subj_raw[1:].isdigit():
            label = 1 if subj_raw[0] == "P" else 0
        if label is None:
            n_no_label += 1
        if subj_raw is None:
            n_no_subject += 1
        if label is None or subj_raw is None:
            continue
        subject = f"{'PD' if label == 1 else 'HC'}:{subj_raw}"
        provided = ("training" if any("train" in s for s in parts_low)
                    else "testing" if any("test" in s for s in parts_low) else None)
        samples.append(Sample(str(p), label, subject, task, provided))

    if not samples:
        examples = [Path(p).name for p, _ in candidates[:6]]
        raise RuntimeError(
            f"Found {len(candidates)} {task} image(s) but could not parse them.\n"
            f"  - {n_no_label} had no PD/HC label: ensure images sit under folders whose "
            "names contain 'Control'/'Healthy' or 'Patients'/'Parkinson'.\n"
            f"  - {n_no_subject} had no subject id: expected names like '0092-3.jpg', "
            "'sp1-P3.jpg' or 'V01HE01.png'.\n"
            f"  Example filenames seen: {examples}")

    # per-subject image counts -> surface augmented/mis-parsed data immediately
    from collections import Counter
    ips = Counter(s.subject for s in samples)
    counts = sorted(ips.values())
    med = counts[len(counts) // 2]
    print(f"[load] {len(samples)} {task} images | "
          f"{len(ips)} subjects "
          f"({len({s.subject for s in samples if s.label==1})} PD / "
          f"{len({s.subject for s in samples if s.label==0})} HC) | "
          f"{sum(s.label for s in samples)} PD imgs / "
          f"{sum(1-s.label for s in samples)} HC imgs")
    print(f"[load] images/subject: min={counts[0]} median={med} max={counts[-1]} | "
          f"examples: {[Path(s.path).name for s in samples[:3]]}")
    if med > 8:
        warnings.warn(
            f"\n{'!'*70}\n[DATA SANITY] median images/subject = {med}. HandPD/NewHandPD "
            f"have 4 drawings/subject/task. A value this high usually means the folder is "
            f"AUGMENTED (each drawing copied many times) or the subject id is being "
            f"mis-parsed from the filenames (collapsing many people into few ids). Either "
            f"way, LOSO results from this folder are NOT trustworthy. Verify the filenames "
            f"before using these numbers.\n{'!'*70}")
    if len(ips) < 10:
        warnings.warn(f"[DATA SANITY] only {len(ips)} subjects found — too few for a "
                      f"credible subject-independent estimate; check parsing/provenance.")
    return samples


def report_provided_split_leakage(samples: list[Sample]) -> Optional[dict]:
    """
    For datasets that ship a train/test split (Kaggle), measure how many subjects
    appear in BOTH the provided training and testing folders. Any overlap means the
    widely-reported numbers on that split are subject-leaked -- a direct, dataset-level
    demonstration of the paper's thesis.
    """
    if not any(s.provided_split for s in samples):
        return None
    train_subj = {s.subject for s in samples if s.provided_split == "training"}
    test_subj = {s.subject for s in samples if s.provided_split == "testing"}
    overlap = train_subj & test_subj
    info = {
        "n_train_subjects": len(train_subj),
        "n_test_subjects": len(test_subj),
        "n_overlapping_subjects": len(overlap),
        "overlap_fraction_of_test": round(len(overlap) / max(len(test_subj), 1), 3),
        "overlapping_subjects": sorted(overlap),
    }
    print("\n" + "=" * 74 + "\n[PROVIDED-SPLIT LEAKAGE CHECK]")
    print(f"  provided TRAIN subjects: {info['n_train_subjects']}")
    print(f"  provided TEST  subjects: {info['n_test_subjects']}")
    print(f"  subjects in BOTH       : {info['n_overlapping_subjects']} "
          f"({info['overlap_fraction_of_test']*100:.0f}% of test subjects)")
    if overlap:
        print("  --> Provided split is SUBJECT-LEAKY; accuracy on it is inflated. "
              "Report LOSO instead.")
    print("=" * 74)
    return info


# ==============================================================================
# Torch dataset (augmentation lives here and is applied to TRAIN folds only)
# ==============================================================================
def make_transforms(img_size: int, train: bool):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(12),
            transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])


class DrawingDataset(Dataset):
    def __init__(self, samples: list[Sample], img_size: int, train: bool):
        self.samples = samples
        self.tf = make_transforms(img_size, train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        return self.tf(img), s.label


# ==============================================================================
# Models
# ==============================================================================
def build_model(name: str) -> "nn.Module":
    name = name.lower()
    if name == "vgg16":
        m = torchvision.models.vgg16(weights="IMAGENET1K_V1")
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2)
    elif name == "resnet50":
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        raise ValueError(f"unknown model {name}")
    return m


def train_one_fold(model, tr_ds, va_ds, cfg: Config, class_weights=None):
    dev = cfg.device
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    # class weighting helps when the TRAIN set is imbalanced (e.g. HandPD ~80% PD);
    # used in cross-dataset mode so the model does not collapse to the majority class.
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32, device=dev)
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        crit = nn.CrossEntropyLoss()
    tr = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    va = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(cfg.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        # validation AUC for early stopping
        probs, ys = _predict(model, va, dev)
        try:
            auc = roc_auc_score(ys, probs) if len(set(ys)) > 1 else 0.5
        except ValueError:
            auc = 0.5
        if auc > best_auc:
            best_auc, best_state, bad = auc, {k: v.cpu().clone()
                                              for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad() if _TORCH_OK else (lambda f: f)
def _predict(model, loader, dev):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(dev)
        p = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
        ys.extend(y.numpy().tolist())
    return np.array(probs), np.array(ys)


# ==============================================================================
# Metrics
# ==============================================================================
def compute_metrics(y_true, y_prob, thr=0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    # prevalence-aware and calibration metrics (reviewer-requested)
    prevalence = float(y_true.mean()) if len(y_true) else float("nan")
    try:
        pr_auc = (average_precision_score(y_true, y_prob)
                  if len(set(y_true)) > 1 else float("nan"))
    except ValueError:
        pr_auc = float("nan")
    try:
        brier = brier_score_loss(y_true, y_prob)
    except ValueError:
        brier = float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "prevalence": prevalence,          # no-skill baseline for pr_auc
        "brier": brier,                    # calibration: lower is better
        "ece": expected_calibration_error(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE: |observed rate - mean predicted prob| averaged over equal-width bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        m = (y_prob >= edges[i]) & (y_prob <= edges[i + 1]) if i == 0 else \
            (y_prob > edges[i]) & (y_prob <= edges[i + 1])
        if m.sum() == 0:
            continue
        err += m.sum() / len(y_true) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(err)


def subject_bootstrap_ci(rows: pd.DataFrame, metric: str,
                         n_boot: int, seed: int) -> tuple[float, float, float]:
    """Bootstrap over SUBJECTS (rows must have columns subject,y_true,y_prob)."""
    rng = np.random.default_rng(seed)
    subjects = rows["subject"].unique()
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        sub = pd.concat([rows[rows.subject == s] for s in pick])
        m = compute_metrics(sub["y_true"].values, sub["y_prob"].values)[metric]
        if not np.isnan(m):
            vals.append(m)
    if not vals:
        return (float("nan"),) * 3
    return (float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


# ==============================================================================
# Cross-validation runners
# ==============================================================================
def run_protocol(samples, cfg: Config, model_name: str, protocol: str) -> pd.DataFrame:
    """protocol in {'P1_image', 'P2_subject', 'LOSO'} -> per-image predictions."""
    y = np.array([s.label for s in samples])
    groups = np.array([s.subject for s in samples])

    if protocol == "P1_image":
        k = min(cfg.n_folds, int(np.bincount(y).min()))  # need >=1 of each class/fold
        if k < 2:
            raise RuntimeError(f"P1 needs >=2 samples of the minority class; got {k}.")
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
        folds = list(splitter.split(np.zeros(len(y)), y))
    elif protocol == "P2_subject":
        if any(g is None for g in groups):
            raise RuntimeError("P2 requires subject ids; none available.")
        n_groups = len(set(groups))
        # need enough groups of each class to stratify; cap folds to #groups
        per_class_groups = min(len({g for g, yy in zip(groups, y) if yy == c})
                               for c in (0, 1))
        k = min(cfg.n_folds, n_groups, max(per_class_groups, 1))
        if k < 2 or per_class_groups < 2:
            raise RuntimeError(
                f"P2 needs >=2 subjects per class; have {per_class_groups} "
                f"in the smaller class ({n_groups} subjects total). "
                "Use --loso, or this dataset is too small for subject-grouped CV.")
        if k < cfg.n_folds:
            print(f"  [P2] only {n_groups} subjects ({per_class_groups}/class); "
                  f"reducing folds {cfg.n_folds} -> {k}")
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                        random_state=cfg.seed)
        folds = list(splitter.split(np.zeros(len(y)), y, groups))
    elif protocol == "LOSO":
        if any(g is None for g in groups):
            raise RuntimeError("LOSO requires subject ids; none available.")
        folds = list(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups))
    else:
        raise ValueError(protocol)

    records = []
    for fold_id, (tr_idx, te_idx) in enumerate(folds):
        tr = [samples[i] for i in tr_idx]
        te = [samples[i] for i in te_idx]
        train_subjects = {s.subject for s in tr}
        model = build_model(model_name)
        model = train_one_fold(model,
                               DrawingDataset(tr, cfg.img_size, train=True),
                               DrawingDataset(te, cfg.img_size, train=False),
                               cfg)
        loader = DataLoader(DrawingDataset(te, cfg.img_size, train=False),
                            batch_size=cfg.batch_size, shuffle=False, num_workers=2)
        probs, ys = _predict(model, loader, cfg.device)
        for s, pr, yt in zip(te, probs, ys):
            records.append({
                "protocol": protocol, "model": model_name, "fold": fold_id,
                "subject": s.subject if s.subject else f"img{len(records)}",
                "y_true": int(yt), "y_prob": float(pr),
                # leakage probe flag: was this subject seen in training? (P1 only)
                "same_subject_in_train": bool(s.subject in train_subjects),
            })
        print(f"  [{protocol}/{model_name}] fold {fold_id} done "
              f"({len(te)} test imgs)")
    return pd.DataFrame.from_records(records)


def aggregate_per_subject(pred: pd.DataFrame) -> pd.DataFrame:
    """Mean predicted probability per subject -> clinical per-person operating point."""
    g = pred.groupby("subject").agg(y_true=("y_true", "max"),
                                    y_prob=("y_prob", "mean")).reset_index()
    return g


# ==============================================================================
# Cross-dataset external validation (train on cohort A, test on cohort B)
# ==============================================================================
def run_cross_dataset(train_samples, test_samples, cfg: Config,
                      model_name: str) -> pd.DataFrame:
    """
    Train on ALL of dataset A, evaluate on ALL of dataset B. Because A and B are
    different cohorts (different subjects, sites, devices), this is inherently both
    subject- and dataset-independent -- the hardest, most clinically realistic test
    of generalisation. A subject-grouped slice of the training data is held out for
    early stopping. Class weighting counters training-set imbalance (e.g. HandPD).
    Report ROC-AUC as primary: the decision threshold is not calibrated across cohorts
    with different class balance, so raw accuracy is not comparable.
    """
    y = np.array([s.label for s in train_samples])
    groups = np.array([s.subject for s in train_samples])

    # subject-grouped ~15% validation slice for early stopping
    per_class = min(len({g for g, yy in zip(groups, y) if yy == c}) for c in (0, 1))
    k = min(6, per_class)
    if k >= 2:
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
        tr_idx, va_idx = next(iter(sgkf.split(np.zeros(len(y)), y, groups)))
        tr = [train_samples[i] for i in tr_idx]
        va = [train_samples[i] for i in va_idx]
    else:  # too few subjects to hold out; validate on the training set itself
        tr, va = train_samples, train_samples

    # inverse-frequency class weights from the TRAIN split
    ytr = np.array([s.label for s in tr])
    counts = np.bincount(ytr, minlength=2).astype(float)
    weights = (counts.sum() / (2.0 * np.maximum(counts, 1))).tolist()

    model = build_model(model_name)
    model = train_one_fold(model,
                           DrawingDataset(tr, cfg.img_size, train=True),
                           DrawingDataset(va, cfg.img_size, train=False),
                           cfg, class_weights=weights)
    loader = DataLoader(DrawingDataset(test_samples, cfg.img_size, train=False),
                        batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    probs, ys = _predict(model, loader, cfg.device)
    records = [{"model": model_name, "subject": s.subject,
                "y_true": int(yt), "y_prob": float(pr)}
               for s, pr, yt in zip(test_samples, probs, ys)]
    print(f"  [cross/{model_name}] trained on {len(tr)} imgs "
          f"({len({s.subject for s in tr})} subj), tested on {len(test_samples)} imgs "
          f"({len({s.subject for s in test_samples})} subj)")
    return pd.DataFrame.from_records(records)


# ==============================================================================
# Leakage probe
# ==============================================================================
def leakage_probe(pred_p1: pd.DataFrame, samples, cfg: Config,
                  model_name: str) -> dict:
    """
    (1) P1 subject-leak fraction + (where a clean stratum exists) the accuracy gap
        between test drawings whose subject WAS vs WAS NOT in training.
    (2) label-permutation floor under P2, reported as BALANCED accuracy and AUC
        (both ~0.5 with no leakage). NB: raw accuracy floors at the majority-class
        rate under class imbalance, so it must NOT be used here.
    """
    out = {}
    # (1a) how leaky is the P1 image-level split? fraction of test drawings whose
    #      subject also appears in that fold's training set. On small datasets this
    #      is typically 1.0 (every subject leaks) -- itself direct evidence.
    out["p1_leak_fraction"] = float(pred_p1["same_subject_in_train"].mean())
    # (1b) seen-vs-unseen accuracy contrast, only if BOTH strata are non-empty
    seen = pred_p1[pred_p1.same_subject_in_train]
    unseen = pred_p1[~pred_p1.same_subject_in_train]
    out["acc_same_subject_seen"] = (
        accuracy_score(seen.y_true, (seen.y_prob >= .5).astype(int))
        if len(seen) else float("nan"))
    out["acc_subject_unseen"] = (
        accuracy_score(unseen.y_true, (unseen.y_prob >= .5).astype(int))
        if len(unseen) else float("nan"))
    if len(unseen) == 0:
        print("  [probe] note: P1 leaks 100% of subjects (no unseen stratum); "
              "seen-vs-unseen contrast is undefined -- see p1_leak_fraction=1.0.")

    # (2) permutation floor: shuffle labels across subjects, run P2, expect chance.
    if all(s.subject for s in samples):
        rng = np.random.default_rng(cfg.seed + 7)
        subj_labels = {s.subject: s.label for s in samples}
        shuffled = list(subj_labels.values())
        rng.shuffle(shuffled)
        perm_map = dict(zip(subj_labels.keys(), shuffled))
        perm_samples = [Sample(s.path, perm_map[s.subject], s.subject, s.task,
                               s.provided_split) for s in samples]
        perm_cfg = Config(**{**cfg.__dict__})
        perm_cfg.n_folds = min(3, cfg.n_folds)  # cheap
        try:
            pp = run_protocol(perm_samples, perm_cfg, model_name, "P2_subject")
            yp = (pp.y_prob >= .5).astype(int)
            # balanced accuracy & AUC are the correct chance-level references (~0.5)
            out["permutation_floor_balacc"] = float(balanced_accuracy_score(pp.y_true, yp))
            out["permutation_floor_auc"] = (
                float(roc_auc_score(pp.y_true, pp.y_prob))
                if pp.y_true.nunique() > 1 else float("nan"))
            out["permutation_floor_acc"] = float(accuracy_score(pp.y_true, yp))  # ref only
        except Exception as e:
            out["permutation_floor_balacc"] = float("nan")
            out["permutation_floor_auc"] = float("nan")
            out["permutation_floor_acc"] = float("nan")
            print(f"  [probe] permutation floor skipped: {e}")
    else:
        out["permutation_floor_balacc"] = float("nan")
        out["permutation_floor_auc"] = float("nan")
        out["permutation_floor_acc"] = float("nan")
    return out


# ==============================================================================
# Orchestration
# ==============================================================================
def summarise(pred: pd.DataFrame, cfg: Config, level: str) -> dict:
    """level in {'image','subject'}; returns metric means + subject bootstrap CIs."""
    if level == "subject":
        agg = aggregate_per_subject(pred)
        base = compute_metrics(agg.y_true.values, agg.y_prob.values)
        rows = agg.assign(subject=agg.subject)
    else:
        base = compute_metrics(pred.y_true.values, pred.y_prob.values)
        rows = pred[["subject", "y_true", "y_prob"]]
    ci = {}
    for m in ("accuracy", "roc_auc", "sensitivity", "specificity"):
        mean, lo, hi = subject_bootstrap_ci(rows, m, cfg.n_bootstrap, cfg.seed)
        ci[f"{m}_bootmean"] = mean
        ci[f"{m}_lo95"] = lo
        ci[f"{m}_hi95"] = hi
    return {**base, **ci}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=["handpd", "newhandpd", "kaggle"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--task", default="spiral",
                    choices=["spiral", "meander", "wave"])
    ap.add_argument("--models", nargs="+",
                    default=["vgg16", "resnet50", "efficientnet_b0"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--loso", action="store_true",
                    help="also run full leave-one-subject-out (slow)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1234],
                    help="random seeds to repeat P1/P2 over, e.g. --seeds 1 2 3 4 5. "
                         "Results are aggregated to mean +/- SD across seeds.")
    # cross-dataset external validation: train on --root/--task, test on these
    ap.add_argument("--test-root", default=None,
                    help="if set, run CROSS-DATASET mode: train on --root/--task, "
                         "test on --test-root/--test-task (external validation).")
    ap.add_argument("--test-task", default=None,
                    help="task for the external test set (defaults to --task).")
    ap.add_argument("--test-name", default="testset",
                    help="short label for the external test set, used in output names.")
    args = ap.parse_args()

    if not _TORCH_OK:
        sys.exit(f"[fatal] torch/torchvision/PIL not importable: {_IMPORT_ERR}\n"
                 "Install: pip install torch torchvision pillow")

    cfg = Config(dataset=args.dataset, root=args.root, task=args.task,
                 models=tuple(args.models), quick=args.quick).finalize()
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"[cfg] {cfg}")

    # 1) load (unified loader handles HandPD/NewHandPD/Kaggle filename formats)
    samples = discover_samples(cfg.root, cfg.task)
    have_subjects = all(s.subject for s in samples)

    # ---- CROSS-DATASET EXTERNAL VALIDATION MODE -------------------------------
    if args.test_root:
        test_task = args.test_task or args.task
        print(f"\n[cross-dataset] TRAIN = {args.dataset}:{args.task}  ->  "
              f"TEST = {args.test_name}:{test_task}")
        test_samples = discover_samples(args.test_root, test_task)
        train_tag = f"{args.dataset}_{args.task}"
        test_tag = f"{args.test_name}_{test_task}"
        cross_pred, cross_rows = [], []
        for seed in args.seeds:
            if _TORCH_OK:
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            np.random.seed(seed); cfg.seed = seed
            if len(args.seeds) > 1:
                print(f"\n{'#'*28} SEED {seed} {'#'*28}")
            for model_name in cfg.models:
                pred = run_cross_dataset(samples, test_samples, cfg, model_name)
                pred["train"] = train_tag; pred["test"] = test_tag; pred["seed"] = seed
                cross_pred.append(pred)
                for level in ("image", "subject"):
                    s = summarise(pred, cfg, level)
                    cross_rows.append({"train": train_tag, "test": test_tag,
                                       "model": model_name, "level": level,
                                       "seed": seed, **s})
        tag = f"{train_tag}_to_{test_tag}"
        pd.concat(cross_pred, ignore_index=True).to_csv(
            Path(cfg.out_dir) / f"predictions_cross_{tag}.csv", index=False)
        cs = pd.DataFrame(cross_rows)
        cs.to_csv(Path(cfg.out_dir) / f"summary_cross_{tag}.csv", index=False)
        # aggregate AUC/MCC across seeds
        met = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "mcc"]
        agg = (cs.groupby(["train", "test", "model", "level"])[met]
               .agg(["mean", "std"]))
        agg.columns = [f"{m}_{s}" for m, s in agg.columns]
        agg.reset_index().to_csv(
            Path(cfg.out_dir) / f"summary_cross_{tag}_agg.csv", index=False)
        print("\n" + "=" * 74 +
              f"\nCROSS-DATASET external validation (train {train_tag} -> test "
              f"{test_tag}), subject level, mean+/-SD over {len(args.seeds)} seed(s)"
              "\n(AUC is primary; thresholded metrics not calibrated across cohorts)\n"
              + "=" * 74)
        sub = cs[cs.level == "subject"]
        for model_name in cfg.models:
            v = sub[sub.model == model_name]
            if len(v):
                print(f"  {model_name:16s} AUC {v.roc_auc.mean():.3f}+/-{v.roc_auc.std():.3f} "
                      f"| bal_acc {v.balanced_accuracy.mean():.3f} "
                      f"| MCC {v.mcc.mean():.3f}")
        print(f"\n[done] wrote results/summary_cross_{tag}_agg.csv "
              f"and predictions_cross_{tag}.csv")
        return
    # ---- end cross-dataset mode ----------------------------------------------

    protocols = (["P1_image", "P2_subject"] + (["LOSO"] if args.loso else [])
                 if have_subjects else ["P1_image"])

    # 1b) if the dataset ships a train/test split (Kaggle), quantify its leakage
    all_pred, summary_rows = [], []
    split_leak = report_provided_split_leakage(samples)
    if split_leak is not None:
        summary_rows.append({"dataset": f"{args.dataset}:{args.task}",
                             "model": "-", "protocol": "provided_split_leakage",
                             "level": "subject",
                             **{k: v for k, v in split_leak.items()
                                if k != "overlapping_subjects"}})

    # 2) run each model x protocol, repeated over seeds for robustness
    for seed in args.seeds:
        if _TORCH_OK:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        cfg.seed = seed  # changes CV splits AND training stochasticity per seed
        if len(args.seeds) > 1:
            print(f"\n{'#'*30} SEED {seed} {'#'*30}")
        for model_name in cfg.models:
            pred_by_proto = {}
            for proto in protocols:
                print(f"\n=== seed {seed} | {model_name} | {proto} ===")
                pred = run_protocol(samples, cfg, model_name, proto)
                pred["dataset"] = f"{args.dataset}:{args.task}"
                pred["seed"] = seed
                pred_by_proto[proto] = pred
                all_pred.append(pred)
                for level in ("image", "subject"):
                    s = summarise(pred, cfg, level)
                    summary_rows.append({"dataset": f"{args.dataset}:{args.task}",
                                         "model": model_name, "protocol": proto,
                                         "level": level, "seed": seed, **s})
            # 3) leakage probe (needs P1 predictions + subject ids)
            if "P1_image" in pred_by_proto and have_subjects:
                probe = leakage_probe(pred_by_proto["P1_image"], samples, cfg, model_name)
                summary_rows.append({"dataset": f"{args.dataset}:{args.task}",
                                     "model": model_name, "protocol": "leakage_probe",
                                     "level": "image", "seed": seed, **probe})

    # 4) save per-seed predictions and summary
    preds = pd.concat(all_pred, ignore_index=True)
    tag = f"{args.dataset}_{args.task}"
    preds.to_csv(Path(cfg.out_dir) / f"predictions_{tag}.csv", index=False)
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(Path(cfg.out_dir) / f"summary_{tag}.csv", index=False)

    # 4b) aggregate metrics across seeds (mean +/- SD) -> the table to fill the paper from
    metric_cols = ["accuracy", "balanced_accuracy", "sensitivity", "specificity",
                   "f1", "roc_auc", "pr_auc", "prevalence", "brier", "ece", "mcc"]
    present = [c for c in metric_cols if c in summ.columns]
    cv = summ[summ.protocol.isin(["P1_image", "P2_subject", "LOSO"])]
    if len(cv):
        agg = (cv.groupby(["dataset", "model", "protocol", "level"])[present]
               .agg(["mean", "std"]))
        agg.columns = [f"{m}_{s}" for m, s in agg.columns]
        agg = agg.reset_index()
        agg.to_csv(Path(cfg.out_dir) / f"summary_{tag}_agg.csv", index=False)

    # 5) print the P1 vs P2 gap (mean +/- SD across seeds), subject level.
    n_seeds = len(args.seeds)
    print("\n" + "=" * 74 +
          f"\nP1 (image-level) vs P2 (subject-independent) -- subject level, "
          f"mean+/-SD over {n_seeds} seed(s)"
          "\n(raw accuracy is misleading under imbalance; watch bal_acc / AUC / MCC)\n"
          + "=" * 74)
    sub = summ[(summ.level == "subject") &
               (summ.protocol.isin(["P1_image", "P2_subject"]))]
    for model_name in cfg.models:
        def ms(proto, col):
            v = sub[(sub.model == model_name) & (sub.protocol == proto)][col]
            return (v.mean(), v.std()) if len(v) else (float("nan"), float("nan"))
        (a1, a1s), (a2, a2s) = ms("P1_image", "roc_auc"), ms("P2_subject", "roc_auc")
        (m1, _), (m2, _) = ms("P1_image", "mcc"), ms("P2_subject", "mcc")
        print(f"  {model_name:16s} "
              f"AUC {a1:.3f}+/-{a1s:.3f} -> {a2:.3f}+/-{a2s:.3f} (Δ{a1-a2:+.3f}) | "
              f"MCC {m1:.3f} -> {m2:.3f} (Δ{m1-m2:+.3f})")
    extra = f" and summary_{tag}_agg.csv" if n_seeds > 1 else ""
    print(f"\n[done] wrote results/summary_{tag}.csv{extra} and predictions_{tag}.csv")


if __name__ == "__main__":
    main()
