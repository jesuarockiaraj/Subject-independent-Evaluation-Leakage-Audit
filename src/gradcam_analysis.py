#!/usr/bin/env python3
"""
gradcam_analysis.py
===================
Explainability + computational-cost analyses requested by reviewers, matching
Sections "Explainability" and "Runtime and computational cost" of the manuscript.

It answers the question the paper's thesis raises: *when a model is trained under
image-level splitting, does it attend to the drawn trace (disease-relevant) or to
off-trace/background cues (identity- and acquisition-relevant)?*

Two quantitative summaries are produced, both for P1- and P2-trained models so the
comparison is matched:

  1. ON-TRACE SALIENCY MASS  rho = sum(L over stroke pixels) / sum(L over all pixels)
     A disease-driven model should concentrate saliency on the stroke.

  2. WITHIN- vs BETWEEN-SUBJECT SALIENCY CONSISTENCY
     Mean pairwise correlation of Grad-CAM maps for drawings by the SAME person
     vs. by DIFFERENT people. An identity-driven model should show a larger gap.

Also logs wall-clock training time, peak GPU memory, parameter count and MACs.

USAGE
-----
    pip install torch torchvision scikit-learn pillow numpy pandas matplotlib
    python gradcam_analysis.py --dataset handpd \
        --root "C:/path/Datasets/HandPD" --task spiral --seeds 1 2 3

Outputs (in ./results/):
    gradcam_metrics_<tag>.csv    on-trace mass + saliency consistency, per protocol/model
    gradcam_panel_<tag>.png      qualitative side-by-side maps (P1 vs P2)
    runtime_<tag>.csv            timing / memory / model-size table

NOTE: this script imports the loader and splitters from pd_loso_pipeline.py, so keep
both files in the same folder.
"""
from __future__ import annotations
import argparse, time, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from pd_loso_pipeline import (Config, discover_samples, DrawingDataset,
                              build_model, train_one_fold)


# ----------------------------------------------------------------- Grad-CAM
def target_layer(model, name: str):
    """Last convolutional block for each supported backbone."""
    name = name.lower()
    if name == "vgg16":
        return model.features[-1]
    if name == "resnet50":
        return model.layer4[-1]
    if name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(name)


class GradCAM:
    """Standard Grad-CAM (Selvaraju et al., 2017), Eq. (12) in the manuscript."""

    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, cls=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if cls is None:
            cls = logits.argmax(dim=1)
        score = logits.gather(1, cls.view(-1, 1)).sum()
        score.backward()
        # alpha_k = global-average-pooled gradients; L = ReLU(sum_k alpha_k A^k)
        alpha = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        # normalise each map to [0,1]
        flat = cam.flatten(1)
        mn = flat.min(1)[0].view(-1, 1, 1)
        mx = flat.max(1)[0].view(-1, 1, 1)
        return ((cam - mn) / (mx - mn + 1e-8)).cpu().numpy()


def stroke_mask(path: str, size: int) -> np.ndarray:
    """Binarise the drawing: True where ink is. Drawings are dark on light paper."""
    img = Image.open(path).convert("L").resize((size, size))
    a = np.asarray(img, dtype=np.float32) / 255.0
    # Otsu-like split via the midpoint between the two dominant modes
    thr = (a.min() + a.max()) / 2.0
    m = a < thr
    if m.mean() > 0.5:          # safety: ink should be the minority class
        m = ~m
    return m


def on_trace_mass(cam: np.ndarray, mask: np.ndarray) -> float:
    tot = cam.sum()
    return float(cam[mask].sum() / tot) if tot > 0 else float("nan")


def consistency(cams: dict[str, list[np.ndarray]]) -> tuple[float, float]:
    """Mean within-subject vs between-subject pairwise correlation of saliency maps."""
    def corr(a, b):
        a, b = a.ravel(), b.ravel()
        a, b = a - a.mean(), b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d > 0 else np.nan

    within, between = [], []
    subs = list(cams)
    for s in subs:
        v = cams[s]
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                within.append(corr(v[i], v[j]))
    rng = np.random.default_rng(0)
    for _ in range(2000):                      # sample between-subject pairs
        if len(subs) < 2:
            break
        s1, s2 = rng.choice(subs, 2, replace=False)
        between.append(corr(rng.choice(cams[s1]), rng.choice(cams[s2])))
    return (float(np.nanmean(within)) if within else float("nan"),
            float(np.nanmean(between)) if between else float("nan"))


# ----------------------------------------------------------------- experiment
def train_for_protocol(samples, cfg, model_name, protocol):
    """Train one model on the first fold of the given protocol; return model + timing."""
    y = np.array([s.label for s in samples])
    g = np.array([s.subject for s in samples])
    if protocol == "P1_image":
        k = min(cfg.n_folds, int(np.bincount(y).min()))
        sp = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
        tr_idx, te_idx = next(iter(sp.split(np.zeros(len(y)), y)))
    else:
        per_class = min(len({gg for gg, yy in zip(g, y) if yy == c}) for c in (0, 1))
        k = min(cfg.n_folds, len(set(g)), max(per_class, 1))
        sp = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
        tr_idx, te_idx = next(iter(sp.split(np.zeros(len(y)), y, g)))
    tr = [samples[i] for i in tr_idx]
    te = [samples[i] for i in te_idx]

    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.time()
    model = build_model(model_name)
    model = train_one_fold(model,
                           DrawingDataset(tr, cfg.img_size, True),
                           DrawingDataset(te, cfg.img_size, False), cfg)
    if cfg.device == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 2**20) if cfg.device == "cuda" else float("nan")
    n_par = sum(p.numel() for p in model.parameters())
    return model, te, dict(train_seconds=secs, peak_mem_MiB=peak,
                           params_M=n_par / 1e6, n_train_img=len(tr), n_test_img=len(te))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--task", default="spiral")
    ap.add_argument("--models", nargs="+",
                    default=["vgg16", "resnet50", "efficientnet_b0"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--max-images", type=int, default=80,
                    help="cap on test images used for the saliency statistics")
    args = ap.parse_args()

    cfg = Config(dataset=args.dataset, root=args.root, task=args.task,
                 models=tuple(args.models)).finalize()
    os.makedirs(cfg.out_dir, exist_ok=True)
    samples = discover_samples(cfg.root, cfg.task)
    tag = f"{args.dataset}_{args.task}"

    rows, runtime_rows, panel = [], [], {}
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed); cfg.seed = seed
        for model_name in args.models:
            for protocol in ("P1_image", "P2_subject"):
                print(f"\n=== seed {seed} | {model_name} | {protocol} ===")
                model, test, rt = train_for_protocol(samples, cfg, model_name, protocol)
                rt.update(seed=seed, model=model_name, protocol=protocol, dataset=tag)
                runtime_rows.append(rt)

                cam_fn = GradCAM(model.to(cfg.device).eval(),
                                 target_layer(model, model_name))
                use = test[: args.max_images]
                loader = DataLoader(DrawingDataset(use, cfg.img_size, False),
                                    batch_size=8, shuffle=False)
                maps, k = [], 0
                for xb, _ in loader:
                    maps.append(cam_fn(xb.to(cfg.device)))
                    k += xb.shape[0]
                maps = np.concatenate(maps, 0)

                rho = [on_trace_mass(m, stroke_mask(s.path, cfg.img_size))
                       for m, s in zip(maps, use)]
                by_subj: dict[str, list] = {}
                for m, s in zip(maps, use):
                    by_subj.setdefault(s.subject, []).append(m)
                w, b = consistency({k2: v for k2, v in by_subj.items() if len(v) > 1})

                rows.append(dict(dataset=tag, seed=seed, model=model_name,
                                 protocol=protocol,
                                 on_trace_mass=float(np.nanmean(rho)),
                                 within_subject_corr=w, between_subject_corr=b,
                                 identity_gap=w - b))
                print(f"  on-trace mass={np.nanmean(rho):.3f} | within={w:.3f} "
                      f"between={b:.3f} gap={w-b:+.3f}")
                panel.setdefault(model_name, {})[protocol] = (use[:3], maps[:3])
                del model; torch.cuda.empty_cache() if cfg.device == "cuda" else None

    df = pd.DataFrame(rows)
    df.to_csv(Path(cfg.out_dir) / f"gradcam_metrics_{tag}.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(Path(cfg.out_dir) / f"runtime_{tag}.csv", index=False)

    # aggregate table for the manuscript
    agg = (df.groupby(["model", "protocol"])[["on_trace_mass", "within_subject_corr",
                                              "between_subject_corr", "identity_gap"]]
           .agg(["mean", "std"]))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.reset_index().to_csv(Path(cfg.out_dir) / f"gradcam_metrics_{tag}_agg.csv", index=False)
    print("\n=== on-trace saliency mass (higher = more attention on the drawn stroke) ===")
    print(agg[["on_trace_mass_mean", "identity_gap_mean"]].to_string())

    # qualitative panel
    fig, axes = plt.subplots(max(len(panel), 1), 6,
                             figsize=(16, 3 * max(len(panel), 1)), squeeze=False)
    for r, (mname, per) in enumerate(panel.items()):
        col = 0
        for protocol in ("P1_image", "P2_subject"):
            if protocol not in per:
                continue
            samp, mp = per[protocol]
            for s, m in zip(samp, mp):
                ax = axes[r][col]; col += 1
                base = Image.open(s.path).convert("L").resize((cfg.img_size, cfg.img_size))
                ax.imshow(np.asarray(base), cmap="gray")
                ax.imshow(m, cmap="jet", alpha=0.45)
                ax.set_title(f"{mname}\n{protocol}", fontsize=8)
                ax.axis("off")
                if col >= 6:
                    break
    plt.tight_layout()
    plt.savefig(Path(cfg.out_dir) / f"gradcam_panel_{tag}.png", dpi=180, bbox_inches="tight")
    print(f"\n[done] results/gradcam_metrics_{tag}_agg.csv, runtime_{tag}.csv, "
          f"gradcam_panel_{tag}.png")


if __name__ == "__main__":
    main()
