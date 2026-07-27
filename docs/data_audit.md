# Dataset audit and provenance

This file documents what was checked in each dataset and the decisions that follow from it.
The audit is itself part of the paper's contribution: several issues are only visible on
inspection of the real files, not from the assumed structure.

## HandPD (spiral, meander)
- 368 images each task, 92 subjects (74 PD / 18 HC), 4 drawings per subject.
- Subject identity is encoded in the filename (`0092-3.jpg` → subject `0092`, drawing 3);
  the label comes from the class folder. Subject IDs are namespaced by class so identifiers
  never merge across PD/HC.
- 80% PD prevalence: accuracy and sensitivity are near their majority ceilings, so ROC-AUC is
  the primary metric and balanced-class metrics are read from the Kaggle sets.

## Kaggle "Parkinson's Drawings" (spiral, wave)
- 102 images each task, 28 subjects (15 PD / 13 HC).
- Subject identity is recoverable from the filename (`V03PE05.png` → volunteer `V03`).
- **The provided train/test split is entirely subject-leaky**: every test subject also appears
  in training (25/25 spiral, 24/24 wave; `overlap_fraction_of_test = 1.0`). Any accuracy
  reported on this split is inflated by construction. Recorded in the `provided_split_leakage`
  rows of `results/summary_kaggle_spiral.csv` and `results/summary_kaggle_wave.csv`.
- The Kaggle download ships a duplicate nested mirror of the images and macOS junk files
  (`__MACOSX`, `._*`); the loader de-duplicates and skips these.

## NewHandPD — EXCLUDED
- The redistribution available to us parsed as **8 subjects × 30 images**, not the expected
  **66 subjects × 4 images**: the signature of an augmented-then-redistributed copy.
- Augmented-then-split data reintroduces exactly the leakage this study measures, and subject
  identifiers were not recoverable. Including it would undermine the paper's thesis, so it is
  excluded and the exclusion is stated in the manuscript's Limitations.

## General
- Augmentation is applied **inside the training fold only**, after splitting, to avoid a second
  leakage channel.
- A label-permutation control confirms the pipeline introduces no residual leakage
  (performance falls to chance under P2 when labels are shuffled across subjects).
