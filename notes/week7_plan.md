# Week 7 — Mitigation, part 1

Goal: test whether simple, standard interventions close the generalization gap
measured in Week 5. Change one variable at a time.

## Experiment 1 — Strong augmentation

Aggressive train-only albumentations pipeline (`strong_train_transform` in
`src/data_loader.py`):

- RandomResizedCrop (scale 0.6-1.0) — background randomization proxy
- HorizontalFlip, Affine (shift/scale/rotate)
- RandomBrightnessContrast, HueSaturationValue — reduce dataset color signature
- GaussianBlur
- CoarseDropout (random_uniform fill) — force reliance on multiple regions

Re-run the six Week 5 ordered transfer pairs with this pipeline on the TRAIN
loader only; eval stays deterministic.

- Output: `results/crossdataset_matrix_aug.csv`,
  `results/generalization_gap_aug.csv`
- Checkpoints/eval artifacts get an `__aug-strong` tag so they never collide
  with the Week 5 baseline.

## Experiment 2 — Leave-one-dataset-out (LODO)

For each held-out dataset, train on the other two combined and evaluate on the
held-out third. Label space per held-out is the largest non-degenerate shared
set (class present in the held-out AND in at least one training dataset):

| Held out        | Train on              | Classes                              |
|-----------------|-----------------------|--------------------------------------|
| RiceLeafBD      | Dhan + BRRI           | brown_spot, healthy, tungro          |
| Dhan-Shomadhan  | RiceLeafBD + BRRI     | brown_spot, rice_blast, scald, tungro|
| BRRI            | RiceLeafBD + Dhan     | healthy, rice_blast, scald, tungro   |

Classes unique to the held-out set (e.g. bacterial_leaf_blight for RiceLeafBD)
cannot be learned from the training sources and are excluded by design. LODO
uses default augmentation to keep it a single-variable experiment.

- Output: `results/lodo_results.csv`

## Comparison

`run_mitigation.py` aggregates:

- `results/mitigation_pairwise_aug.csv` — per pair: baseline vs augmentation
  cross macro-F1, cross-F1 improvement, and generalization-gap reduction.
- `results/mitigation_comparison.csv` — by target dataset: single-source
  baseline vs single-source augmentation vs LODO macro-F1.

Note: LODO and single-source use different class sets (LODO is limited by the
3-way overlap), so the comparison is strategy-level, not matched-class. This
is documented in the paper.

## Definition of done

- `crossdataset_matrix_aug.csv` — augmented cross-dataset results.
- `lodo_results.csv` — leave-one-dataset-out results.
- `mitigation_comparison.csv` — baseline vs augmentation vs LODO.
- Written observation of which mitigation worked best in
  `notes/week7_retro.md` (fill after the Kaggle run).
