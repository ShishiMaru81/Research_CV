# Phase 1 multi-seed summary

Reconstructed from `frozen_results/` (seed 42) + campaign logs / `week11_light/`
(seeds 7, 2024). Registry confirms **90/90 complete** on Kaggle; metrics below
are recovered from logs + frozen tables.

**Rebuild:** `python scripts/rebuild_multiseed_summary.py`  
(after `python scripts/_extract_by_ckpt.py` if refreshing from the agent transcript)

## Coverage

| Block | Seed 42 | Seed 7 | Seed 2024 |
|---|---|---|---|
| Transfer baseline (18) | complete | complete | complete |
| Transfer strong-aug (18) | complete | complete | **6 / 18 metrics recovered** |
| LODO (9) | complete | complete | complete |

The 12 missing seed-2024 strong-aug metric rows were never pasted into chat and
are not in `week11_light/` (that zip only kept the last session’s partial
`crossdataset_matrix_aug.csv`). Later root-level downloads named
`crossdataset_matrix_aug.csv` / `generalization_gap_aug.csv` were checked and
are still only those **5 session-2 rows**, not the full 18. The runs themselves
finished (registry `status=complete`). To fill the gap: pull the full matrix
from a Kaggle Save Version, or re-eval those 12 checkpoints.

## Headline numbers (paper-facing)

| Quantity | Value | Notes |
|---|---|---|
| Mean across-seed std, baseline cross macro-F1 | **0.057** | all 18 cells × 3 seeds |
| Mean across-seed std, strong-aug cross macro-F1 | ~0.062 | biased; only 6/18 cells have seed 2024 |
| Mean paired aug delta (strong − baseline) | **+0.062** | n=42 seed-cells (incomplete 2024) |
| Cells with full 3-seed strong-aug | 6 / 18 | |

**Interpretation (workflow trap):** mean aug gain (+0.062) is **about the same
size as** mean across-seed std (±0.057). The augmentation story must be framed
as a modest, noisy improvement — not a decisive fix — until the 12 missing
2024 cells are filled and Phase 4 Wilcoxon is run.

## LODO (3 seeds × 3 models, models pooled)

| Held-out dataset | mean macro-F1 | std |
|---|---|---|
| riceleafbd | 0.464 | 0.067 |
| dhan_shomadhan | 0.380 | 0.066 |
| brri_rice_disease_pest | **0.230** | 0.062 |

BRRI remains the hardest holdout across seeds — consistent with the manuscript.

## Sanity: BRRI → RiceLeafBD baseline (seed sensitivity)

| Model | seed 7 | seed 42 | seed 2024 | mean ± std |
|---|---|---|---|---|
| MobileNetV2 | 0.438 | 0.573 | 0.598 | 0.536 ± 0.086 |
| EfficientNet-B0 | 0.443 | 0.424 | 0.655 | 0.507 ± 0.128 |
| ResNet50 | 0.442 | 0.300 | 0.287 | 0.343 ± 0.086 |

Seed-42 MobileNetV2 (0.573) is **not** an outlier high; seed 7 is the low
draw. Treat single-seed anecdotes carefully in Discussion.

## Mean cross macro-F1 by model (all available cells)

| Augmentation | MobileNetV2 | EfficientNet-B0 | ResNet50 |
|---|---|---|---|
| default | 0.434 | 0.418 | 0.483 |
| strong | 0.462 | 0.457 | **0.565** |

ResNet50 still leads under strong aug in the recovered data.

## Outputs in `results/multiseed/`

- `transfer_all_seeds.csv` / `lodo_all_seeds.csv`
- `transfer_cell_mean_std.csv`
- `aug_delta_per_cell.csv` / `aug_delta_paired.csv`
- `lodo_cell_mean_std.csv` / `lodo_by_holdout_mean_std.csv`
- `PHASE1_MULTISEED_SUMMARY.md` (this file)

## Next steps

1. **Fill 12 missing strong-2024 metrics** (Kaggle re-download or re-eval).
2. Phase 2: AdaBN + confound extension (needs checkpoints — seed-42 frozen
   still available; multi-seed AdaBN needs re-pulled or re-trained weights).
3. Phase 4 stats once strong-aug matrix is complete: Wilcoxon on paired aug
   deltas; bootstrap CIs from prediction CSVs.
