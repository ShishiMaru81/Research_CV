# Phase 1 multi-seed summary

Reconstructed from `frozen_results/` (seed 42) + campaign logs / `week11_light/` (seeds 7, 2024).

## Coverage

Transfer cells by seed × augmentation:

```
seed  augmentation
7     default         18
      strong          18
42    bucket-geo       6
      default         18
      strong          18
2024  default         18
      strong           6
```

LODO by seed:

```
seed
7       9
42      9
2024    9
```

Missing transfer seed-cells: **12** (see script output).  
If non-zero, prefer re-downloading `results/multiseed/*.csv` from Kaggle.

## Headline numbers (paper-facing)

| Quantity | Value |
|---|---|
| Mean across-seed std (baseline cross macro-F1) | 0.0667 |
| Mean across-seed std (strong-aug cross macro-F1) | 0.0622 |
| Mean paired aug delta (strong − baseline) | 0.0670 |
| Paired seed-cells used for delta | 42 |

Interpretation cue from the workflow: compare mean aug gain (0.067) to baseline seed noise (0.067).

## LODO (models pooled)

```
      held_out_dataset  f1_mean   f1_std  n
brri_rice_disease_pest 0.229623 0.061997  9
        dhan_shomadhan 0.380456 0.066400  9
            riceleafbd 0.463505 0.067244  9
```

## Outputs

- `transfer_all_seeds.csv`
- `lodo_all_seeds.csv`
- `transfer_cell_mean_std.csv`
- `aug_delta_per_cell.csv` / `aug_delta_paired.csv`
- `lodo_cell_mean_std.csv` / `lodo_by_holdout_mean_std.csv`
