# Manuscript and report audit

Generated from repository CSVs. Tolerance ±0.002 on rounded display values.

**Result: 22/22 checks PASS**

## Numeric claims vs sources

| Claim | Expected | Actual | Source | Status |
|-------|----------|--------|--------|--------|
| In-dataset mean macro-F1 | 0.719 | 0.7192 | `frozen_results/indataset_results.csv` | PASS |
| Baseline cross macro-F1 (seed 42) | 0.436 | 0.4364 | `frozen crossdataset_matrix` | PASS |
| Mean gap (seed 42) | 0.387 | 0.3872 | `frozen generalization_gap` | PASS |
| Strong-aug cross macro-F1 (seed 42) | 0.503 | 0.5033 | `mitigation_pairwise_aug` | PASS |
| Mean aug improvement (seed 42) | 0.067 | 0.0669 | `mitigation_pairwise_aug` | PASS |
| Aug pairs improved (seed 42) | 14.000 | 14.0000 | `mitigation_pairwise_aug` | PASS |
| Baseline cross mean (3-seed cells) | 0.441 | 0.4408 | `transfer_cell_mean_std default` | PASS |
| Baseline cross std avg | 0.067 | 0.0667 | `transfer_cell_mean_std default` | PASS |
| Strong cross mean (partial seeds) | 0.502 | 0.5021 | `transfer_cell_mean_std strong` | PASS |
| Wilcoxon aug p (18 cells) | 0.000 | 0.0002 | `stats_tests.csv` | PASS |
| Mean paired aug delta | 0.070 | 0.0701 | `stats_tests.csv` | PASS |
| AdaBN mean delta | -0.055 | -0.0553 | `adabn_results.csv` | PASS |
| AdaBN pairs improved | 5.000 | 5.0000 | `adabn_results.csv` | PASS |
| AdaBN Wilcoxon p | 0.099 | 0.0987 | `stats_tests.csv` | PASS |
| LODO positive cells | 3.000 | 3.0000 | `mitigation_comparison.csv` | PASS |
| White-bg confound F1 | 0.854 | 0.8535 | `background_confound.csv` | PASS |
| Field-bg confound F1 | 0.705 | 0.7054 | `background_confound.csv` | PASS |
| Cross confound F1 | 0.573 | 0.5725 | `background_confound.csv` | PASS |
| Ablation geometric mean F1 | 0.567 | 0.5669 | `augmentation_ablation.csv` | PASS |
| Ablation geometric delta | 0.085 | 0.0851 | `augmentation_ablation.csv` | PASS |
| ResNet50 baseline 6-pair mean | 0.482 | 0.4819 | `crossdataset_matrix resnet50` | PASS |
| ResNet50 strong 6-pair mean | 0.609 | 0.6092 | `crossdataset_matrix_aug resnet50` | PASS |

## Files audited

- Manuscript: `D:/Research_Own/Research_CV/paper/manuscript.md`
- Progress report: `D:/Research_Own/Research_CV/notes/progress_report.md`

## Known gaps (not failures)

- **12 / 18** strong-augmentation cells lack seed 2024 in `transfer_all_seeds.csv`.
- AdaBN evaluated at seed 42 only (by design).
- Bucket ablation: ResNet50 × seed 42 only.
- Grad-CAM overlay PNG not bundled locally.

## Re-run

```bash
python scripts/rebuild_multiseed_summary.py
python scripts/build_multiseed_tables.py --sync-week11
python scripts/audit_writing.py
python -m freeze_results_v2
```

