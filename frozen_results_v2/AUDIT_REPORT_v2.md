# Freeze v2 audit

**Status: PASS_WITH_WARNINGS**

- Git commit at freeze: `1e30488623c936b42bb5d2072203c45f67007270`
- v1 freeze manifest commit hint: `88f8c5d5d4ce8ef08fb84ff14b7d0fac696b5fb3`
- Checks: 25
- Warnings: 1

## Checks

- manifest.csv: v1 seed-42 values reproduced within tolerance
- indataset_results.csv: v1 seed-42 values reproduced within tolerance
- crossdataset_matrix.csv: v1 seed-42 values reproduced within tolerance
- generalization_gap.csv: v1 seed-42 values reproduced within tolerance
- background_confound.csv: v1 seed-42 values reproduced within tolerance
- gradcam_records.csv: v1 seed-42 values reproduced within tolerance
- crossdataset_matrix_aug.csv: v1 seed-42 values reproduced within tolerance
- generalization_gap_aug.csv: v1 seed-42 values reproduced within tolerance
- lodo_results.csv: v1 seed-42 values reproduced within tolerance
- mitigation_pairwise_aug.csv: v1 seed-42 values reproduced within tolerance
- mitigation_comparison.csv: v1 seed-42 values reproduced within tolerance
- transfer_all_seeds.csv: copied from revision overlay (96 rows)
- transfer_cell_mean_std.csv: copied from revision overlay (36 rows)
- lodo_all_seeds.csv: copied from revision overlay (27 rows)
- adabn_results.csv: copied from revision overlay (18 rows)
- augmentation_ablation.csv: copied from revision overlay (18 rows)
- stats_tests.csv: copied from revision overlay (4 rows)
- seed_variance.csv: copied from revision overlay (36 rows)
- bootstrap_ci.csv: copied from revision overlay (19 rows)
- seed coverage OK: seed=42 aug=default = 18
- seed coverage OK: seed=42 aug=strong = 18
- seed coverage OK: seed=7 aug=default = 18
- seed coverage OK: seed=7 aug=strong = 18
- seed coverage OK: seed=2024 aug=default = 18
- ablation rows: 18 (>=18)

## Warnings

- seed coverage incomplete: seed=2024 aug=strong has 6, expected 18

## Policy

- `frozen_results/` remains immutable.
- This directory is the replacement freeze for revision claims.
- See `notes/freeze_v2_changelog.md` for what changed and why.
