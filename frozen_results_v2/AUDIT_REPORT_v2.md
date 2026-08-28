# Freeze v2 audit

**Status: PASS_WITH_WARNINGS**

- Git commit at freeze: `2d9f08bc5554d5ab9e4fc59c3b8ea3b34ccc9ec3`
- v1 freeze manifest commit hint: `ac77dc2c5e4b3b8239de595d48f179f67a783542`
- Checks: 36
- Warnings: 1

## Checks

- manifest.csv: file-copy integrity vs v1 (columns/rows/numeric values match; this is copy verification, not independent numerical reproduction)
- manifest.csv: SHA-256 identical to v1 (3a1a981ae73c…)
- indataset_results.csv: file-copy integrity vs v1 (columns/rows/numeric values match; this is copy verification, not independent numerical reproduction)
- indataset_results.csv: SHA-256 identical to v1 (5c152652926c…)
- crossdataset_matrix.csv: file-copy integrity vs v1 (columns/rows/numeric values match; this is copy verification, not independent numerical reproduction)
- crossdataset_matrix.csv: SHA-256 identical to v1 (91496ba84253…)
- generalization_gap.csv: file-copy integrity vs v1 (columns/rows/numeric values match; this is copy verification, not independent numerical reproduction)
- generalization_gap.csv: SHA-256 identical to v1 (f48ea89ef746…)
- background_confound.csv: file-copy integrity vs v1 (columns/rows/numeric values match; this is copy verification, not independent numerical reproduction)
- background_confound.csv: SHA-256 identical to v1 (573ed21fa74b…)
- gradcam_records.csv: file-copy integrity vs v1 (columns/rows/numeric values match; this is copy verification, not independent numerical reproduction)
- gradcam_records.csv: SHA-256 identical to v1 (e96f34b27841…)
- crossdataset_matrix_aug.csv: file-copy integrity vs v1 (columns/rows/numeric values match; this is copy verification, not independent numerical reproduction)
- crossdataset_matrix_aug.csv: SHA-256 identical to v1 (f653bee2ec0a…)
- generalization_gap_aug.csv: file-copy integrity vs v1 (columns/rows/numeric values match; this is copy verification, not independent numerical reproduction)
- generalization_gap_aug.csv: SHA-256 identical to v1 (af560ad997e7…)
- lodo_results.csv: file-copy integrity vs v1 (columns/rows/numeric values match; this is copy verification, not independent numerical reproduction)
- lodo_results.csv: SHA-256 identical to v1 (61d586c50399…)
- mitigation_pairwise_aug.csv: file-copy integrity vs v1 (columns/rows/numeric values match; this is copy verification, not independent numerical reproduction)
- mitigation_pairwise_aug.csv: SHA-256 identical to v1 (16253fc4c6a4…)
- mitigation_comparison.csv: file-copy integrity vs v1 (columns/rows/numeric values match; this is copy verification, not independent numerical reproduction)
- mitigation_comparison.csv: SHA-256 identical to v1 (5875ffccf257…)
- transfer_all_seeds.csv: copied from revision overlay (102 rows)
- transfer_cell_mean_std.csv: copied from revision overlay (42 rows)
- lodo_all_seeds.csv: copied from revision overlay (27 rows)
- adabn_results.csv: copied from revision overlay (18 rows)
- augmentation_ablation.csv: copied from revision overlay (18 rows)
- stats_tests.csv: copied from revision overlay (4 rows)
- seed_variance.csv: copied from revision overlay (42 rows)
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
- v1-core checks verify **file-copy / hash integrity** after `shutil.copy2`, not independent re-derivation from metrics JSON.
- For numerical checks that can fail, run `python scripts/numerical_freeze_audit.py`.
- See `notes/freeze_v2_changelog.md` for what changed and why.
