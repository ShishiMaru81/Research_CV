# Week 10-14 Audit Report

Frozen at: `2026-09-04T13:09:55.748272+00:00`

## v1 Integrity

All 11 CSVs in `frozen_results/` match the Week-8 SHA-256 manifest: **PASS**

## v2 Artifacts

- `adabn_labelshift.csv` (18 rows x 16 columns): AUDITED
- `adabn_results.csv` (18 rows x 18 columns): AUDITED
- `augmentation_ablation.csv` (18 rows x 12 columns): HASHED
- `background_confound.csv` (3 rows x 9 columns): AUDITED
- `bootstrap_ci.csv` (19 rows x 6 columns): HASHED
- `crossdataset_matrix.csv` (18 rows x 9 columns): AUDITED
- `crossdataset_matrix_aug.csv` (18 rows x 10 columns): AUDITED
- `crossdataset_matrix_masked_hsv_leaf.csv` (18 rows x 10 columns): AUDITED
- `crossdataset_matrix_masked_sam_leaf.csv` (18 rows x 10 columns): AUDITED
- `deployment_profile.csv` (3 rows x 7 columns): AUDITED
- `dinov2_crossdataset.csv` (18 rows x 10 columns): AUDITED
- `dinov2_indataset.csv` (9 rows x 8 columns): AUDITED
- `generalization_gap.csv` (18 rows x 13 columns): AUDITED
- `generalization_gap_aug.csv` (18 rows x 14 columns): AUDITED
- `gradcam_negative_summary.csv` (19 rows x 2 columns): AUDITED
- `gradcam_records.csv` (12 rows x 11 columns): AUDITED
- `hsv_mask_quality.csv` (5419 rows x 5 columns): AUDITED
- `indataset_results.csv` (9 rows x 7 columns): AUDITED
- `lodo_all_seeds.csv` (27 rows x 5 columns): HASHED
- `lodo_results.csv` (9 rows x 11 columns): AUDITED
- `manifest.csv` (5419 rows x 7 columns): AUDITED
- `mitigation_comparison.csv` (9 rows x 7 columns): AUDITED
- `mitigation_pairwise_aug.csv` (18 rows x 11 columns): AUDITED
- `sam_mask_quality.csv` (5419 rows x 6 columns): AUDITED
- `seed_variance.csv` (42 rows x 9 columns): HASHED
- `stats_tests.csv` (4 rows x 8 columns): HASHED
- `transfer_all_seeds.csv` (102 rows x 8 columns): HASHED
- `transfer_cell_mean_std.csv` (42 rows x 9 columns): HASHED

### Auxiliary non-CSV artifacts

- `deployment_env.json` (286 bytes): HASHED

## v2 Completeness

- `sam_leaf`: 18/18 exact pair/model/seed/class keys: PASS
- `hsv_leaf`: 18/18 exact pair/model/seed/class keys: PASS
- All present masked conditions: 36/36 keys: PASS

## Consistency Checks

- v1_integrity: PASS
- v2_arithmetic: PASS
- v2_completeness: PASS
- background_confound.csv: byte-identical to verified v1 core: PASS
- crossdataset_matrix.csv: byte-identical to verified v1 core: PASS
- crossdataset_matrix_aug.csv: byte-identical to verified v1 core: PASS
- generalization_gap.csv: byte-identical to verified v1 core: PASS
- generalization_gap_aug.csv: byte-identical to verified v1 core: PASS
- gradcam_records.csv: byte-identical to verified v1 core: PASS
- indataset_results.csv: byte-identical to verified v1 core: PASS
- lodo_results.csv: byte-identical to verified v1 core: PASS
- manifest.csv: byte-identical to verified v1 core: PASS
- mitigation_comparison.csv: byte-identical to verified v1 core: PASS
- mitigation_pairwise_aug.csv: byte-identical to verified v1 core: PASS
- deployment_profile.csv: schema, model coverage, and resource metrics are valid: PASS
- gradcam_negative_summary.csv: 19 metrics and internal count/fraction arithmetic are valid: PASS
- dinov2_indataset.csv: metric ranges, sample counts, and unique keys pass: PASS
- dinov2_crossdataset.csv: metric ranges, sample counts, and unique keys pass: PASS
- DINOv2 pair/dataset coverage is complete for every saved seed: PASS
- adabn_labelshift.csv: divergence metrics and BN counts are valid for all 18 rows: PASS
- adabn_labelshift.csv: experiment keys and delta values match the frozen AdaBN results: PASS
- sam_mask_quality.csv: 5419 unique image paths match the v1 manifest: PASS
- sam_mask_quality.csv: mask fractions and component counts are valid: PASS
- sam_leaf: mask-quality artifact is present: PASS
- hsv_mask_quality.csv: 5419 unique image paths match the v1 manifest: PASS
- hsv_mask_quality.csv: mask fractions and component counts are valid: PASS
- hsv_leaf: mask-quality artifact is present: PASS
- crossdataset_matrix_masked_sam_leaf.csv: 18/18 exact masked-transfer keys: PASS
- crossdataset_matrix_masked_hsv_leaf.csv: 18/18 exact masked-transfer keys: PASS
- All specified arithmetic checks passed without filtering rows: PASS

## Interpretation Guardrails

- The audit verifies saved-file integrity, arithmetic ranges, and key coverage; it is not an independent model re-run.
- Presence of a mask artifact does not by itself reconstruct a missing human audit verdict.
- The AdaBN label-shift analysis is observational and does not establish causality.
- Human mask-audit gate: **verified** (Human audit decision explicitly clears sam_leaf and hsv_leaf for Week 13.)

## Decision

All implemented numerical and integrity checks pass. The saved v2 artifacts are internally consistent, and the human mask-audit gate is verified by the supplied decision record.
