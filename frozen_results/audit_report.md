# Week 8 result-freeze audit

**Status: PASS**

- Git commit at freeze: `13b855282369347a2f313e54ede198ce92d7b371`
- Canonical manifest SHA-256: `3a1a981ae73cded5b7dc46f6a3e479594c6d7a71af78e6779f3b3339a8c81466`
- Checks passed: 23

## Checks

- manifest.csv: 5419 rows
- indataset_results.csv: 9 rows
- crossdataset_matrix.csv: 18 rows
- generalization_gap.csv: 18 rows
- background_confound.csv: 3 rows
- gradcam_records.csv: 12 rows
- crossdataset_matrix_aug.csv: 18 rows
- generalization_gap_aug.csv: 18 rows
- lodo_results.csv: 9 rows
- mitigation_pairwise_aug.csv: 18 rows
- mitigation_comparison.csv: 9 rows
- Manifest identity matches canonical: week3_results\manifest.csv
- Manifest identity matches canonical: week4_results\manifest.csv
- Manifest identity matches canonical: week5_progress\manifest.csv
- Canonical manifest has 5,419 unique image paths
- All 9 in-dataset rows match their metrics JSON artifacts
- All 18 rows in generalization_gap.csv pass arithmetic and JSON checks
- All 18 rows in generalization_gap_aug.csv pass arithmetic and JSON checks
- Baseline and augmented transfer keys align 18/18
- All 9 LODO rows match their metrics JSON artifacts
- All 3 background-confound conditions are present
- Grad-CAM records contain both correct and incorrect samples
- Both Week 7 mitigation tables re-derive exactly

## Known limitations

- One Week 5 training-history JSON is absent; its checkpoint, evaluation artifacts, and summary rows are present.
- Week 7 checkpoints were not included in the downloaded result bundle.
- The Week 6 Grad-CAM overlay PNG/checkpoint is unavailable locally; the 12 sample records are frozen.
- All experiments use one frozen split and seed 42.

Frozen CSVs are immutable publication inputs. Regenerate this freeze
through `python -m freeze_results`; do not edit them manually.
