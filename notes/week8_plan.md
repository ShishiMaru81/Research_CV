# Week 8 — Result freeze

## Goal

Freeze the publication-facing results from Weeks 4–7, verify that summary
tables agree with their underlying artifacts, and generate all CSV-driven paper
figures and tables deterministically. No new model training or selective reruns
are part of Week 8.

## Canonical inputs

| Result | Source | Expected rows |
|--------|--------|--------------:|
| Frozen data manifest | `artifacts/manifest.csv` | 5,419 |
| In-dataset baselines | `week4_results/indataset_results.csv` | 9 |
| Baseline transfer matrix | `week5_progress/crossdataset_matrix.csv` | 18 |
| Baseline generalization gaps | `week5_progress/generalization_gap.csv` | 18 |
| Background-confound diagnosis | `week6_results/results/background_confound.csv` | 3 |
| Grad-CAM sample records | `week6_results/results/gradcam_records.csv` | 12 |
| Strong-augmentation transfer matrix | `week7_results/crossdataset_matrix_aug.csv` | 18 |
| Strong-augmentation gaps | `week7_results/generalization_gap_aug.csv` | 18 |
| LODO results | `week7_results/lodo_results.csv` | 9 |
| Pairwise augmentation comparison | `week7_results/mitigation_pairwise_aug.csv` | 18 |
| Strategy comparison | `week7_results/mitigation_comparison.csv` | 9 |

The tracked freeze is written to `frozen_results/`. The configured `results/`
directory remains a gitignored runtime workspace.

## Freeze checks

- Exact row counts and required columns.
- Unique experiment keys and complete model/dataset coverage.
- Accuracy, macro-F1, and gap values in valid ranges.
- `generalization_gap = in_dataset_macro_f1 - cross_macro_f1`.
- Baseline and augmented transfer keys match one-to-one.
- Week 7 comparison tables re-derive exactly from their source CSVs.
- Summary CSV values agree with available per-run metrics JSON files.
- Manifest split and label identities are unchanged across archived copies;
  image roots may differ because Kaggle paths were rewritten.
- SHA-256 hashes are recorded for every frozen input.

## Freeze policy

After audit sign-off, files under `frozen_results/` are immutable publication
inputs. Corrections require a documented replacement freeze, not manual CSV
editing. Results are reported for the prespecified frozen split and seed 42.

## Known non-blocking limitations

- One Week 5 training-history JSON is absent, while its checkpoint, evaluation
  metrics, confusion matrices, and summary rows are present.
- Week 7 checkpoints were not included in the downloaded bundle.
- The Week 6 Grad-CAM overlay image and checkpoint are not available locally;
  its 12 sample records remain available, but the overlay is not regenerated in
  Week 8.

## Definition of done

- `python -m freeze_results` passes and writes the freeze manifest/audit report.
- `python -m make_figures` writes the deterministic paper figures and tables.
- A second figure run produces identical file hashes.
- `notes/week8_retro.md` records the completed audit and limitations.
