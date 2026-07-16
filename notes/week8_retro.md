# Week 8 retrospective — Result freeze

## Outcome

The publication-facing results from Weeks 4–7 are frozen under
`frozen_results/`. The audit completed with **PASS** status and 23 checks.
No model was retrained and no result row was manually edited.

## Audit coverage

- Canonical manifest: 5,419 unique image paths.
- In-dataset baselines: 9/9 model-dataset runs.
- Baseline transfer and gap tables: 18/18 rows each.
- Strong-augmentation transfer and gap tables: 18/18 rows each.
- LODO: 9/9 held-out-dataset/model runs.
- Pairwise augmentation comparison: 18/18 rows.
- Strategy comparison: 9/9 target/model rows.
- Background-confound diagnosis: 3/3 conditions.
- Grad-CAM records: 12 samples, including correct and incorrect predictions.

The archived Week 3–5 manifests preserve the same split and label identities as
the canonical manifest after allowing for Kaggle image-root rewriting. All
available Week 4, Week 5, and Week 7 summary metrics agree with their per-run
JSON files. Both Week 7 comparison tables re-derived exactly from their source
CSVs.

Canonical manifest SHA-256:
`3a1a981ae73cded5b7dc46f6a3e479594c6d7a71af78e6779f3b3339a8c81466`.

## Publication artifacts

`make_figures.py` generated:

- 9 paper figures in `paper/figures/`.
- 8 paper tables in both CSV and LaTeX formats in `paper/tables/`.
- A figure manifest linking every generated artifact to frozen input hashes.

A second complete generation produced identical hashes for all 25 figure and
table files.

## Known limitations

- One Week 5 training-history JSON is missing, but its checkpoint, evaluation
  metrics, confusion matrices, and summary rows are present.
- Week 7 checkpoints were not included in the downloaded result bundle.
- The Week 6 Grad-CAM overlay and checkpoint are unavailable locally, so the
  overlay was not regenerated. The 12 sample records are frozen.
- All reported experiments use one frozen split and seed 42.

## Freeze rule

The files under `frozen_results/` are now immutable paper inputs. Any correction
must be made through a documented replacement freeze and full audit, never by
editing a frozen CSV.

## Next step

Week 9 can now write the complete manuscript using the frozen tables and
figures, beginning with Methods, Results, Discussion, and Limitations.
