# Research CV: Cross-Dataset Generalization of Bangladeshi Rice Leaf Disease Models

## Project Goal
This repository studies how well rice leaf disease classifiers trained on one Bangladeshi dataset generalize to other Bangladeshi datasets.

## Contribution Framing
This is an application-plus-insight contribution with three first-class layers:
1. Benchmark: cross-dataset transfer matrix and generalization gap.
2. Diagnosis: Grad-CAM attention and background-confound analysis.
3. Mitigation: strong augmentation and leave-one-dataset-out training.

## Workflow
- Week 1: Setup, framing, and data inventory.
- Week 2: Harmonization (manifest, deduplication, and data loader).
- Week 3: Training and evaluation pipeline.
- Week 4: In-dataset baselines.
- Week 5: Cross-dataset transfer matrix.
- Week 6: Diagnosis (Grad-CAM + background confound).
- Weeks 7-8: Mitigation and result freeze.
- Week 9: Paper writing and figure generation.
- Week 10: Preprint and submission.

## Running Steps (high level)
1. Build manifest: `python -m src.build_manifest`
2. Deduplicate: `python -m src.dedup`
3. Train (Kaggle GPU): `python -m src.train --model mobilenetv2_100 --train_datasets riceleafbd --seed 42`
4. Evaluate: `python -m src.eval --checkpoint <path.pth> --eval_dataset riceleafbd`
5. In-dataset baselines (Week 4): `python -m run_indataset`
6. Cross-dataset transfer (Week 5): `python -m run_crossdataset`
7. Diagnosis (Week 6): `python -m run_diagnosis`
8. Mitigation (Week 7): `python -m run_crossdataset --augmentation strong`, `python -m run_lodo`, then `python -m run_mitigation`
9. Freeze results (Week 8): `python -m freeze_results`
10. Generate paper artifacts: `python -m make_figures`
11. Paper draft (Week 9): `paper/manuscript.md`

## Week 8 result freeze

The tracked publication inputs live under `frozen_results/`; the gitignored
`results/` directory remains a runtime workspace.

```bash
python -m freeze_results
python -m make_figures
```

The freeze command validates experiment coverage, unique keys, sample counts,
gap arithmetic, archived manifests, metrics JSON files, and Week 7 derived
tables. It writes `frozen_results/freeze_manifest.json` and
`frozen_results/audit_report.md`.

The figure command reads only frozen inputs and writes deterministic figures to
`paper/figures/` and matching CSV/LaTeX tables to `paper/tables/`.

## Week 9 paper draft

Full manuscript: [`paper/manuscript.md`](paper/manuscript.md).  
**Claude / LLM writing brief (all facts in one file):** [`paper/claude_paper_brief.md`](paper/claude_paper_brief.md).  
Plan/retro: `notes/week9_plan.md`, `notes/week9_retro.md`.

Week 10 is venue formatting, citations, preprint, and submission.

## Kaggle
- Week 3: `notebooks/kaggle_week3.md`
- Week 4: `notebooks/kaggle_week4.md`
- Week 5: `notebooks/kaggle_week5.md`
- Week 6: `notebooks/kaggle_week6.md`
- Week 7: `notebooks/kaggle_week7.md`
Frozen manifest is tracked at `artifacts/manifest.csv` (prepare with `python -m src.prepare_kaggle_manifest` on Kaggle).

## Notes
- Keep only original images (exclude pre-augmented copies).
- De-duplicate across datasets before evaluation.
- Freeze test splits after harmonization.
