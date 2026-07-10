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
8. Figures: `python -m make_figures`

## Kaggle
- Week 3: `notebooks/kaggle_week3.md`
- Week 4: `notebooks/kaggle_week4.md`
- Week 5: `notebooks/kaggle_week5.md`
- Week 6: `notebooks/kaggle_week6.md`
Frozen manifest is tracked at `artifacts/manifest.csv` (prepare with `python -m src.prepare_kaggle_manifest` on Kaggle).

## Notes
- Keep only original images (exclude pre-augmented copies).
- De-duplicate across datasets before evaluation.
- Freeze test splits after harmonization.
