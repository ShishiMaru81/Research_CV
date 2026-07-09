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
1. Build manifest: python -m src.build_manifest
2. Deduplicate: python -m src.dedup
3. In-dataset baselines: python -m run_indataset
4. Cross-dataset transfer: python -m run_crossdataset
5. Diagnosis: python -m run_diagnosis
6. Figures: python -m make_figures

## Notes
- Keep only original images (exclude pre-augmented copies).
- De-duplicate across datasets before evaluation.
- Freeze test splits after harmonization.
