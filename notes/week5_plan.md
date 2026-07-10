# Week 5 — Cross-dataset transfer matrix

## Protocol

- Six ordered source→target pairs.
- Three models: MobileNetV2, EfficientNet-B0, ResNet50.
- Seed 42 for the first complete matrix.
- Train only on pairwise shared classes.
- Evaluate each checkpoint on both source test and target test.
- Use the same ordered class list and assert identical class-index mappings.

## Shared-class pairs

- RiceLeafBD → Dhan-Shomadhan: brown_spot, tungro
- Dhan-Shomadhan → RiceLeafBD: brown_spot, tungro
- RiceLeafBD → BRRI: healthy, tungro
- BRRI → RiceLeafBD: healthy, tungro
- Dhan-Shomadhan → BRRI: rice_blast, scald, tungro
- BRRI → Dhan-Shomadhan: rice_blast, scald, tungro

## Outputs

- `results/crossdataset_matrix.csv`
- `results/generalization_gap.csv`
- Unique pair/subset checkpoints and metrics files
- Same-domain and cross-domain confusion matrices

## Resume behavior

- A combination present in both output CSVs is skipped.
- A unique pair checkpoint is reused if training completed but evaluation or CSV
  writing was interrupted.
- Restore the whole `results/` directory before resuming on a new Kaggle
  session.
- A run interrupted during training still restarts that one pair; the code does
  not yet save optimizer/scheduler state every epoch.

## Generalization gap

The reference is not the Week 4 full-class score. For each ordered pair:

`gap = source-test macro-F1 on shared classes − target-test macro-F1 on shared classes`

## Definition of done

- 18 rows in `crossdataset_matrix.csv`
- 18 rows in `generalization_gap.csv`
- Zero duplicate experiment keys
- Several target predictions hand-verified
- Class mapping identical in every source/target evaluation
- Progress archive downloaded locally

