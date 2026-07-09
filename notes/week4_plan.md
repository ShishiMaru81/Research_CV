# Week 4 Notes — In-dataset baselines

## Goal
Fill the matrix diagonal: 3 models x 3 datasets, evaluate on each dataset's own test split.

## Week 3 carry-over
- MobileNetV2 / RiceLeafBD / seed 42
- accuracy=0.8936, macro_f1=0.9070, n=235
- Local copy: `week3_results/`

## Code added
- `run_indataset.py` — orchestration + `results/indataset_results.csv`
- `src/prepare_kaggle_manifest.py` — rewrite Windows-relative paths to Kaggle absolute paths for all datasets
- Fixed backslash filename bug in `src/data_loader.py`
- Added `--image_roots dataset=/abs/path` support

## Kaggle
Follow `notebooks/kaggle_week4.md`.

## Definition of done
- `indataset_results.csv` has all 9 rows (seed 42)
- All macro-F1 in ~0.80–0.95
- Download `week4_results.zip`
