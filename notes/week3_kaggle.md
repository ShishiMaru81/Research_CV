# Week 3 Notes — Training / Eval Pipeline (Kaggle GPU)

## Delivered
- `src/train.py`: timm fine-tune, 3-epoch head freeze then full fine-tune, Adam, ReduceLROnPlateau, early stopping on val macro-F1, class-weighted CE, AMP on CUDA.
- `src/eval.py`: metrics JSON, confusion matrix PNG, sample predictions, experiment_log.csv append.
- `notebooks/kaggle_week3.md`: copy-paste Kaggle cells.
- `artifacts/manifest.csv`: frozen Week-2 manifest for Kaggle (tracked in git).

## Kaggle run command (after pull)
```bash
python -m src.train --model mobilenetv2_100 --train_datasets riceleafbd --eval_dataset riceleafbd --seed 42 --path_remap data/raw/riceleafbd /kaggle/input/riceleafbd
python -m src.eval --checkpoint /kaggle/working/results/checkpoints/mobilenetv2_100__train-riceleafbd__seed42.pth --eval_dataset riceleafbd --seed 42 --path_remap data/raw/riceleafbd /kaggle/input/riceleafbd
```

Adjust `/kaggle/input/riceleafbd` to your actual Kaggle dataset slug path.

## Success gate
Test macro-F1 on RiceLeafBD should land near 0.85–0.95 before starting Week 4 matrix runs.
