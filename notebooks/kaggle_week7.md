# Week 7 Kaggle — Mitigation (strong augmentation + LODO)

Two mitigation experiments, each changing exactly one variable versus the Week 5
baseline:

1. Strong augmentation on the same 6 transfer pairs (train loader only).
2. Leave-one-dataset-out (LODO): train on two datasets, test on the held-out
   third.

Use a **T4 GPU**. Do not reinstall PyTorch. Expect several hours total; the
runners are resumable, so re-run the same cell after any interruption.

## Cell 1 — Clone or pull

```python
from pathlib import Path
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo = Path("/kaggle/working/Research_CV")

if (repo / ".git").exists():
    %cd /kaggle/working/Research_CV
    !git pull
else:
    %cd /kaggle/working
    !git clone https://{token}@github.com/ShishiMaru81/Research_CV.git
    %cd Research_CV

!pip install -q timm scikit-learn albumentations opencv-python-headless grad-cam pandas matplotlib seaborn pillow imagehash pyyaml tqdm
```

## Cell 2 — Prepare the frozen manifest

```python
roots = {
    "riceleafbd": "/kaggle/input/datasets/happychamp/research/riceleafbd/riceleafbd",
    "dhan_shomadhan": "/kaggle/input/datasets/happychamp/research/dhan_shomadhan/dhan_shomadhan",
    "brri_rice_disease_pest": "/kaggle/input/datasets/happychamp/research/brri_rice_disease_pest/brri_rice_disease_pest",
}

root_args = " ".join(f"{key}={value}" for key, value in roots.items())

!python -m src.prepare_kaggle_manifest \
  --src artifacts/manifest.csv \
  --out results/manifest.csv \
  --image_roots {root_args}
```

Confirm the printed samples show `exists=True`.

## Cell 3 — Optional: restore Week 5 baseline CSVs

The mitigation comparison reads the Week 5 baseline. If your session is fresh,
copy the Week 5 CSVs into `results/` so they can be compared side by side.

```python
from pathlib import Path
import shutil

for name in ["crossdataset_matrix.csv", "generalization_gap.csv"]:
    src = Path("week5_progress") / name
    if src.exists():
        shutil.copy2(src, Path("results") / name)
        print("Restored:", name)
    else:
        print("Missing baseline:", name, "(comparison will skip baseline columns)")
```

## Cell 4 — Strong-augmentation transfer matrix

```python
root_args = " ".join(f"{k}={v}" for k, v in roots.items())

!python -m run_crossdataset \
  --augmentation strong \
  --image_roots {root_args}
```

Writes `results/crossdataset_matrix_aug.csv` and
`results/generalization_gap_aug.csv`. Re-running skips completed cells.

## Cell 5 — Leave-one-dataset-out

```python
!python -m run_lodo --image_roots {root_args}
```

Writes `results/lodo_results.csv`. Label spaces are chosen automatically as the
largest non-degenerate shared set per held-out dataset:

- hold out RiceLeafBD -> {brown_spot, healthy, tungro}
- hold out Dhan-Shomadhan -> {brown_spot, rice_blast, scald, tungro}
- hold out BRRI -> {healthy, rice_blast, scald, tungro}

## Cell 6 — Build the comparison table

```python
!python -m run_mitigation
```

Writes `results/mitigation_pairwise_aug.csv` (per-pair augmentation effect and
gap reduction) and `results/mitigation_comparison.csv` (single-source vs
augmentation vs LODO, by target dataset).

## Cell 7 — Inspect results

```python
import pandas as pd

for name in [
    "crossdataset_matrix_aug.csv",
    "mitigation_pairwise_aug.csv",
    "lodo_results.csv",
    "mitigation_comparison.csv",
]:
    print(f"\n===== {name} =====")
    print(pd.read_csv(f"results/{name}").to_string(index=False))
```

## Cell 8 — Bundle and download

```python
from pathlib import Path
import shutil

bundle = Path("/kaggle/working/week7_bundle")
if bundle.exists():
    shutil.rmtree(bundle)
bundle.mkdir(parents=True)

wanted = [
    "crossdataset_matrix_aug.csv",
    "generalization_gap_aug.csv",
    "lodo_results.csv",
    "mitigation_pairwise_aug.csv",
    "mitigation_comparison.csv",
]
for name in wanted:
    src = Path("results") / name
    if src.exists():
        shutil.copy2(src, bundle / name)

# Include the per-run eval metrics/confusion matrices produced this week.
for extra in Path("results").glob("*aug-strong*"):
    shutil.copy2(extra, bundle / extra.name)
for extra in Path("results").glob("*lodo-holdout*"):
    shutil.copy2(extra, bundle / extra.name)

archive = shutil.make_archive("/kaggle/working/week7_results", "zip", root_dir=bundle)
print("Download:", archive)
```

Download `week7_results.zip` from Kaggle's output/files panel and share it back.

## Discipline

Each mitigation changes one variable. Do not combine strong augmentation with
LODO in the same run when attributing improvement; the comparison table keeps
them separate on purpose.
