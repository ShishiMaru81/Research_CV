# Week 11 Kaggle — Multi-seed replication (seeds 7 and 2024)

Re-runs the comparative experiments at two additional **train seeds** while
**split_seed stays 42** (frozen Week 3–9 splits). Seed-42 checkpoints in
`frozen_results/` are not retrained.

Scope per new seed:

- Transfer baseline: 6 pairs × 3 models = **18** runs
- Strong augmentation: 6 pairs × 3 models = **18** runs
- LODO: 3 holdouts × 3 models = **9** runs

**45 runs × 2 seeds = 90 training jobs** (~18–23 GPU-hrs on T4).

Use a **T4 GPU**. Do not reinstall PyTorch. The campaign is resumable via
`results/run_registry.csv` — re-run the same cell after any timeout.

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

!git checkout phase0-instrumentation  # or main after merge

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

## Cell 3 — Resume: restore a previous results bundle

If a prior session was interrupted, upload your downloaded zip to Kaggle
(e.g. as a dataset) and restore `results/` before continuing.

```python
from pathlib import Path
import shutil
import zipfile

# Option A: copy from an uploaded dataset path
RESTORE_ZIP = Path("/kaggle/input/your-week11-results/week11_results.zip")

if RESTORE_ZIP.exists():
    extract_to = Path("/kaggle/working/restore")
    if extract_to.exists():
        shutil.rmtree(extract_to)
    extract_to.mkdir()
    with zipfile.ZipFile(RESTORE_ZIP) as zf:
        zf.extractall(extract_to)
    for name in ["run_registry.csv", "manifest.csv"]:
        src = extract_to / name
        if not src.exists():
            src = extract_to / "results" / name
        if src.exists():
            shutil.copy2(src, Path("results") / name)
            print("Restored:", name)
    for folder in ["checkpoints", "predictions", "multiseed"]:
        src_dir = extract_to / folder
        if not src_dir.exists():
            src_dir = extract_to / "results" / folder
        if src_dir.exists():
            dest = Path("results") / folder
            dest.mkdir(parents=True, exist_ok=True)
            for item in src_dir.iterdir():
                target = dest / item.name
                if item.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
            print("Restored folder:", folder)
else:
    print("No restore zip found — starting fresh (registry will be empty).")
```

## Cell 4 — GPU check + dry run

```python
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

!python -m run_multiseed --dry_run
```

Expect **90 total jobs** enumerated and **90 would execute** on a fresh registry.
If the total is not 90, stop and diagnose before training.

## Cell 5 — Warm pretrained backbones (optional, avoids mid-run Hub stalls)

```python
import timm
for name in ["mobilenetv2_100", "efficientnet_b0", "resnet50"]:
    timm.create_model(name, pretrained=True)
    print("Cached:", name)
```

## Cell 6 — Run the multi-seed campaign

**Do not pass `--image_roots` here** if Cell 2 already rewrote manifest paths
and samples show `exists=True` (same Week 7 discipline).

```python
!python -m run_multiseed
```

Runs seed-major: all of seed **7**, then all of seed **2024**. Skips any run
already `status=complete` in `results/run_registry.csv`.

Outputs:

- `results/multiseed/crossdataset_matrix_baseline.csv`
- `results/multiseed/crossdataset_matrix_aug.csv`
- `results/multiseed/generalization_gap_baseline.csv`
- `results/multiseed/generalization_gap_aug.csv`
- `results/multiseed/lodo_results.csv`
- `results/predictions/*.csv` (one per eval run)
- `results/checkpoints/*.pth`
- `results/run_registry.csv`

## Cell 7 — Inspect progress

```python
import pandas as pd
from pathlib import Path

reg = pd.read_csv("results/run_registry.csv")
print(reg.groupby(["experiment_type", "status"]).size())
print("\nPending:", reg.loc[reg["status"] != "complete", ["run_id", "train_seed", "experiment_type"]].head(20))

for name in [
    "multiseed/crossdataset_matrix_baseline.csv",
    "multiseed/crossdataset_matrix_aug.csv",
    "multiseed/lodo_results.csv",
]:
    path = Path("results") / name
    if path.exists():
        df = pd.read_csv(path)
        print(f"\n{name}: {len(df)} rows, seeds={sorted(df['seed'].unique())}")
```

## Cell 8 — Bundle and download (every session)

```python
from pathlib import Path
import shutil

bundle = Path("/kaggle/working/week11_bundle")
if bundle.exists():
    shutil.rmtree(bundle)
bundle.mkdir(parents=True)

for folder in ["multiseed", "checkpoints", "predictions"]:
    src = Path("results") / folder
    if src.exists():
        shutil.copytree(src, bundle / folder)

for name in ["run_registry.csv", "manifest.csv"]:
    src = Path("results") / name
    if src.exists():
        shutil.copy2(src, bundle / name)

archive = shutil.make_archive("/kaggle/working/week11_results", "zip", root_dir=bundle)
print("Download:", archive)
```

Download `week11_results.zip` from Kaggle's output panel **every session**, even
if the campaign is not finished. Re-upload in Cell 3 on the next session.

## Discipline

- Never write to `frozen_results/` — multi-seed tables live in `results/multiseed/`.
- `split_seed=42` always; only `train_seed` varies (7, 2024).
- If a headline seed-42 number moves after replication, update the manuscript —
  do not cherry-pick the old anecdote.
