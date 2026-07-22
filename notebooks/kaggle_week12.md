# Week 12 Kaggle — AdaBN (Phase 2)

Adaptive BatchNorm on **source-trained baseline checkpoints**. Recalibrates BN
running stats on the **target train** split (no labels used for adaptation),
then evaluates on the target test split.

Scope: 6 transfer pairs × 3 models × seeds = up to **54** eval-only jobs
(~1–2 min each on T4). Start with seed **42**.

Requires the baseline `.pth` files under `results/checkpoints/` (restore from
your Week 5/7/11 bundle, or from a previous Kaggle output).

## Cell 1 — Clone / pull

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

!git checkout phase0-instrumentation

!pip install -q timm scikit-learn albumentations opencv-python-headless pandas matplotlib seaborn pillow imagehash pyyaml tqdm
```

## Cell 2 — Manifest

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

## Cell 3 — Restore baseline checkpoints

Upload a zip that contains `checkpoints/*.pth` for seed-42 transfer baselines
(18 files), or copy from a prior Kaggle dataset.

```python
from pathlib import Path
import shutil
import zipfile

RESTORE_ZIP = Path("/kaggle/input/your-baseline-ckpts/week5_or_week11_ckpts.zip")
# Adjust path to your uploaded dataset.

if RESTORE_ZIP.exists():
    extract_to = Path("/kaggle/working/restore_ckpts")
    if extract_to.exists():
        shutil.rmtree(extract_to)
    extract_to.mkdir()
    with zipfile.ZipFile(RESTORE_ZIP) as zf:
        zf.extractall(extract_to)
    dest = Path("results/checkpoints")
    dest.mkdir(parents=True, exist_ok=True)
    for pth in extract_to.rglob("*.pth"):
        if "seed42" in pth.name and "aug-strong" not in pth.name and "lodo" not in pth.name:
            shutil.copy2(pth, dest / pth.name)
            print("Restored", pth.name)
else:
    print("No checkpoint zip — place .pth files under results/checkpoints/")

print("seed42 baseline ckpts:", len(list(Path("results/checkpoints").glob("*seed42.pth"))))
```

## Cell 4 — Dry run

```python
!python -m run_adabn --seeds 42 --dry_run
```

Expect **18 would execute** when all seed-42 baseline checkpoints are present.

## Cell 5 — Run AdaBN (seed 42)

```python
!python -m run_adabn --seeds 42
```

Output: `results/adabn/adabn_results.csv`

## Cell 6 — Optional: seeds 7 and 2024

Only if those baseline checkpoints were restored.

```python
!python -m run_adabn --seeds 7 2024
```

## Cell 7 — Inspect + light download

```python
import pandas as pd
from pathlib import Path
import shutil

df = pd.read_csv("results/adabn/adabn_results.csv")
print(df.groupby("model")[["baseline_macro_f1", "adabn_macro_f1", "delta_macro_f1"]].mean())
print(df[["train_dataset", "test_dataset", "model", "delta_macro_f1"]].sort_values("delta_macro_f1"))

bundle = Path("/kaggle/working/week12_adabn")
if bundle.exists():
    shutil.rmtree(bundle)
bundle.mkdir()
shutil.copytree("results/adabn", bundle / "adabn")
archive = shutil.make_archive("/kaggle/working/week12_adabn", "zip", root_dir=bundle)
print("Download:", archive)
```

## Protocol notes (for the paper)

- Adaptation uses the **target dataset train split** only; never the test split.
- Labels are not used during BN recalibration (forward passes only).
- Non-BN modules stay in eval mode (no dropout noise during adaptation).
