# Week 13 Kaggle — Augmentation bucket ablation (Phase 3)

Three mechanism buckets, each alone, on **ResNet50 × 6 transfer pairs × seed 42**
= **18 training runs** (~4 GPU-hrs on T4).

| Bucket | CLI flag | Transforms |
|--------|----------|------------|
| Geometric / background | `bucket-geo` | RandomResizedCrop, HorizontalFlip, Affine |
| Photometric | `bucket-photo` | BrightnessContrast, HueSaturationValue |
| Occlusion | `bucket-occlusion` | GaussianBlur, CoarseDropout |

Outputs land under `results/ablation/`:

- `augmentation_ablation.csv` (aggregated)
- `crossdataset_matrix_bucket_{geo,photo,occlusion}.csv`
- matching gap CSVs + checkpoints / predictions via the usual paths

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

!pip install -q timm scikit-learn albumentations opencv-python-headless pandas matplotlib seaborn pillow imagehash pyyaml tqdm scipy
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

## Cell 3 — Dry run (resume check)

```python
!python -m run_ablation --dry_run --image_roots {root_args}
```

Expect 18 pending jobs on a fresh session (or a mix of `complete` / `pending`
if resuming).

## Cell 4 — Run all buckets

```python
!python -m run_ablation --image_roots {root_args}
```

To run one bucket at a time (safer across session timeouts):

```python
!python -m run_ablation --buckets bucket-geo --image_roots {root_args}
!python -m run_ablation --buckets bucket-photo --image_roots {root_args}
!python -m run_ablation --buckets bucket-occlusion --image_roots {root_args}
```

## Cell 5 — Verify + light zip

```python
from pathlib import Path
import pandas as pd

abl = Path("results/ablation/augmentation_ablation.csv")
assert abl.exists(), abl
df = pd.read_csv(abl)
print(df.groupby("bucket").size())
print("rows:", len(df))
assert len(df) >= 18

!mkdir -p /kaggle/working/week13_light
!cp -r results/ablation /kaggle/working/week13_light/
!cp results/run_registry.csv /kaggle/working/week13_light/ || true
!cd /kaggle/working && zip -r week13_ablation_light.zip week13_light
print("Download week13_ablation_light.zip from Output")
```

## After download (local)

```bash
# place augmentation_ablation.csv under results/ablation/
python -m scripts.build_ablation_tables
```

That writes `paper/tables/table_ablation*.csv` and
`notes/ablation_interpretation.md`.
