# Week 3 Kaggle GPU Notebook (copy-paste cells)

Use Accelerator: **GPU** (T4 x2 or P100).  
Attach your uploaded datasets + `rice-manifest` (with `manifest.csv`).

## Cell 1 — Clone private repo + install

```python
import os
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone https://{token}@github.com/ShishiMaru81/Research_CV.git
%cd Research_CV
!pip install -r requirements.txt -q
```

## Cell 2 — Configure Kaggle paths + place manifest

```python
import os, yaml, shutil
from pathlib import Path

# Adjust these slugs to match your Kaggle Dataset names exactly
DATASET_SLUGS = {
    "riceleafbd": "/kaggle/input/riceleafbd",
    "dhan_shomadhan": "/kaggle/input/dhan-shomadhan",
    "brri_rice_disease_pest": "/kaggle/input/brri-rice-disease-pest",
}
MANIFEST_CANDIDATES = [
    "/kaggle/input/rice-manifest/manifest.csv",
    "/kaggle/input/manifest/manifest.csv",
]

config = yaml.safe_load(open("config.yaml"))
config["data_root"] = "/kaggle/input"
config["results_root"] = "/kaggle/working/results"
os.makedirs(config["results_root"], exist_ok=True)
with open("config.yaml", "w") as f:
    yaml.dump(config, f)

# Put frozen manifest where the code expects it
results_manifest = Path(config["results_root"]) / "manifest.csv"
for cand in MANIFEST_CANDIDATES:
    if Path(cand).exists():
        shutil.copy(cand, results_manifest)
        print("Copied manifest from", cand)
        break
else:
    raise FileNotFoundError(
        "Attach a Kaggle dataset containing manifest.csv "
        "(recommended name: rice-manifest)."
    )

print("CUDA available:", __import__("torch").cuda.is_available())
print("Config ready.")
```

## Cell 3 — Inspect one remapped path (sanity)

```python
import pandas as pd
from pathlib import Path

m = pd.read_csv("/kaggle/working/results/manifest.csv")
# Example remap: local Windows-ish relative path -> Kaggle mount
sample = m[m.dataset == "riceleafbd"].iloc[0].image_path.replace("\\", "/")
# If your uploaded dataset root already contains class folders, remap like:
# old: data/raw/riceleafbd
# new: /kaggle/input/riceleafbd
print("sample path in manifest:", sample)
print("exists after expected remap?", Path(sample.replace("data/raw/riceleafbd", "/kaggle/input/riceleafbd")).exists())
```

## Cell 4 — Train MobileNetV2 on RiceLeafBD (Week 3 baseline)

```python
!python -m src.train \
  --model mobilenetv2_100 \
  --train_datasets riceleafbd \
  --eval_dataset riceleafbd \
  --seed 42 \
  --path_remap data/raw/riceleafbd /kaggle/input/riceleafbd
```

If your Kaggle dataset slug/path differs, change the second remap value.

If all three datasets are mounted and you rebuilt the manifest on Kaggle with matching local-style relative paths, you can omit `--path_remap` and instead rebuild:

```python
# Optional alternative: rebuild manifest on Kaggle after arranging
# /kaggle/working/data/raw/<dataset>/...
# !python -m src.build_manifest
```

## Cell 5 — Evaluate best checkpoint

```python
!python -m src.eval \
  --checkpoint /kaggle/working/results/checkpoints/mobilenetv2_100__train-riceleafbd__seed42.pth \
  --eval_dataset riceleafbd \
  --seed 42 \
  --path_remap data/raw/riceleafbd /kaggle/input/riceleafbd
```

## Cell 6 — Download / persist outputs

Download `/kaggle/working/results/` from the notebook Output panel, or push selected files back to GitHub.

Use **Save & Run All (Commit)** for long training so the session survives disconnects.

## Success check
- `torch.cuda.is_available()` is True
- val macro-F1 rises over epochs
- test macro-F1 roughly in **0.85–0.95**
- sample predictions look label-correct
