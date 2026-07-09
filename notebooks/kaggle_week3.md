# Week 3 Kaggle GPU Notebook (copy-paste cells)

Use Accelerator: **GPU T4** if available (P100 can conflict with newer PyTorch wheels).  
Attach your uploaded `riceleafbd` dataset (+ later the other two).

## Important before anything
1. Do **not** reinstall `torch` on Kaggle (`pip install -r requirements.txt` can break P100).
2. Avoid nested clones. If you already cloned once, `cd` into that folder and `git pull` instead of cloning again.
3. First discover your real dataset path with Cell 0.

## Cell 0 — Find the real image root

```python
from pathlib import Path

root = Path("/kaggle/input")
print("Attached inputs:")
for p in sorted(root.iterdir()):
    print(" -", p)

# Dig until you see class folders like "Bacterial Leaf Blight", "Brown Spot", ...
def find_class_root(start: Path, marker="Bacterial Leaf Blight"):
    for p in start.rglob(marker):
        if p.is_dir():
            return p.parent
    return None

for ds in sorted(root.iterdir()):
    hit = find_class_root(ds)
    print(ds.name, "-> class root:", hit)
```

Copy the printed class-root path (example: `/kaggle/input/riceleafbd` or `/kaggle/input/riceleafbd/riceleafbd`).

## Cell 1 — Clone/pull private repo + install (no torch reinstall)

```python
import os
from pathlib import Path
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo_dir = Path("/kaggle/working/Research_CV")

if (repo_dir / ".git").exists():
    %cd /kaggle/working/Research_CV
    !git pull
else:
    %cd /kaggle/working
    !git clone https://{token}@github.com/ShishiMaru81/Research_CV.git
    %cd Research_CV

# Install research deps WITHOUT touching the Kaggle torch build
!pip install -q timm scikit-learn albumentations opencv-python-headless grad-cam pandas matplotlib seaborn pillow imagehash pyyaml tqdm
```

## Cell 2 — Configure paths + place frozen manifest

```python
import os, yaml, shutil
from pathlib import Path

# <<< PASTE the class-root from Cell 0 here >>>
IMAGE_ROOT = "/kaggle/input/riceleafbd"   # change if Cell 0 shows nesting

config = yaml.safe_load(open("config.yaml"))
config["data_root"] = "/kaggle/input"
config["results_root"] = "/kaggle/working/results"
os.makedirs(config["results_root"], exist_ok=True)
with open("config.yaml", "w") as f:
    yaml.dump(config, f)

# Prefer tracked frozen manifest from the repo
src_manifest = Path("artifacts/manifest.csv")
dst_manifest = Path(config["results_root"]) / "manifest.csv"
if src_manifest.exists():
    shutil.copy(src_manifest, dst_manifest)
    print("Copied artifacts/manifest.csv ->", dst_manifest)
else:
    raise FileNotFoundError("artifacts/manifest.csv missing; git pull latest main.")

import torch
print("CUDA available:", torch.cuda.is_available())
print("IMAGE_ROOT exists:", Path(IMAGE_ROOT).exists())
print("Example class folders:", [p.name for p in Path(IMAGE_ROOT).iterdir()][:10])
```

## Cell 3 — Path sanity (must print True)

```python
import pandas as pd
from pathlib import Path

IMAGE_ROOT = "/kaggle/input/riceleafbd"  # same as Cell 2
m = pd.read_csv("/kaggle/working/results/manifest.csv")
row = m[m.dataset == "riceleafbd"].iloc[0]
rebuilt = str(Path(IMAGE_ROOT) / row.original_class / Path(row.image_path).name)
print("rebuilt:", rebuilt)
print("exists:", Path(rebuilt).exists())
```

If `exists: False`, your `IMAGE_ROOT` is wrong — go back to Cell 0.

## Cell 4 — Train MobileNetV2 on RiceLeafBD

```python
!python -m src.train \
  --model mobilenetv2_100 \
  --train_datasets riceleafbd \
  --eval_dataset riceleafbd \
  --seed 42 \
  --image_root /kaggle/input/riceleafbd
```

Use the exact `IMAGE_ROOT` from Cell 0/2.

## Cell 5 — Evaluate

```python
!python -m src.eval \
  --checkpoint /kaggle/working/results/checkpoints/mobilenetv2_100__train-riceleafbd__seed42.pth \
  --eval_dataset riceleafbd \
  --seed 42 \
  --image_root /kaggle/input/riceleafbd
```

## Cell 6 — Persist outputs
Download `/kaggle/working/results/` from the notebook Output panel.

## Notes on your previous errors
- `Failed to read image: /kaggle/input/riceleafbd/...` = wrong folder nesting / wrong remap. Use `--image_root` after Cell 0.
- P100 + `pip install torch` warning = do not reinstall torch; use Kaggle’s preinstalled build, prefer T4 if available.
- Nested path `Research_CV/Research_CV/Research_CV` = you cloned repeatedly. Pull instead.
- HF token warning is harmless for this run.
