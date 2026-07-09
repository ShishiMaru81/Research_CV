# Week 4 Kaggle — In-dataset baselines (3 models x 3 datasets)

Use **GPU T4**. Do not reinstall torch.

Your uploaded data layout (from Week 3 debugging):
```text
/kaggle/input/datasets/happychamp/research/riceleafbd/riceleafbd/...
/kaggle/input/datasets/happychamp/research/dhan_shomadhan/...
/kaggle/input/datasets/happychamp/research/brri_rice_disease_pest/...
/kaggle/input/datasets/happychamp/result/manifest.csv
```

## Cell 1 — Pull latest code

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

## Cell 2 — Discover the 3 dataset roots

```python
from pathlib import Path

base = Path("/kaggle/input/datasets/happychamp/research")

def find_root(dataset: str, markers: list[str]) -> Path | None:
    start = base / dataset
    if not start.exists():
        # maybe dataset folder is already the class root
        start = base
    for marker in markers:
        hits = [p for p in start.rglob(marker) if p.is_dir()]
        if hits:
            return hits[0].parent
    return None

roots = {
    "riceleafbd": find_root("riceleafbd", ["Bacterial Leaf Blight", "Brown Spot", "Healthy Leaf", "Tungro Virus"]),
    "dhan_shomadhan": find_root("dhan_shomadhan", ["Field Background", "White Background", "Brown Spot", "Rice Blast"]),
    "brri_rice_disease_pest": find_root("brri_rice_disease_pest", ["Healthy", "Rice Blast", "Leaf Scald", "Rice Tungro"]),
}
for k, v in roots.items():
    print(k, "->", v)
assert all(v is not None for v in roots.values()), "Could not find one or more dataset roots"
```

Expected riceleafbd root (from your machine):
`/kaggle/input/datasets/happychamp/research/riceleafbd/riceleafbd`

For dhan_shomadhan, root should be the folder that contains `Field Background` and `White Background`.

## Cell 3 — Prepare config + rewritten manifest for all datasets

```python
import os, yaml
from pathlib import Path

%cd /kaggle/working/Research_CV

config = yaml.safe_load(open("config.yaml"))
config["results_root"] = "/kaggle/working/Research_CV/results"
os.makedirs(config["results_root"], exist_ok=True)
with open("config.yaml", "w") as f:
    yaml.dump(config, f)

# Prefer uploaded frozen manifest, else repo artifact
src = Path("/kaggle/input/datasets/happychamp/result/manifest.csv")
if not src.exists():
    src = Path("artifacts/manifest.csv")

args = " ".join([f"{k}={v}" for k, v in roots.items()])
!python -m src.prepare_kaggle_manifest --src {src} --out results/manifest.csv --image_roots {args}
```

All three sample paths must print `exists=True`.

## Cell 4 — Seed Week 3 result (optional but useful)

If you already have MobileNetV2×RiceLeafBD metrics, record them so `run_indataset` skips that combo:

```python
import pandas as pd
from pathlib import Path

out = Path("results/indataset_results.csv")
row = {
    "model": "mobilenetv2_100",
    "dataset": "riceleafbd",
    "seed": 42,
    "accuracy": 0.8936170212765957,
    "macro_f1": 0.9069841686306421,
    "n_samples": 235,
    "checkpoint_path": "results/checkpoints/mobilenetv2_100__train-riceleafbd__seed42.pth",
}
df = pd.DataFrame([row])
if out.exists():
    old = pd.read_csv(out)
    old = old[~((old.model==row["model"]) & (old.dataset==row["dataset"]) & (old.seed==row["seed"]))]
    df = pd.concat([old, df], ignore_index=True)
df.to_csv(out, index=False)
print(df)
```

## Cell 5 — Run Week 4 matrix

```python
!python -m run_indataset
```

This trains/evaluates remaining model×dataset combos with seed 42 and writes:
`results/indataset_results.csv`

Use **Save & Run All (Commit)** for long runs.

### Run a subset if needed
```bash
!python -m run_indataset --models mobilenetv2_100 --datasets dhan_shomadhan brri_rice_disease_pest
!python -m run_indataset --models efficientnet_b0 resnet50
```

## Cell 6 — Summary + download

```python
!python -m run_indataset --summary_only

import shutil
from pathlib import Path
src = Path("/kaggle/working/Research_CV/results")
shutil.make_archive("/kaggle/working/week4_results", "zip", root_dir=src)
print("Download /kaggle/working/week4_results.zip from Output panel")
```

## Success gate
Every in-dataset macro-F1 should be about **0.80–0.95**.  
If any run is far below 0.80, stop and debug before Week 5.
