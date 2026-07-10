# Week 5 Kaggle — Cross-dataset transfer matrix

This week trains 18 models: 3 architectures × 6 ordered dataset pairs. Each
checkpoint is evaluated on its source test set (shared-class reference) and its
target test set (cross-domain result).

Use a **T4 GPU**. Do not reinstall PyTorch.

## Cell 1 — Pull the latest code

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

# Keep Kaggle's preinstalled torch build.
!pip install -q timm scikit-learn albumentations opencv-python-headless grad-cam pandas matplotlib seaborn pillow imagehash pyyaml tqdm
```

## Cell 2 — Restore prior Week 5 progress (only after a break)

Upload your latest `week5_progress.zip` as a private Kaggle Dataset and attach
it. Change the path below to the attached zip's actual path.

```python
from pathlib import Path
import zipfile

restore_zip = Path("/kaggle/input/week5-progress/week5_progress.zip")
results_dir = Path("/kaggle/working/Research_CV/results")
results_dir.mkdir(parents=True, exist_ok=True)

if restore_zip.exists():
    with zipfile.ZipFile(restore_zip) as archive:
        archive.extractall(results_dir)
    print("Restored:", restore_zip)
else:
    print("No progress archive attached; starting a fresh Week 5 run.")
```

Restoring all of `results/` is important. The two progress CSVs let the runner
skip completed combinations; restored checkpoints let it continue from the
evaluation stage if training finished just before interruption.

## Cell 3 — Prepare the frozen manifest with Kaggle paths

```python
%cd /kaggle/working/Research_CV

from pathlib import Path
import shutil

roots = {
    "riceleafbd": "/kaggle/input/datasets/happychamp/research/riceleafbd/riceleafbd",
    "dhan_shomadhan": "/kaggle/input/datasets/happychamp/research/dhan_shomadhan/dhan_shomadhan",
    "brri_rice_disease_pest": "/kaggle/input/datasets/happychamp/research/brri_rice_disease_pest/brri_rice_disease_pest",
}

src_manifest = Path("artifacts/manifest.csv")
root_args = " ".join(f"{key}={value}" for key, value in roots.items())
!python -m src.prepare_kaggle_manifest \
  --src {src_manifest} \
  --out results/manifest.csv \
  --image_roots {root_args}
```

All three sample paths must print `exists=True`.

## Cell 4 — First-pair verification run

Run one model/pair first:

```python
!python -m run_crossdataset \
  --models mobilenetv2_100 \
  --pairs riceleafbd:dhan_shomadhan
```

Verify:

- shared classes are exactly `brown_spot`, `tungro`;
- source and target mappings are both `brown_spot=0`, `tungro=1`;
- sample prediction paths point to the correct target dataset;
- both `crossdataset_matrix.csv` and `generalization_gap.csv` contain one row.

## Cell 5 — Run in resumable model-sized chunks

The first command automatically skips the completed verification pair.

```python
!python -m run_crossdataset --models mobilenetv2_100
```

Back up results (Cell 6), then run:

```python
!python -m run_crossdataset --models efficientnet_b0
```

Back up again, then:

```python
!python -m run_crossdataset --models resnet50
```

You may stop between commands. Do not delete `results/`.

## Cell 6 — Back up progress after every model

```python
from pathlib import Path
import shutil

results_dir = Path("/kaggle/working/Research_CV/results")
archive = shutil.make_archive(
    "/kaggle/working/week5_progress",
    "zip",
    root_dir=results_dir,
)
print("Created:", archive)
print("Size MB:", round(Path(archive).stat().st_size / 1e6, 2))
```

Download `week5_progress.zip` from the Output panel. If the session ends, attach
that archive in the next notebook and run Cell 2 before resuming.

## Cell 7 — Final summary and checks

```python
!python -m run_crossdataset --summary_only

import pandas as pd

matrix = pd.read_csv("results/crossdataset_matrix.csv")
gaps = pd.read_csv("results/generalization_gap.csv")

print("matrix rows:", len(matrix), "expected: 18")
print("gap rows:", len(gaps), "expected: 18")
print("duplicate matrix keys:", matrix.duplicated(
    ["train_dataset", "test_dataset", "model", "classes", "seed"]
).sum())
print("duplicate gap keys:", gaps.duplicated(
    ["train_dataset", "test_dataset", "model", "classes", "seed"]
).sum())
```

Both files must have 18 rows and zero duplicate keys for seed 42.

## Optional: run selected ordered pairs

```python
!python -m run_crossdataset \
  --models efficientnet_b0 \
  --pairs dhan_shomadhan:brri_rice_disease_pest brri_rice_disease_pest:dhan_shomadhan
```

## Important interpretation

Week 4 full-class scores are not the gap references. Week 5 retrains and
evaluates on each pair's exact shared-class subset, then computes:

```text
generalization_gap = source-test shared-class macro-F1
                   - target-test shared-class macro-F1
```

