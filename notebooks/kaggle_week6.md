# Week 6 Kaggle — Grad-CAM and background-confound diagnosis

The representative diagnosis is ResNet50 trained on Dhan-Shomadhan and tested
on RiceLeafBD using `brown_spot` and `tungro`.

Use a **T4 GPU**. Do not reinstall PyTorch.

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

!pip install -q timm scikit-learn albumentations opencv-python-headless \
  grad-cam pandas matplotlib seaborn pillow imagehash pyyaml tqdm
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

All three samples must print `exists=True`.

## Cell 3 — Optional: restore the Week 5 checkpoint

The diagnosis needs:

```text
resnet50__train-dhan_shomadhan__run-to-riceleafbd__classes-brown_spot+tungro__seed42.pth
```

If you saved that checkpoint as a Kaggle input, copy it into
`results/checkpoints/`. Otherwise skip this cell: Cell 4 automatically retrains
only this one representative model.

```python
from pathlib import Path
import shutil

source = Path("/kaggle/input/week5-checkpoint/resnet50__train-dhan_shomadhan__run-to-riceleafbd__classes-brown_spot+tungro__seed42.pth")
destination = Path("results/checkpoints") / source.name
destination.parent.mkdir(parents=True, exist_ok=True)

if source.exists():
    shutil.copy2(source, destination)
    print("Restored:", destination)
else:
    print("Checkpoint input not attached; diagnosis will retrain one model.")
```

## Cell 4 — Run the full diagnosis

```python
!python -m run_diagnosis --sample_size 12 --seed 42
```

This produces:

```text
paper/figures/gradcam_examples.png
paper/figures/background_confound.png
results/gradcam_records.csv
results/background_confound.csv
```

## Cell 5 — Inspect quantitative output

```python
import pandas as pd

print(pd.read_csv("results/background_confound.csv").to_string(index=False))
print("\nGrad-CAM sample records:")
print(pd.read_csv("results/gradcam_records.csv")[
    [
        "true_label",
        "pred_label",
        "correct",
        "border_attention_fraction",
        "border_attention_enrichment",
    ]
].to_string(index=False))
```

## Cell 6 — Display figures

```python
from IPython.display import Image, display

display(Image("paper/figures/gradcam_examples.png"))
display(Image("paper/figures/background_confound.png"))
```

Human verification is mandatory:

- Does attention land on lesions, leaf shape, or background?
- Are incorrect predictions more background-focused?
- Does white-background macro-F1 exceed field and RiceLeafBD field performance?

The border-CAM value is only a reproducible proxy. Do not call it lesion
attention without segmentation masks.

## Cell 7 — Bundle and download Week 6

```python
from pathlib import Path
import shutil

bundle = Path("/kaggle/working/week6_bundle")
if bundle.exists():
    shutil.rmtree(bundle)
(bundle / "results").mkdir(parents=True)
(bundle / "paper" / "figures").mkdir(parents=True)

for name in ["gradcam_records.csv", "background_confound.csv"]:
    shutil.copy2(Path("results") / name, bundle / "results" / name)

for name in ["gradcam_examples.png", "background_confound.png"]:
    shutil.copy2(
        Path("paper/figures") / name,
        bundle / "paper" / "figures" / name,
    )

archive = shutil.make_archive(
    "/kaggle/working/week6_results",
    "zip",
    root_dir=bundle,
)
print("Download:", archive)
```

