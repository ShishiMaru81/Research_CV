# Week 13 Kaggle — Masked transfer matrix

Retrain the **6 pairs × 3 models = 18** Week 5 transfer runs on leaf-masked
images (seed **42** only). Run **only** variants cleared by the Week 12 audit
gate (`notes/mask_audit/audit_decision.md`).

| Condition | Local mask root | Output CSV |
|-----------|-----------------|------------|
| `sam_leaf` | `data/masked/sam_leaf/` | `frozen_results_v2/crossdataset_matrix_masked_sam_leaf.csv` |
| `hsv_leaf` | `data/masked/hsv_leaf/` | `frozen_results_v2/crossdataset_matrix_masked_hsv_leaf.csv` |

Protocol matches `run_crossdataset.py` (BN warmup, Adam 1e-3, class weights,
ReduceLROnPlateau, early stop on val macro-F1). Images are remapped
`data/raw` → `data/masked/{condition}`.

> Existing notebook `kaggle_week13.md` is the **augmentation bucket ablation**.
> This file is the **masking** Week 13 from the Week 12–13 plan.

## Prerequisites (local, before Kaggle)

1. Full Week 12 masks (5419 images each cleared variant).
2. Hand audit + `python scripts/parse_audit_verdicts.py`.
3. Upload `data/masked/{condition}/` as a Kaggle dataset (preserve relative
   paths under each dataset folder), **or** zip and attach as input.

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

!git checkout main

!pip install -q timm scikit-learn albumentations opencv-python-headless pandas matplotlib seaborn pillow imagehash pyyaml tqdm
```

## Cell 2 — Mount masked images

Adjust `MASKED_INPUT` to your uploaded dataset. Expected layout after copy:

```text
data/masked/sam_leaf/riceleafbd/...
data/masked/sam_leaf/dhan_shomadhan/...
data/masked/sam_leaf/brri_rice_disease_pest/...
```

```python
from pathlib import Path
import shutil

CONDITION = "sam_leaf"  # or hsv_leaf — only if audit cleared it
MASKED_INPUT = Path(f"/kaggle/input/your-masked-{CONDITION}")  # <-- edit
dest = Path(f"data/masked/{CONDITION}")
dest.mkdir(parents=True, exist_ok=True)

# If the upload is already the tree under sam_leaf/, copy contents in.
src = MASKED_INPUT
if (MASKED_INPUT / "riceleafbd").exists():
    src = MASKED_INPUT
elif (MASKED_INPUT / CONDITION / "riceleafbd").exists():
    src = MASKED_INPUT / CONDITION
else:
    raise FileNotFoundError(f"Cannot find dataset folders under {MASKED_INPUT}")

for name in ("riceleafbd", "dhan_shomadhan", "brri_rice_disease_pest"):
    s = src / name
    d = dest / name
    if not s.exists():
        raise FileNotFoundError(s)
    if d.exists():
        print("exists", d)
    else:
        shutil.copytree(s, d)
        print("copied", name)

print("files:", sum(1 for _ in dest.rglob("*") if _.is_file()))
```

## Cell 3 — Frozen manifest into working tree

`run_transfer_masked.py` copies `frozen_results/manifest.csv` into
`results_masked/{condition}/` itself. Ensure the frozen file is present:

```python
from pathlib import Path
assert Path("frozen_results/manifest.csv").exists()
assert Path("frozen_results/crossdataset_matrix.csv").exists()
print("frozen inputs OK")
```

If you also need the audit gate file on Kaggle, upload
`notes/mask_audit/audit_decision.md` or use `--allow-without-audit` only when
you have already confirmed the gate locally.

## Cell 4 — Dry run

```python
CONDITION = "sam_leaf"

!python run_transfer_masked.py --condition {CONDITION} --dry-run
# If audit_decision.md is not on the notebook filesystem yet:
# !python run_transfer_masked.py --condition {CONDITION} --dry-run --allow-without-audit
```

Expect 18 `PENDING` lines and a successful mask-path verification (5419 files).

## Cell 5 — Train (18 runs)

```python
!python run_transfer_masked.py --condition {CONDITION}
```

Resume-safe: completed rows in
`frozen_results_v2/crossdataset_matrix_masked_{CONDITION}.csv` are skipped;
existing checkpoints under `results_masked/{CONDITION}/checkpoints/` are reused.

## Cell 6 — Verify + zip

```python
from pathlib import Path
import pandas as pd

CONDITION = "sam_leaf"
out = Path(f"frozen_results_v2/crossdataset_matrix_masked_{CONDITION}.csv")
assert out.exists(), out
df = pd.read_csv(out)
print(df.shape)
print(df.head())
assert len(df) == 18, len(df)
assert set(df["condition"]) == {CONDITION}

!mkdir -p /kaggle/working/week13_masked_light
!cp frozen_results_v2/crossdataset_matrix_masked_{CONDITION}.csv /kaggle/working/week13_masked_light/
!cp -r results_masked/{CONDITION}/checkpoints /kaggle/working/week13_masked_light/checkpoints || true
!cd /kaggle/working && zip -r week13_masked_{CONDITION}_light.zip week13_masked_light
print("Download week13_masked_*_light.zip from Output")
```

## After download (local)

Place the CSV under `frozen_results_v2/`. Do **not** interpret deltas in code —
report numbers to the researcher for manuscript writing.

If both `sam_leaf` and `hsv_leaf` passed the Week 12 gate, run Cell 2–6 twice
(once per condition).
