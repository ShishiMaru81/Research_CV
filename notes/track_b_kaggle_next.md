# Track B — Kaggle checkpoint verification (your next step)

**When:** After Track A integrity pass is pushed (local CPU work complete).  
**Where:** Kaggle GPU notebook — not local. Local has **0/144** `.pth` files.

## Goal

Confirm checkpoints referenced in CSVs/registry exist on Kaggle and optionally download:
- Diagnosis checkpoint: Dhan→RiceLeafBD ResNet50 (for future leaf-removal eval)
- Remaining **12** strong-aug seed-2024 cells (multi-seed closure)

## Procedure

Full cell-by-cell script: `notes/kaggle_checkpoint_verification.md`

Minimal verification (run in Kaggle after attaching your results dataset):

```python
from pathlib import Path
import pandas as pd

ROOT = Path("/kaggle/working/Research_CV")  # adjust to your extract path
reg = pd.read_csv(ROOT / "week11_light" / "run_registry.csv")

rows = []
for r in reg.itertuples():
    ckpt = Path(r.checkpoint_path)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    rows.append({
        "run_id": getattr(r, "run_id", r.Index),
        "checkpoint_path": str(ckpt),
        "exists": ckpt.is_file(),
        "status": getattr(r, "status", ""),
    })

out = pd.DataFrame(rows)
out.to_csv(ROOT / "checkpoint_verification.csv", index=False)
print(out["exists"].value_counts())
print("missing:", (~out["exists"]).sum(), "/", len(out))
```

## Success criteria

- Save `checkpoint_verification.csv` back to the repo under `notes/` or `week11_results/`
- At minimum: diagnosis ResNet50 checkpoint **exists**
- Optional: pull metrics/predictions for the 12 missing strong seed-2024 rows

## After verification

**Track C1 (local CPU):** `feasibility_check/` — segmentation QC on BRRI + Dhan-white (GrabCut).  
**Track C2 (Kaggle eval):** leaf-removal causal test on verified checkpoint.

No new training until checkpoints are confirmed.
