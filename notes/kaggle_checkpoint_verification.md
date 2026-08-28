# Kaggle checkpoint verification

**Local status (2026-08-26).** Zero `*.pth` / `*.pt` files under
`Research_CV/`. Of **144** unique `checkpoint_path` values referenced in
tracked CSVs, **0** exist on this machine. `week11_light/run_registry.csv`
lists **90/90** multi-seed jobs as `complete` with Kaggle wall-clock
timestamps, but the checkpoint and most prediction files were never
downloaded.

`week11_light/predictions/` currently holds **19** prediction CSVs
(enough for partial bootstrap CIs only).

## What the CSVs claim

- Week 4–7 seed-42 checkpoints: paths like
  `results/checkpoints/{model}__…__seed42.pth` in
  `frozen_results/crossdataset_matrix*.csv`, `lodo_results.csv`,
  `indataset_results.csv`.
- Multi-seed (seeds 7, 2024): paths in `week11_light/run_registry.csv`
  with `status=complete` for all 90 transfer/LODO jobs.

## How to verify on Kaggle (run there, not locally)

Attach your Week-11 results dataset / Output zip, then:

```python
from pathlib import Path
import pandas as pd

ROOT = Path("/kaggle/working/Research_CV")  # or extract path
reg = pd.read_csv(ROOT / "week11_light" / "run_registry.csv")
# If using a restored results/ tree instead:
# reg = pd.read_csv("/kaggle/working/results/run_registry.csv")

rows = []
for r in reg.itertuples():
    ckpt = Path(r.checkpoint_path)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
        if not ckpt.exists():
            ckpt = Path("/kaggle/working") / r.checkpoint_path
    pred = Path(str(r.predictions_path)) if pd.notna(r.predictions_path) else None
    if pred is not None and not pred.is_absolute():
        cand = ROOT / pred
        pred = cand if cand.exists() else Path("/kaggle/working") / pred
    rows.append({
        "run_id": r.run_id,
        "status": r.status,
        "ckpt_exists": ckpt.exists(),
        "ckpt_bytes": ckpt.stat().st_size if ckpt.exists() else 0,
        "pred_exists": bool(pred and pred.exists()),
    })

df = pd.DataFrame(rows)
print(df.groupby("status")[["ckpt_exists", "pred_exists"]].sum())
print("ckpt missing:", (~df.ckpt_exists).sum(), "/", len(df))
print("pred missing:", (~df.pred_exists).sum(), "/", len(df))
df.to_csv("/kaggle/working/checkpoint_verification.csv", index=False)
```

Also spot-check that a frozen seed-42 path referenced in
`frozen_results/crossdataset_matrix.csv` resolves under the restored
`results/checkpoints/` tree (Week-7 bundle may still be absent — already
noted in the Week-8 freeze limitations).

## Honest paper / reproducibility statement

Until the Kaggle check above is run and the CSV is committed:

> Multi-seed and mitigation **metrics** are archived locally; model
> **checkpoints** remain on the Kaggle working tree / Output artifacts and
> are not present in this clone. Frozen seed-42 metrics were validated
> against per-run metrics JSON at Week-8 freeze time; Week-7 `.pth` files
> were already listed as absent from the downloaded bundle in
> `frozen_results/audit_report.md`.

Do not claim “checkpoints verified” until
`checkpoint_verification.csv` shows `ckpt_exists` for the claimed set.
