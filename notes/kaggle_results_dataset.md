# Kaggle results dataset — what to upload and attach

There is **no pre-made public results dataset** in the repo. You create one on Kaggle
by uploading a zip, then attach it as **Add Input** in your notebook.

---

## Option A — Use the zip we packaged locally (partial)

**File on your machine:**

```
d:\Research_Own\Research_CV\dist\week11_results.zip
```

**Contains (from `week11_light/`):**
- `run_registry.csv` (90 runs, all `complete`)
- `manifest.csv`
- `predictions/` — 19 CSVs (partial)
- `multiseed/` — 3 CSVs
- `checkpoints/` — **empty** (no `.pth` files were ever downloaded)

**Honest limit:** Good for registry verification and partial bootstrap CIs.
**Not enough** for AdaBN, diagnosis re-eval, or leaf-removal without checkpoints.

### Upload to Kaggle

1. Kaggle → **Datasets** → **New Dataset**
2. Upload `dist/week11_results.zip`
3. Title suggestion: `research-cv-week11-results-partial`
4. Set visibility (Private is fine)

### Attach in notebook

After publishing, Kaggle shows the dataset slug. Use:

```python
RESTORE_ZIP = Path("/kaggle/input/datasets/<your-username>/research-cv-week11-results-partial/week11_results.zip")
```

Example if your username is `happychamp`:

```python
RESTORE_ZIP = Path("/kaggle/input/datasets/happychamp/research-cv-week11-results-partial/week11_results.zip")
```

Then run the restore cell from `notebooks/kaggle_week11.md` Cell 3.

---

## Option B — Full results (what you actually need for Track B+)

If you still have an **old Kaggle notebook session**, go to:

**Notebook → Output** (right panel) and look for:

- `week11_results.zip` or `week11_bundle/`
- `week5_progress.zip`
- `results/checkpoints/*.pth`

Download that, upload as a **new Kaggle dataset** (e.g. `research-cv-checkpoints-full`),
and attach:

```python
RESTORE_ZIP = Path("/kaggle/input/datasets/happychamp/research-cv-checkpoints-full/week11_results.zip")
```

A full bundle should have **~144 `.pth` files** under `checkpoints/`.

---

## Image datasets (separate from results)

These are **not** the results dataset — they are raw images for training/eval:

| Add Input dataset | Path inside notebook |
|---|---|
| riceleafbd | `/kaggle/input/datasets/happychamp/research/riceleafbd/riceleafbd` |
| dhan_shomadhan | `/kaggle/input/datasets/happychamp/research/dhan_shomadhan/dhan_shomadhan` |
| brri | `/kaggle/input/datasets/happychamp/research/brri_rice_disease_pest/brri_rice_disease_pest` |

---

## Diagnosis checkpoint filename (when you have ckpts)

Dhan → RiceLeafBD ResNet50, seed 42 (Week 6 / confound):

```
results/checkpoints/resnet50__train-dhan_shomadhan__run-to-riceleafbd__classes-brown_spot+tungro__seed42.pth
```

Week 5 also referenced a single-file dataset:

```
/kaggle/input/week5-checkpoint/resnet50__train-dhan_shomadhan__run-to-riceleafbd__classes-brown_spot+tungro__seed42.pth
```

If that Kaggle dataset still exists on your account, attach **week5-checkpoint** as input
for the fastest path to one diagnosis checkpoint.

---

## Rebuild the zip locally

```bash
python scripts/package_week11_results.py
# custom source:
python scripts/package_week11_results.py --src path/to/results --out dist/week11_results.zip
```
