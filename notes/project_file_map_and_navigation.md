# Project file map and navigation (Research_CV)

Standing map for Week 12+ masking / transfer work. Verify paths before use.

## Authoritative inputs (read-only)

| Path | Role |
|------|------|
| `frozen_results/manifest.csv` | Frozen splits + paths (5419 rows; do not regenerate) |
| `frozen_results_v2/manifest.csv` | Bit-copy of v1 for v2 freeze overlays |
| `notes/data_rules.md` | Non-negotiable data rules |

## Week 12 masking outputs (writable)

| Path | Role |
|------|------|
| `data/masked/sam_leaf/` | SAM-masked RGB images (mirror of `data/raw/` tree) |
| `data/masked/hsv_leaf/` | HSV/ExG-masked RGB images |
| `frozen_results_v2/sam_mask_quality.csv` | Per-image SAM quality metrics |
| `frozen_results_v2/hsv_mask_quality.csv` | Per-image HSV quality metrics |
| `notes/mask_audit/` | Manual audit panels + `audit_sheet.csv` |
| `notes/mask_audit/audit_decision.md` | Gate result after hand audit |

## Week 13 (Kaggle; later)

| Path | Role |
|------|------|
| `run_transfer_masked.py` | Retrain transfer matrix on masked images |
| `frozen_results_v2/crossdataset_matrix_masked*.csv` | Masked transfer results |

## Image path convention

Manifest `image_path` values look like:
`data\raw\riceleafbd\<class>\<file>.jpg`

Masked outputs preserve the relative suffix under:
`data/masked/{sam_leaf|hsv_leaf}/riceleafbd/<class>/<file>.jpg`
