# Week 12 — Masking pipeline (CPU)

**Preamble confirmed:** `notes/data_rules.md` + `notes/project_file_map_and_navigation.md`.

## Goal

Build SAM and HSV leaf masks, sample 60 panels for hand audit, freeze the
decision gate **before** Week 13 GPU training.

## Run order

```bash
cd Research_CV

# Optional smoke tests (first N images)
python scripts/build_hsv_masks.py --limit 20
python scripts/build_sam_masks.py --limit 5   # needs MobileSAM + weights

# Full runs (~minutes for HSV, ~hours for SAM on CPU)
python scripts/build_hsv_masks.py
python scripts/build_sam_masks.py

# After both quality CSVs exist:
python scripts/sample_audit_images.py

# YOU fill notes/mask_audit/audit_sheet.csv (sam_verdict / hsv_verdict)
# Then:
python scripts/parse_audit_verdicts.py
```

## MobileSAM setup (Script 1)

```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
pip install scikit-image scipy tqdm pillow pandas numpy torch
```

Download `mobile_sam.pt` into `Research_CV/weights/mobile_sam.pt`.

Note: the script uses the official registry key `vit_t` with
`SamAutomaticMaskGenerator` (automatic masks). The brief's
`predictor.generate()` API does not match MobileSAM's published API.

## Hand audit

Open `notes/mask_audit/panel_001.png` … `panel_060.png`.

Fill `notes/mask_audit/audit_sheet.csv`:

| Field | Values |
|-------|--------|
| `sam_verdict` / `hsv_verdict` | `PASS` \| `PARTIAL` \| `FAIL` |
| `reason_code` | `edge_artifacts` \| `too_much_background` \| `too_little_leaf` \| `canopy_loss` \| `other` |
| `notes` | free text (optional) |

**Gate:** acceptable (PASS+PARTIAL) ≥ 80% **per dataset** per variant.

## Outputs

| Path | Role |
|------|------|
| `data/masked/sam_leaf/` | SAM-masked images |
| `data/masked/hsv_leaf/` | HSV-masked images |
| `frozen_results_v2/sam_mask_quality.csv` | SAM metrics |
| `frozen_results_v2/hsv_mask_quality.csv` | HSV metrics |
| `notes/mask_audit/audit_sheet.csv` | Hand audit |
| `notes/mask_audit/audit_decision.md` | Gate result |

## Week 13

Only after `parse_audit_verdicts.py` prints cleared variants. Do **not** start
Kaggle masked retraining until the gate passes.

```bash
# Local dry-run (needs full masks + audit decision)
python run_transfer_masked.py --condition sam_leaf --dry-run

# Full 18-run train (GPU strongly recommended)
python run_transfer_masked.py --condition sam_leaf
```

Kaggle cells: `notebooks/kaggle_week13_masked.md`  
(Do not confuse with `kaggle_week13.md`, which is the older aug-bucket ablation.)
