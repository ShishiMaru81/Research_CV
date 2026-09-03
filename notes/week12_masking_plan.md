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

Note: the script uses the official registry key `vit_t`. The brief's
`predictor.generate()` API does not match MobileSAM's published API.

### SAM selection criterion (deviation from brief, 2026-09-03)

The brief specified: automatic mask generation, keep the mask with the
highest mean ExG (2G − R − B). On the first Kaggle smoke image
(`riceleafbd/Bacterial Leaf Blight/287707627_...jpg`) this kept a thin
healthy green blade in the top-left corner (foreground fraction 0.05) and
discarded the brown blighted leaf in the centre. Second image: 0.001.

Cause: mean ExG is maximised by the greenest fragment. A diseased leaf is
discoloured, so the greenest region is always background canopy. Changing
mean → sum would only pick the largest green background blade. Greenness
cannot identify the subject leaf in disease imagery.

Replacement rule (no result-tuning, decided before any full run):
`SamPredictor` with a single positive point at the image centre,
`multimask_output=True`, keep the mask with the highest SAM predicted IoU,
then the largest connected component. Rationale: the subject leaf is
centred by the photographer in all three datasets. Side effect: one decoder
pass per image instead of the AMG point grid (~10× faster).

`sam_mask_quality.csv` gains a `sam_score` column (SAM's own IoU estimate).
The HSV variant still uses ExG/Otsu as specified; it is the crude control
arm and is expected to fail the audit on in-canopy RiceLeafBD.

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
