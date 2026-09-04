# Freeze v2 changelog

**Purpose.** Document the replacement freeze (`frozen_results_v2/`) required by
the post-review revision. The original `frozen_results/` directory is
**immutable** and is never edited.

**How to refresh.** After Phase 1–3 artifacts are complete locally:

```bash
python -m run_stats
python -m freeze_results_v2
```

---

## What changed vs v1

| Artifact | v1 | v2 |
|----------|----|----|
| Seed-42 core matrices / gaps / LODO / confound | Present | Bit-copied from v1 (identity check) |
| Multi-seed transfer / LODO aggregates | Absent | Added from `week11_results/multiseed/` |
| AdaBN matched evaluations | Absent | Added from `adabn_results.csv` |
| Augmentation bucket ablation | Absent | Added from `results/ablation/` when available |
| Statistics layer | Absent | Added from `results/stats/` |

## Why

Review items #1 (multi-seed), #3 (AdaBN), #5/#8 (stats), and #6 (ablation)
require new rows that did not exist at the Week-8 freeze. Appending into
`frozen_results/` would violate the freeze policy in `data_rules.md`.

## Provenance notes

- **Split seed** remains 42 for all revision experiments; only **train seed**
  varies (42 / 7 / 2024).
- Seed-42 baseline and strong-aug **files** in v2 core are bit-copied from
  v1; checks confirm file-copy / SHA-256 integrity after `shutil.copy2`,
  **not** independent numerical re-derivation from training logs.
- Incomplete overlays (e.g. missing seed-2024 strong-aug cells, missing
  ablation) are recorded as **warnings**, not silent drops.
- Git commit strings stamped in older manifests may no longer resolve after
  the 2026-08-02 co-author-trailer history rewrite; see
  `notes/missing_commit_investigation.md`. Use CSV SHA-256 digests as the
  integrity ground truth.

## Commit hashes

Record after each successful freeze:

| Freeze | Git commit | Date | Notes |
|--------|------------|------|-------|
| v1 content | SHA-256 digests in `frozen_results/freeze_manifest.json` | Week 8 | Immutable CSVs |
| v1 commit (current rewrite) | `ac77dc2` (“Add Week 8 result freeze…”) | 2026-07-16 / rewritten 2026-08-02 | Replaces obsolete `88f8c5d` |
| Freeze stamp in manifest | originally `13b8552` (HEAD when freeze ran) | Week 8 generation | Now `008d2cd` after rewrite |
| v2 | `4ea9b46` tip at last freeze refresh; re-run `freeze_results_v2` after commit | revision | Replacement; copy/hash wording |

## Reviewer-facing sentence

> Publication CSVs for the original seed-42 Week 5–7 release remain the
> immutable files under `frozen_results/`, integrity-checked by recorded
> SHA-256 digests. Multi-seed, AdaBN, ablation, and inferential statistics
> are released under a documented replacement freeze (`frozen_results_v2/`)
> that **copies** the v1 core files unchanged (copy/hash verification) and
> adds revision overlays with an explicit changelog. Independent numerical
> recomputation checks are provided by
> `python scripts/numerical_freeze_audit.py`.

---

## 2026-09-04 — Week 14 audit refresh

This refresh extends the replacement freeze through Weeks 10–14 without
modifying any file in `frozen_results/`.

### Added artifacts

- `dinov2_indataset.csv` and `dinov2_crossdataset.csv`
- `sam_mask_quality.csv`
- `crossdataset_matrix_masked_sam_leaf.csv`
- `adabn_labelshift.csv`

HSV mask-quality and masked-transfer artifacts were not present, so the audit
records `hsv_leaf` as **not run** rather than manufacturing a result.

### Audit changes

- Replaced the former delete/copy/self-compare workflow with an audit that reads
  the Week-8 manifest first and aborts on any v1 file-set or SHA-256 mismatch.
- Inventories every CSV in the v2 directory, including the previously
  unmanifested deployment and Grad-CAM summary tables.
- Checks finite metric ranges and positive sample counts on every specified row.
- Checks masked transfer completeness by the exact
  `(train_dataset, test_dataset, model, classes, seed, condition)` key set, not
  by row count alone.
- Records 26 v2 CSVs (11,327 total rows) and 18/18 SAM masked-transfer keys.

### AdaBN label-shift result

The analysis mirrors the executed AdaBN run's target-**train** adaptation set.
AdaBN deltas were negative in 13/18 cells, not uniformly negative. No tested
label-prior divergence metric had a significant correlation in the predicted
negative direction; the mechanism therefore remains unresolved. The report
also flags repeated domain-pair predictors and the fact that BN-layer count is
not a validated proxy for model depth.
