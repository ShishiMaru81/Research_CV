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
- Seed-42 baseline and strong-aug point estimates in v2 core files must match
  v1 within numeric tolerance (enforced by `freeze_results_v2.py`).
- Incomplete overlays (e.g. missing seed-2024 strong-aug cells, missing
  ablation) are recorded as **warnings**, not silent drops.

## Commit hashes

Record after each successful freeze:

| Freeze | Git commit | Date | Notes |
|--------|------------|------|-------|
| v1 | *(see `frozen_results/freeze_manifest.json`)* | Week 8 | Immutable |
| v2 | *(fill after `python -m freeze_results_v2`)* | Week 14 | Replacement |

## Reviewer-facing sentence

> Publication numbers for the original seed-42 Week 5–7 release remain those
> audited in `frozen_results/`. Multi-seed, AdaBN, ablation, and inferential
> statistics are released under a documented replacement freeze
> (`frozen_results_v2/`) that reproduces the v1 seed-42 files unchanged and
> adds revision overlays with an explicit changelog.
