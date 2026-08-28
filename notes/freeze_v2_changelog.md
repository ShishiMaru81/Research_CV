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
