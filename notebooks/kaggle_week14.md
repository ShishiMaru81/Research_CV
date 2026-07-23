# Week 14 — Statistics, freeze v2, figures (Phase 4)

No GPU required. Run locally after Phase 1–3 artifacts are on disk.

## 1. Statistics layer

```bash
python -m run_stats
```

Optional smoke test (few prediction CSVs):

```bash
python -m run_stats --bootstrap_limit 5 --n_boot 200
```

Writes:

| File | Contents |
|------|----------|
| `results/stats/stats_tests.csv` | Wilcoxon signed-rank tests |
| `results/stats/seed_variance.csv` | mean ± std per cell across seeds |
| `results/stats/bootstrap_ci.csv` | per-run macro-F1 bootstrap CIs |
| `results/stats/STATS_SUMMARY.md` | headline noise-floor paragraph |

### Tests wired

1. Augmentation > baseline (Wilcoxon on 18 cell-mean deltas over seeds)
2. Same, all seed-pair rows
3. Same, seed-42 only
4. AdaBN Δ vs 0 (18 pairs, seed 42) if `adabn_results.csv` present
5. Bootstrap over test samples for every discoverable `predictions/*.csv`

## 2. Freeze v2

```bash
python -m freeze_results_v2
```

- Never touches `frozen_results/`
- Copies v1 core with identity checks
- Overlays multi-seed / AdaBN / ablation / stats when present
- Writes `frozen_results_v2/AUDIT_REPORT_v2.md` + `freeze_manifest_v2.json`
- Update commit hash table in `notes/freeze_v2_changelog.md`

## 3. Revision figures

```bash
python -m make_figures --revision
```

Adds (when inputs exist):

- `fig10_aug_paired_seed.png` — baseline vs strong with per-seed points
- `fig11_ablation_buckets.png` — ResNet50 bucket bars
- `fig12_adabn_delta.png` — AdaBN Δ by pair
- `fig13_seed_std_heatmap.png` — across-seed std of baseline cross F1

Core Week-8 figures still regenerate with:

```bash
python -m make_figures
```

## 4. Definition of done

- [ ] `stats_tests.csv`, `seed_variance.csv`, `bootstrap_ci.csv`
- [ ] `frozen_results_v2/` audit PASS or PASS_WITH_WARNINGS (documented)
- [ ] Revision figures present under `paper/figures/`
- [ ] Changelog commit hash filled
- [ ] Tag: `week14-freeze-v2`
