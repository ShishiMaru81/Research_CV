# Numerical freeze audit

These checks **recompute** statistics from frozen CSVs / multi-seed tables
and compare to stored aggregates. Unlike `freeze_results_v2` (copy/hash
integrity), a wrong number here will FAIL.

**Result: 12/12 PASS**

| Check | Expected | Actual | Status | Detail |
|-------|----------|--------|--------|--------|
| generalization_gap.csv gap arithmetic | 0 | 2.22045e-16 | PASS | max |recomputed − stored| = 2.220e-16 |
| generalization_gap_aug.csv gap arithmetic | 0 | 1.66533e-16 | PASS | max |recomputed − stored| = 1.665e-16 |
| seed-42 default cross_macro_f1 == frozen | 0 | 0 | PASS | max abs diff = 0.000e+00 (n=18) |
| seed-42 strong cross_macro_f1 == frozen | 0 | 0 | PASS | max abs diff = 0.000e+00 (n=18) |
| transfer_cell_mean_std cross_mean recomputes | 0 | 5.55112e-17 | PASS | max abs diff = 5.551e-17 |
| Wilcoxon mean_delta matches stats_tests | 0.0634855 | 0.0634855 | PASS | stored=0.063486 recomputed=0.063486 |
| Wilcoxon pvalue matches stats_tests | 0.00769043 | 0.00769043 | PASS | stored=0.00769043 recomputed=0.00769043 |
| Wilcoxon n_positive matches stats_tests | 14 | 14 | PASS | stored=14 recomputed=14 |
| Option A baseline cross mean ≈ 0.445 | 0.445 | 0.445034 | PASS | actual=0.445034 |
| Option A strong cross mean ≈ 0.502 | 0.502 | 0.502145 | PASS | actual=0.502145 |
| mitigation_pairwise baseline matches matrix | 0 | 0 | PASS | max abs diff = 0.000e+00 |
| mitigation_pairwise aug matches matrix | 0 | 0 | PASS | max abs diff = 0.000e+00 |

## Re-run

```bash
python scripts/rebuild_multiseed_summary.py
python scripts/build_multiseed_tables.py --sync-week11
python -m run_stats
python scripts/numerical_freeze_audit.py
```

