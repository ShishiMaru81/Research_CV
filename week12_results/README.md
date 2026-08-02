# Week 12 — AdaBN results

Adaptive BatchNorm on 18 seed-42 baseline transfer checkpoints.

| Artifact | Path |
|----------|------|
| Full run CSV | `adabn/adabn_results.csv` |
| Publication table | `adabn/table_adabn.csv` (also `paper/tables/table_adabn.csv`) |
| By-model summary | `adabn/table_adabn_by_model.csv` |
| Checkpoints (local) | `../week12_baselines_and_adabn/checkpoints/` |
| Manuscript | `paper/manuscript.md` §4.5, §5.6, Discussion |

## Headline

- Mean Δ macro-F1 (AdaBN − matched baseline): **−0.055**
- Pairs improved: **5 / 18**
- ResNet50: never improves (mean Δ −0.119)

Rebuild tables: `python scripts/build_adabn_tables.py`
