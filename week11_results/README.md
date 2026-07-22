# Week 11 results — multi-seed Phase 1

Artifacts from the 90-run multi-seed campaign (train seeds 7 and 2024;
split_seed fixed at 42). Seed-42 baselines remain in `frozen_results/`.

## Status

- `run_registry.csv` — **90/90 complete**
- `multiseed/transfer_all_seeds.csv` — reconstructed metrics (logs + frozen)
- `multiseed/lodo_all_seeds.csv` — full 3-seed LODO
- `multiseed/PHASE1_MULTISEED_SUMMARY.md` — headline mean±std

**Gap:** 12 seed-2024 strong-augmentation transfer metric rows are still
missing locally. Session-2 partial CSVs are kept under
`multiseed/*_session2_partial.csv` (5 rows only). Re-download the full
`crossdataset_matrix_aug.csv` from Kaggle when available, then re-run:

```bash
python scripts/_extract_by_ckpt.py
python scripts/rebuild_multiseed_summary.py
```

## Do not confuse

Root-level `crossdataset_matrix_aug.csv` downloads that contain only seed 2024
and 5 rows are **not** the full matrix.
