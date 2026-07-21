# Week 9 retrospective — Paper draft

## Outcome

A complete venue-agnostic manuscript draft is written at
`paper/manuscript.md`. It covers Abstract through Conclusion using only frozen
Week 8 numbers, figures `fig01`–`fig09`, and paper tables. No models were
retrained and no frozen CSV was edited.

## Structure

1. Abstract — three-layer contribution and headline numbers
2. Introduction — problem framing
3. Related work — brief, non-exhaustive
4. Datasets and preprocessing — harmonization, splits, pairs
5. Methods — models, benchmark, diagnosis, mitigation, freeze
6. Results — in-dataset, transfer/gap, confound, augmentation, LODO
7. Discussion — practical implications and negative LODO result
8. Limitations — seed/split, Grad-CAM overlay gap, strategy-level LODO
9. Conclusion and reproducibility checklist

## Headline claims (checksummed against `table_summary_stats.csv`)

- Mean cross macro-F1: 0.436 → 0.503 under strong augmentation
- Mean gap: 0.387 → 0.333
- Augmentation improves 14/18 pairs; LODO improves 3/9 strategy cells
- Background confound ordering (white > field > cross) met for the
  representative ResNet50 Dhan → RiceLeafBD setting

## Remaining for Week 10

- Choose venue and convert Markdown → LaTeX/Word template
- Expand Related Work with full citations
- Optionally restore Grad-CAM overlay figure from Kaggle Week 6 bundle
- Author list, acknowledgments, funding, ethics, and conflict statements
- Final proofread, preprint (e.g. arXiv), and/or submission
