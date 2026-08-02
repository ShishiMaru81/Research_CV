# Phase 3 ablation interpretation

Source: `D:/Research_Own/Research_CV/results/ablation/augmentation_ablation.csv`

## Ranking (mean Δ macro-F1 vs ResNet50 seed-42 baseline)

- **Geometric**: mean Δ = +0.085 (mean cross F1 = 0.567; 3/6 pairs improve)
- **Occlusion**: mean Δ = +0.030 (mean cross F1 = 0.512; 5/6 pairs improve)
- **Photometric**: mean Δ = +0.026 (mean cross F1 = 0.508; 3/6 pairs improve)

## One-paragraph claim

On ResNet50 × six transfer pairs (seed 42), the **Geometric** bucket alone recovers the largest share of the strong-augmentation signal (mean Δ vs baseline +0.085; next is Occlusion at +0.030). This scopes the Week 7 bundled pipeline to a mechanism-level comparison; residual gaps relative to the full strong stack indicate that buckets are complementary rather than fully redundant.
