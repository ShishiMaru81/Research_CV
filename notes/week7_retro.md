# Week 7 retrospective — Mitigation

Week 7 tested two interventions against the Week 5 single-source baseline:
strong train-only augmentation and leave-one-dataset-out (LODO) training.
All experiments used seed 42 and the frozen dataset splits.

## Strong augmentation

Across the 18 matched transfer/model comparisons, strong augmentation increased
cross-dataset macro-F1 in 14 cases and reduced the generalization gap in 13.
Mean cross-dataset macro-F1 rose from 0.436 to 0.503 (mean change +0.067), while
the mean generalization gap fell from 0.387 to 0.333 (mean reduction 0.055).

The average improvement was positive for every target dataset:

| Target dataset | Mean macro-F1 change |
|----------------|---------------------:|
| BRRI           | +0.092 |
| RiceLeafBD     | +0.064 |
| Dhan-Shomadhan | +0.045 |

ResNet50 was the strongest augmented model overall, with mean cross-domain
macro-F1 0.609 and mean gap 0.244. Its largest pairwise gain was BRRI to
RiceLeafBD: macro-F1 increased from 0.300 to 0.716 (+0.416), and the gap
decreased by 0.343. Strong augmentation was not universally beneficial. The
largest regression was MobileNetV2 on BRRI to RiceLeafBD (0.573 to 0.273).

## Leave-one-dataset-out training

LODO improved over the corresponding baseline single-source aggregate in only
3 of 9 target/model comparisons:

- EfficientNet-B0 on Dhan-Shomadhan: 0.317 to 0.489 (+0.171).
- EfficientNet-B0 on RiceLeafBD: 0.474 to 0.541 (+0.067).
- MobileNetV2 on RiceLeafBD: 0.549 to 0.574 (+0.025).

Mean LODO macro-F1 by held-out target was 0.500 for RiceLeafBD, 0.402 for
Dhan-Shomadhan, and 0.203 for BRRI. BRRI was therefore the hardest unseen
domain. Its best LODO result was only 0.267 macro-F1 (ResNet50).

LODO and single-source scores are not strictly matched comparisons because the
LODO label spaces differ from the pairwise shared-class spaces. These results
should be interpreted as strategy-level evidence, not as controlled per-class
effects.

## Conclusion

Strong augmentation was the more reliable mitigation: it improved most matched
transfer runs and reduced the average generalization gap. LODO was inconsistent
and usually underperformed the single-source aggregate. This negative result
suggests that adding source-domain diversity does not automatically solve the
shift to an unseen collection, especially for BRRI. All conclusions remain
limited to one frozen split and one seed.

## Outputs

- `week7_results/crossdataset_matrix_aug.csv`
- `week7_results/generalization_gap_aug.csv`
- `week7_results/lodo_results.csv`
- `week7_results/mitigation_pairwise_aug.csv`
- `week7_results/mitigation_comparison.csv`
