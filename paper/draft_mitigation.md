# Mitigation experiments (paper draft)

## Methods

We evaluated two standard mitigation strategies while holding the frozen data
splits, pretrained initialization, model architectures, and seed (42) fixed.
First, we replaced the default training transform with a stronger augmentation
pipeline comprising random resized cropping, horizontal flipping, affine
perturbation, brightness/contrast and hue/saturation variation, Gaussian blur,
and coarse dropout. Augmentation was applied only to training images; validation
and test preprocessing remained deterministic. We repeated all six ordered
cross-dataset transfer pairs for MobileNetV2, EfficientNet-B0, and ResNet50,
yielding 18 comparisons matched to the Week 5 baseline.

Second, we performed leave-one-dataset-out (LODO) training. For each held-out
dataset, the other two datasets were combined for training and the held-out
dataset was used only for evaluation. The label space was the largest
non-degenerate set present in the held-out dataset and at least one source:
{brown spot, healthy, tungro} for RiceLeafBD; {brown spot, rice blast, scald,
tungro} for Dhan-Shomadhan; and {healthy, rice blast, scald, tungro} for BRRI.
LODO used the default augmentation pipeline so that source composition was the
only changed experimental factor.

## Results

Strong augmentation improved cross-dataset macro-F1 in 14 of 18 matched runs.
The mean cross-dataset macro-F1 increased from 0.436 to 0.503 (mean improvement
0.067), and the mean generalization gap decreased from 0.387 to 0.333 (mean
reduction 0.055). Improvements were positive after averaging by every target:
+0.092 for BRRI, +0.064 for RiceLeafBD, and +0.045 for Dhan-Shomadhan.

ResNet50 benefited most overall, reaching mean cross-domain macro-F1 0.609 with
a mean gap of 0.244 under strong augmentation, compared with 0.457/0.373 for
EfficientNet-B0 and 0.444/0.381 for MobileNetV2. The largest individual gain
occurred for ResNet50 transferred from BRRI to RiceLeafBD: macro-F1 increased
from 0.300 to 0.716, while the gap fell by 0.343. Effects were nevertheless
heterogeneous. Four of 18 transfers declined, most notably MobileNetV2 from BRRI
to RiceLeafBD (0.573 to 0.273).

LODO was less reliable. It exceeded the baseline single-source aggregate in
3 of 9 target/model comparisons. The clearest improvement was EfficientNet-B0
on held-out Dhan-Shomadhan (0.317 to 0.489). Smaller gains occurred for
EfficientNet-B0 (0.474 to 0.541) and MobileNetV2 (0.549 to 0.574) on held-out
RiceLeafBD. Mean LODO macro-F1 was 0.500 for RiceLeafBD, 0.402 for
Dhan-Shomadhan, and 0.203 for BRRI. BRRI remained particularly difficult:
even its best LODO result was 0.267 (ResNet50).

## Interpretation

The augmentation results support the diagnosis that sensitivity to acquisition
and background characteristics contributes to the transfer gap. Perturbing
crop, color, geometry, blur, and visible context during training generally
improved performance on unseen collections, with the strongest and most
consistent gains for ResNet50. However, the remaining gaps and several negative
transfer effects show that augmentation did not eliminate domain dependence.

Combining two source datasets through LODO did not generally improve
generalization. This suggests that source diversity alone is insufficient when
the held-out collection differs in acquisition conditions, class appearance, or
class balance. The particularly weak BRRI LODO results reinforce its role as
the most challenging target domain in this study.

AdaBN (BatchNorm statistic recalibration on unlabeled target-train images)
likewise failed on average: mean Δ macro-F1 −0.055 with only 5 of 18 pairs
improving, and no ResNet50 gains. This is an honest weak/negative domain-
adaptation baseline on the same 18 matched transfers. Among the three simple
interventions tested, strong augmentation remains the most reliable.

The LODO and single-source aggregates use different label spaces and are
therefore strategy-level rather than class-matched comparisons. In addition,
all mitigation estimates use one seed and one frozen split. The results should
be reported as evidence about this benchmark, not as proof that one mitigation
will dominate across resampling or other rice-disease datasets.
