# Week 6 retrospective — Diagnosis

Representative model: ResNet50 trained on Dhan-Shomadhan, shared classes
`brown_spot` and `tungro`, seed 42.

## Grad-CAM (qualitative)

The overlay grid (`paper/figures/gradcam_examples.png`) shows attention that is
frequently spread across neighbouring leaves, grass, and background rather than
staying on the visible lesion regions. Some correct predictions do focus on a
leaf, but the model does not consistently localize disease-relevant tissue.

Border-attention enrichment (normalized so a uniform heatmap = 1.0) did **not**
cleanly separate correct from incorrect predictions: both groups contain low-
and high-border cases. So Grad-CAM is supporting qualitative evidence, not proof
on its own.

## Background-confound (quantitative, the anchor result)

Same Dhan-trained ResNet50 evaluated on three test conditions:

| Condition          | n   | Accuracy | Macro-F1 |
|--------------------|-----|----------|----------|
| Dhan white-bg      | 29  | 0.862    | 0.854    |
| Dhan field-bg      | 22  | 0.727    | 0.705    |
| RiceLeafBD field   | 133 | 0.586    | 0.573    |

Brown-spot F1 degrades monotonically: 0.818 (white) -> 0.625 (field) -> 0.495
(cross-dataset field).

Prespecified support criterion (white > field AND white > cross) is **met**.
This supports the hypothesis that background/domain characteristics — not only
disease features — drive part of the cross-dataset generalization failure.

## Honest scope

- Single model, single seed, one transfer direction.
- Border-CAM is a proxy, not lesion segmentation.
- The ordering is descriptive; it does not by itself establish causality.

These points go directly into the paper's Discussion and Limitations.
