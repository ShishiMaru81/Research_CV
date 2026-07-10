# Week 6 — Failure diagnosis

## Representative experiment

- Model: ResNet50
- Train: Dhan-Shomadhan
- Test: RiceLeafBD
- Shared classes: brown_spot, tungro
- Seed: 42

This pair directly supports the background-confound test because Dhan contains
both field and white-background images while RiceLeafBD is field imagery.

## Grad-CAM

- Sample correct and incorrect RiceLeafBD predictions across both classes.
- Target each prediction's predicted class.
- Save labeled overlays to `paper/figures/gradcam_examples.png`.
- Save sample metadata and border-attention proxy to
  `results/gradcam_records.csv`.
- Human review decides whether overlays are biologically sensible.

Border enrichment is normalized so a uniform heatmap equals 1.0. It is not
lesion segmentation and must not be presented as direct lesion-vs-background
localization.

## Background-confound experiment

Evaluate the same Dhan-trained shared-class checkpoint on:

1. Dhan field-only test (`n=22` in the frozen split)
2. Dhan white-background-only test (`n=29`)
3. RiceLeafBD field test (`n=133`)

Report accuracy, macro-F1, per-class F1, and sample counts in
`results/background_confound.csv`.

Prespecified descriptive support criterion:

`Dhan white macro-F1 > Dhan field macro-F1` and
`Dhan white macro-F1 > RiceLeafBD field macro-F1`.

This ordering supports the confound hypothesis but does not alone establish
causality.

## Checkpoint handling

The downloaded Week 5 bundle did not contain `.pth` files. On Kaggle,
`run_diagnosis.py` reuses the Week 5 checkpoint if restored; otherwise it
retrains only the ResNet50 Dhan→Rice shared-class run.

## Definition of done

- Grad-CAM grid generated and manually inspected
- Background comparison CSV and bar chart generated
- Interpretation recorded without overclaiming
- Week 6 result bundle downloaded locally

