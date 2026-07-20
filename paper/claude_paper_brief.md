# CLAUDE PAPER BRIEF — Full Context for Writing the Manuscript

**Purpose:** Give this single file to Claude (or any writing model) so it can produce a complete research paper without needing the rest of the repo.

**Repo:** https://github.com/ShishiMaru81/Research_CV (private)  
**Topic:** Cross-dataset generalization of Bangladeshi rice leaf disease classifiers  
**Contribution type:** Application-plus-insight (NOT a new method paper)  
**Seed:** 42 everywhere  
**Status:** Experiments Weeks 1–8 frozen; draft exists at `paper/manuscript.md`

---

## 0. Instructions for Claude (read first)

### What to write
Produce a complete academic paper in Markdown (or LaTeX if asked) with:

1. Title  
2. Abstract (150–250 words)  
3. Introduction  
4. Related Work (expand with real citations; placeholders OK if unsure)  
5. Datasets and Preprocessing  
6. Methods  
7. Results (with tables/figures referenced)  
8. Discussion  
9. Limitations  
10. Conclusion  
11. Optional: Reproducibility / Data Availability  

### Hard rules
- **Do not invent numbers.** Use only the tables and headline stats in this file.
- **Do not claim a new architecture or SOTA method.** Frame as benchmark + diagnosis + mitigation.
- **Do not overclaim causality** from Grad-CAM or the background confound; call them supportive / descriptive.
- **LODO vs single-source is strategy-level**, not matched-class (different label spaces).
- Round display numbers to 3 decimals unless quoting exact table values.
- Primary metric is **macro-F1**.
- Prefer honest negative results (LODO often fails) over hype.
- Tone: clear, technical, publication-ready; avoid marketing language.

### Three-layer contribution (must remain first-class)
1. **Benchmark:** cross-dataset transfer matrix + generalization gap  
2. **Diagnosis:** Grad-CAM + background-confound experiment  
3. **Mitigation:** strong augmentation vs LODO (one variable at a time)

### Suggested title options
- Cross-Dataset Generalization of Bangladeshi Rice Leaf Disease Classifiers: Benchmark, Diagnosis, and Mitigation
- When In-Dataset Accuracy Misleads: Cross-Collection Transfer of Rice Leaf Disease Models on Bangladeshi Datasets

### Figure assets (attach these PNGs if available)
Located under `paper/figures/` in the repo:

| Fig | File | Caption focus |
|-----|------|---------------|
| 1 | `fig01_indataset_macro_f1.png` | In-dataset macro-F1 by dataset × model |
| 2 | `fig02_crossdataset_heatmap_baseline.png` | Baseline cross-dataset macro-F1 heatmaps |
| 3 | `fig03_generalization_gap_baseline.png` | Baseline generalization gaps |
| 4 | `fig04_background_confound.png` | White vs field vs cross-dataset field |
| 5 | `fig05_crossdataset_heatmap_aug.png` | Strong-aug transfer heatmaps |
| 6 | `fig06_generalization_gap_aug.png` | Strong-aug gaps |
| 7 | `fig07_augmentation_f1_delta.png` | Per-pair augmentation Δ macro-F1 |
| 8 | `fig08_mitigation_by_target.png` | Baseline vs aug vs LODO by target |
| 9 | `fig09_lodo_heldout.png` | LODO held-out macro-F1 |

Optional: Grad-CAM overlay from Week 6 Kaggle bundle (may be missing locally).

---

## 1. Problem statement (for Introduction)

Deep learning rice-disease classifiers often report high accuracy when train and test images come from the **same** collection. In Bangladesh there are multiple independent collections (citizen/field apps, research datasets, BRRI station data) that differ in camera, background (field vs white/studio), class inventory, and labeling. A model that looks strong in-dataset may fail on another Bangladeshi source.

**Research questions**
1. How large is the gap between in-dataset and cross-dataset macro-F1?
2. Do background / acquisition characteristics contribute to failure?
3. Do simple mitigations (strong augmentation; multi-source LODO) close the gap?

---

## 2. Datasets and preprocessing (facts)

### Collections (originals only)
| Dataset | Images used | Notes |
|---------|------------:|-------|
| RiceLeafBD | 1,560 | Field-oriented |
| Dhan-Shomadhan | 1,106 | Field + white background |
| BRRI Rice Leaf Disease and Pest | 2,753 | Originals only; augmented archive excluded |
| **Total harmonized** | **5,419** | After exclusions |

### Non-negotiable data rules
- Originals only (no pre-augmented BRRI images)
- Stratified 70/15/15 train/val/test per dataset×class, **seed 42**, frozen
- Cross-dataset perceptual-hash dedup (threshold 2): **0 true duplicates**
- BRRI folder `Rice` (16 images): **excluded** (ambiguous / not an official disease class)

### Canonical label taxonomy
healthy, bacterial_leaf_blight, brown_spot, tungro, rice_blast, scald, sheath_blight, leaf_folder, insect, stripes  

Spelling variants mapped (examples): Browon Spot→brown_spot; Rice Turgro→tungro; Shath Blight→sheath_blight.

### Shared-class transfer pairs (6 ordered directions)
| Pair | Classes |
|------|---------|
| RiceLeafBD ↔ Dhan-Shomadhan | brown_spot, tungro |
| Dhan-Shomadhan ↔ BRRI | rice_blast, scald, tungro |
| RiceLeafBD ↔ BRRI | healthy, tungro |

Typical frozen test sizes for shared subsets:
- brown_spot|tungro: RiceLeafBD 133, Dhan 51  
- healthy|tungro: RiceLeafBD 117, BRRI 127  
- rice_blast|scald|tungro: Dhan 103, BRRI 204  

Full-dataset test sizes (in-dataset baselines): RiceLeafBD 235, Dhan 167, BRRI 414.

---

## 3. Models and training protocol

**Backbones:** MobileNetV2 (`mobilenetv2_100`), EfficientNet-B0, ResNet50  
**Init:** ImageNet via `timm`  
**Train:** Adam lr=0.001, batch 32, image 224, class-weighted CE, 3-epoch head freeze then full FT, max 30 epochs, early stop on val macro-F1 (patience 7), AMP on GPU, seed 42.

**Default aug:** mild geometry/color.  
**Strong aug (train only):** RandomResizedCrop(0.6–1.0), HorizontalFlip, Affine, RandomBrightnessContrast, HueSaturationValue, GaussianBlur, CoarseDropout. Eval always deterministic.

**Generalization gap:**
`gap = source_test_macro_F1 − target_test_macro_F1` on the shared-class subset.

---

## 4. Headline statistics (must appear in Abstract / Results)

### In-dataset (9 runs: 3 models × 3 datasets)
- Overall mean macro-F1: **0.719**
- By dataset (models pooled): RiceLeafBD **0.901**, BRRI **0.638**, Dhan **0.618**
- By model (datasets pooled): ResNet50 **0.734**, EfficientNet-B0 **0.726**, MobileNetV2 **0.697**

### Baseline transfer (18 runs: 6 pairs × 3 models)
- Mean source (matched-subset) macro-F1: **0.824**
- Mean cross-dataset macro-F1: **0.436**
- Mean generalization gap: **0.387** (range ≈ 0.195–0.629)

### Diagnosis — background confound (ResNet50, train Dhan, classes brown_spot|tungro)
| Condition | n | Accuracy | Macro-F1 |
|-----------|--:|---------:|---------:|
| Dhan white-bg | 29 | 0.862 | **0.854** |
| Dhan field-bg | 22 | 0.727 | **0.705** |
| RiceLeafBD field | 133 | 0.586 | **0.573** |

Brown-spot F1: 0.818 (white) → 0.625 (field) → 0.495 (cross).  
Prespecified criterion white > field AND white > cross: **MET**.

Grad-CAM (qualitative): attention often spills to neighboring leaves/grass/background; border-attention enrichment does **not** cleanly separate correct vs incorrect. Supporting evidence only.

### Strong augmentation (18 matched pairs)
- Mean cross F1: 0.436 → **0.503** (Δ **+0.067**)
- Mean gap: 0.387 → **0.333** (reduction **0.055**)
- Improved cross F1: **14/18**; reduced gap: **13/18**
- Mean Δ by target: BRRI +0.092, RiceLeafBD +0.064, Dhan +0.045
- Best model under aug: ResNet50 mean cross F1 **0.609**, mean gap **0.244**
- Largest gain: ResNet50 BRRI→RiceLeafBD **0.300 → 0.716** (gap −0.343)
- Largest regression: MobileNetV2 BRRI→RiceLeafBD **0.573 → 0.273**

### LODO (9 runs: 3 held-outs × 3 models)
- Beats baseline single-source aggregate in only **3/9** cells
- Mean LODO F1 by held-out: RiceLeafBD **0.500**, Dhan **0.402**, BRRI **0.203**
- Best BRRI LODO: ResNet50 **0.267**
- Clearest LODO win: EfficientNet-B0 on held-out Dhan **0.317 → 0.489**

**Bottom line for Discussion:** Strong augmentation is the more reliable simple mitigation; LODO is an honest negative / inconsistent result, especially for BRRI.

---

## 5. Full result tables (copy into paper as needed)

### Table A — In-dataset baselines (`table_indataset`)

| model | dataset | accuracy | macro_f1 | n |
|-------|---------|---------:|---------:|--:|
| mobilenetv2_100 | riceleafbd | 0.894 | 0.907 | 235 |
| mobilenetv2_100 | dhan_shomadhan | 0.623 | 0.615 | 167 |
| mobilenetv2_100 | brri_rice_disease_pest | 0.599 | 0.570 | 414 |
| efficientnet_b0 | riceleafbd | 0.868 | 0.883 | 235 |
| efficientnet_b0 | dhan_shomadhan | 0.659 | 0.650 | 167 |
| efficientnet_b0 | brri_rice_disease_pest | 0.674 | 0.645 | 414 |
| resnet50 | riceleafbd | 0.898 | 0.913 | 235 |
| resnet50 | dhan_shomadhan | 0.611 | 0.590 | 167 |
| resnet50 | brri_rice_disease_pest | 0.725 | 0.699 | 414 |

### Table B — Baseline generalization gaps (`table_gap_baseline`)

| train | test | model | classes | in_F1 | cross_F1 | gap | in_n | cross_n |
|-------|------|-------|---------|------:|---------:|----:|-----:|--------:|
| riceleafbd | dhan_shomadhan | mobilenetv2_100 | brown_spot\|tungro | 0.908 | 0.510 | 0.398 | 133 | 51 |
| dhan_shomadhan | riceleafbd | mobilenetv2_100 | brown_spot\|tungro | 0.816 | 0.526 | 0.291 | 51 | 133 |
| riceleafbd | brri | mobilenetv2_100 | healthy\|tungro | 0.962 | 0.333 | 0.629 | 117 | 127 |
| brri | riceleafbd | mobilenetv2_100 | healthy\|tungro | 0.767 | 0.573 | 0.195 | 127 | 117 |
| dhan_shomadhan | brri | mobilenetv2_100 | rice_blast\|scald\|tungro | 0.609 | 0.296 | 0.313 | 103 | 204 |
| brri | dhan_shomadhan | mobilenetv2_100 | rice_blast\|scald\|tungro | 0.642 | 0.409 | 0.233 | 204 | 103 |
| riceleafbd | dhan_shomadhan | efficientnet_b0 | brown_spot\|tungro | 0.953 | 0.383 | 0.570 | 133 | 51 |
| dhan_shomadhan | riceleafbd | efficientnet_b0 | brown_spot\|tungro | 0.896 | 0.524 | 0.372 | 51 | 133 |
| riceleafbd | brri | efficientnet_b0 | healthy\|tungro | 0.981 | 0.375 | 0.606 | 117 | 127 |
| brri | riceleafbd | efficientnet_b0 | healthy\|tungro | 0.882 | 0.424 | 0.458 | 127 | 117 |
| dhan_shomadhan | brri | efficientnet_b0 | rice_blast\|scald\|tungro | 0.673 | 0.360 | 0.313 | 103 | 204 |
| brri | dhan_shomadhan | efficientnet_b0 | rice_blast\|scald\|tungro | 0.700 | 0.251 | 0.449 | 204 | 103 |
| riceleafbd | dhan_shomadhan | resnet50 | brown_spot\|tungro | 0.930 | 0.471 | 0.459 | 133 | 51 |
| dhan_shomadhan | riceleafbd | resnet50 | brown_spot\|tungro | 0.918 | 0.596 | 0.322 | 51 | 133 |
| riceleafbd | brri | resnet50 | healthy\|tungro | 0.990 | 0.625 | 0.366 | 117 | 127 |
| brri | riceleafbd | resnet50 | healthy\|tungro | 0.787 | 0.300 | 0.487 | 127 | 117 |
| dhan_shomadhan | brri | resnet50 | rice_blast\|scald\|tungro | 0.574 | 0.373 | 0.201 | 103 | 204 |
| brri | dhan_shomadhan | resnet50 | rice_blast\|scald\|tungro | 0.836 | 0.527 | 0.309 | 204 | 103 |

*(brri = brri_rice_disease_pest)*

### Table C — Background confound (`table_background_confound`)

| condition | model | train | classes | accuracy | macro_f1 | n | brown_spot F1 | tungro F1 |
|-----------|-------|-------|---------|---------:|---------:|--:|--------------:|----------:|
| dhan_white | resnet50 | dhan_shomadhan | brown_spot\|tungro | 0.862 | 0.854 | 29 | 0.818 | 0.889 |
| dhan_field | resnet50 | dhan_shomadhan | brown_spot\|tungro | 0.727 | 0.705 | 22 | 0.625 | 0.786 |
| riceleafbd_field | resnet50 | dhan_shomadhan | brown_spot\|tungro | 0.586 | 0.573 | 133 | 0.495 | 0.650 |

### Table D — Pairwise augmentation effect (`table_mitigation_pairwise`)

| train | test | model | classes | base_cross | aug_cross | base_gap | aug_gap | ΔF1 | gap↓ |
|-------|------|-------|---------|-----------:|----------:|---------:|--------:|----:|-----:|
| brri | dhan | efficientnet_b0 | blast\|scald\|tungro | 0.251 | 0.401 | 0.449 | 0.350 | +0.150 | +0.099 |
| brri | dhan | mobilenetv2_100 | blast\|scald\|tungro | 0.409 | 0.396 | 0.233 | 0.313 | −0.013 | −0.080 |
| brri | dhan | resnet50 | blast\|scald\|tungro | 0.527 | 0.456 | 0.309 | 0.386 | −0.071 | −0.077 |
| brri | riceleafbd | efficientnet_b0 | healthy\|tungro | 0.424 | 0.469 | 0.458 | 0.290 | +0.045 | +0.168 |
| brri | riceleafbd | mobilenetv2_100 | healthy\|tungro | 0.573 | 0.273 | 0.195 | 0.555 | −0.300 | −0.360 |
| brri | riceleafbd | resnet50 | healthy\|tungro | 0.300 | 0.716 | 0.487 | 0.144 | **+0.416** | **+0.343** |
| dhan | brri | efficientnet_b0 | blast\|scald\|tungro | 0.360 | 0.323 | 0.313 | 0.347 | −0.037 | −0.033 |
| dhan | brri | mobilenetv2_100 | blast\|scald\|tungro | 0.296 | 0.344 | 0.313 | 0.227 | +0.049 | +0.086 |
| dhan | brri | resnet50 | blast\|scald\|tungro | 0.373 | 0.573 | 0.201 | 0.043 | +0.200 | +0.158 |
| dhan | riceleafbd | efficientnet_b0 | brown_spot\|tungro | 0.524 | 0.676 | 0.372 | 0.164 | +0.152 | +0.208 |
| dhan | riceleafbd | mobilenetv2_100 | brown_spot\|tungro | 0.526 | 0.574 | 0.291 | 0.386 | +0.048 | −0.096 |
| dhan | riceleafbd | resnet50 | brown_spot\|tungro | 0.596 | 0.617 | 0.322 | 0.221 | +0.021 | +0.101 |
| riceleafbd | brri | efficientnet_b0 | healthy\|tungro | 0.375 | 0.452 | 0.606 | 0.539 | +0.077 | +0.067 |
| riceleafbd | brri | mobilenetv2_100 | healthy\|tungro | 0.333 | 0.504 | 0.629 | 0.467 | +0.171 | +0.162 |
| riceleafbd | brri | resnet50 | healthy\|tungro | 0.625 | 0.719 | 0.366 | 0.281 | +0.094 | +0.084 |
| riceleafbd | dhan | efficientnet_b0 | brown_spot\|tungro | 0.383 | 0.420 | 0.570 | 0.548 | +0.037 | +0.022 |
| riceleafbd | dhan | mobilenetv2_100 | brown_spot\|tungro | 0.510 | 0.572 | 0.398 | 0.340 | +0.062 | +0.058 |
| riceleafbd | dhan | resnet50 | brown_spot\|tungro | 0.471 | 0.574 | 0.459 | 0.386 | +0.104 | +0.073 |

### Table E — Strategy comparison by target (`table_mitigation_strategy`)

| target | model | baseline_single | aug_single | lodo | lodo−base | aug−base |
|--------|-------|----------------:|-----------:|-----:|----------:|---------:|
| brri | efficientnet_b0 | 0.367 | 0.387 | 0.126 | −0.241 | +0.020 |
| brri | mobilenetv2_100 | 0.314 | 0.424 | 0.216 | −0.099 | +0.110 |
| brri | resnet50 | 0.499 | 0.646 | 0.267 | −0.232 | +0.147 |
| dhan | efficientnet_b0 | 0.317 | 0.411 | 0.489 | **+0.171** | +0.094 |
| dhan | mobilenetv2_100 | 0.459 | 0.484 | 0.321 | −0.138 | +0.025 |
| dhan | resnet50 | 0.499 | 0.515 | 0.395 | −0.104 | +0.016 |
| riceleafbd | efficientnet_b0 | 0.474 | 0.572 | 0.541 | +0.067 | +0.098 |
| riceleafbd | mobilenetv2_100 | 0.549 | 0.423 | 0.574 | +0.025 | −0.126 |
| riceleafbd | resnet50 | 0.448 | 0.666 | 0.383 | −0.065 | **+0.219** |

Note: `baseline_single` / `aug_single` average the two single-source transfers that target that dataset. LODO uses a different class set.

### Table F — LODO full results (`table_lodo`)

| held_out | model | train_on | classes | acc | macro_f1 | n |
|----------|-------|----------|---------|----:|---------:|--:|
| riceleafbd | mobilenetv2_100 | dhan+brri | brown_spot\|healthy\|tungro | 0.596 | 0.574 | 171 |
| riceleafbd | efficientnet_b0 | dhan+brri | brown_spot\|healthy\|tungro | 0.585 | 0.541 | 171 |
| riceleafbd | resnet50 | dhan+brri | brown_spot\|healthy\|tungro | 0.515 | 0.383 | 171 |
| dhan_shomadhan | mobilenetv2_100 | riceleafbd+brri | brown_spot\|blast\|scald\|tungro | 0.355 | 0.321 | 124 |
| dhan_shomadhan | efficientnet_b0 | riceleafbd+brri | brown_spot\|blast\|scald\|tungro | 0.516 | 0.489 | 124 |
| dhan_shomadhan | resnet50 | riceleafbd+brri | brown_spot\|blast\|scald\|tungro | 0.444 | 0.395 | 124 |
| brri | mobilenetv2_100 | riceleafbd+dhan | healthy\|blast\|scald\|tungro | 0.251 | 0.216 | 295 |
| brri | efficientnet_b0 | riceleafbd+dhan | healthy\|blast\|scald\|tungro | 0.149 | 0.126 | 295 |
| brri | resnet50 | riceleafbd+dhan | healthy\|blast\|scald\|tungro | 0.308 | 0.267 | 295 |

### Table G — Summary stats (`table_summary_stats`)

| statistic | value |
|-----------|------:|
| baseline_mean_cross_macro_f1 | 0.436 |
| aug_mean_cross_macro_f1 | 0.503 |
| mean_cross_f1_improvement | 0.067 |
| baseline_mean_gap | 0.387 |
| aug_mean_gap | 0.333 |
| mean_gap_reduction | 0.055 |
| augmentation_positive_pairs | 14 |
| gap_reduction_positive_pairs | 13 |
| lodo_positive_target_model_cells | 3 |
| lodo_mean_riceleafbd | 0.500 |
| lodo_mean_dhan_shomadhan | 0.402 |
| lodo_mean_brri_rice_disease_pest | 0.203 |

---

## 6. Methods details Claude should include

### Diagnosis protocol
- Representative: ResNet50, train Dhan-Shomadhan, classes `{brown_spot, tungro}`, test RiceLeafBD + Dhan bg splits.
- Grad-CAM on correct and incorrect predictions; border enrichment proxy (uniform = 1.0).
- Background confound: same checkpoint on white / field / cross field.

### Mitigation protocol
- Strong aug: train loader only; 18 matched baseline pairs repeated.
- LODO: hold out one dataset; train other two; default aug; label space = largest non-degenerate overlap with held-out.
- One variable at a time: do **not** combine strong aug + LODO when attributing effects.

### Freeze / reproducibility
- Results audited and frozen under `frozen_results/` (SHA-256 of manifest recorded in audit).
- Figures regenerated deterministically via `make_figures.py`.
- Code: Python, PyTorch, timm, albumentations; training on Kaggle GPU T4.

---

## 7. Discussion points to emphasize

1. High in-dataset RiceLeafBD F1 (~0.90) can coexist with weak cross-dataset F1 (~0.44 mean) — single-dataset papers overstate deployability.
2. Background/acquisition ordering supports confound hypothesis without claiming full causality.
3. Strong augmentation is the practical first fix; ResNet50 benefits most.
4. LODO negative result is scientifically useful: pooling sources ≠ automatic generalization, especially for BRRI.
5. Remaining gaps after augmentation show domain dependence is not solved.

---

## 8. Limitations (must include)

- One seed (42), one frozen split
- Three CNNs only; ImageNet init
- Pairwise transfer restricted to shared classes
- LODO vs single-source not class-matched
- Diagnosis focused on one representative transfer
- Grad-CAM overlay may not be available as a paper figure (records exist)
- No real-world field deployment study
- Small white/field Dhan test counts (n=29/22) — report carefully

---

## 9. Related Work guidance for Claude

Expand with real citations where possible on:
- CNN / EfficientNet / ResNet / MobileNet for plant disease classification
- Domain shift / dataset bias / background bias in plant pathology
- Grad-CAM and attribution for agriculture models
- Data augmentation and multi-source / leave-one-domain-out evaluation
- Prior rice disease datasets (RiceLeafBD, Dhan-Shomadhan, BRRI Data in Brief if citable)

If unsure of exact citation metadata, use placeholders like `[Author et al., Year]` and mark them for the human to fill.

---

## 10. Existing draft Abstract (optional starting point — rewrite freely)

Deep learning models for rice leaf disease recognition often report strong in-dataset accuracy, yet their reliability across independently collected Bangladeshi datasets remains poorly characterized. We study cross-dataset generalization over three original-image collections—RiceLeafBD, Dhan-Shomadhan, and the BRRI Rice Leaf Disease and Pest dataset—after label harmonization, exclusion of pre-augmented images, and freezing of stratified 70/15/15 splits (5,419 images; seed 42). Using MobileNetV2, EfficientNet-B0, and ResNet50, we (i) measure a transfer benchmark and generalization gap on six ordered shared-class pairs, (ii) diagnose failure via Grad-CAM inspection and a background-confound evaluation, and (iii) test two standard mitigations: strong train-only augmentation and leave-one-dataset-out (LODO) training.

In-dataset mean macro-F1 is 0.719, while matched cross-dataset mean macro-F1 falls to 0.436 (mean gap 0.387). A Dhan-trained ResNet50 evaluated on white-background, field-background, and cross-dataset field tests yields macro-F1 0.854, 0.705, and 0.573, meeting a prespecified ordering that implicates acquisition/background shift. Strong augmentation raises mean cross-domain macro-F1 from 0.436 to 0.503 (14/18 pairs improve) and reduces the mean gap by 0.055; ResNet50 benefits most. LODO improves only 3 of 9 target/model comparisons and collapses on held-out BRRI (mean macro-F1 0.203). We conclude that strong augmentation is a more reliable first mitigation than naive multi-source training on this benchmark, while residual gaps and LODO failures underscore persistent domain dependence.

---

## 11. Suggested prompt to paste into Claude (with this file)

```
You are an academic writing assistant. Using ONLY the facts, tables, and rules in the attached CLAUDE_PAPER_BRIEF.md, write a complete research paper in Markdown.

Requirements:
- Application-plus-insight paper (benchmark + diagnosis + mitigation), not a new method.
- Do not invent any numbers.
- Include Abstract, Introduction, Related Work, Datasets, Methods, Results, Discussion, Limitations, Conclusion.
- Insert figure references Fig. 1–9 with captions based on the figure table in the brief.
- Include the main result tables (you may reformat Table A–G for readability).
- Keep Limitations honest.
- Target length: ~3500–5500 words.
- Style: clear scientific English suitable for a computer vision / agriculture AI journal.
```

---

## 12. File map (if Claude/user has the repo)

| Path | Content |
|------|---------|
| `paper/claude_paper_brief.md` | **This file** |
| `paper/manuscript.md` | Current Week 9 draft |
| `paper/figures/` | fig01–fig09 PNGs |
| `paper/tables/` | CSV + LaTeX tables |
| `frozen_results/` | Audited immutable CSVs |
| `notes/week6_retro.md` | Diagnosis narrative |
| `notes/week7_retro.md` | Mitigation narrative |
| `notes/week8_retro.md` | Freeze audit |

---

**End of brief.** All publication numbers above are taken from the Week 8 freeze and must not be altered without a new audited experiment.
