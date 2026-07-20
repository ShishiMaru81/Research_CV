# Cross-Dataset Generalization of Bangladeshi Rice Leaf Disease Classifiers: Benchmark, Diagnosis, and Mitigation

**Draft manuscript (Week 9)**  
Venue-agnostic Markdown. Numbers and figures are taken from the frozen Week 8 release (`frozen_results/`, `paper/figures/`, `paper/tables/`). Seed = 42 throughout.

---

## Abstract

Deep learning models for rice leaf disease recognition often report strong in-dataset accuracy, yet their reliability across independently collected Bangladeshi datasets remains poorly characterized. We study cross-dataset generalization over three original-image collections—RiceLeafBD, Dhan-Shomadhan, and the BRRI Rice Leaf Disease and Pest dataset—after label harmonization, exclusion of pre-augmented images, and freezing of stratified 70/15/15 splits (5,419 images; seed 42). Using MobileNetV2, EfficientNet-B0, and ResNet50, we (i) measure a transfer benchmark and generalization gap on six ordered shared-class pairs, (ii) diagnose failure via Grad-CAM inspection and a background-confound evaluation, and (iii) test two standard mitigations: strong train-only augmentation and leave-one-dataset-out (LODO) training.

In-dataset mean macro-F1 is 0.719, while matched cross-dataset mean macro-F1 falls to 0.436 (mean gap 0.387). A Dhan-trained ResNet50 evaluated on white-background, field-background, and cross-dataset field tests yields macro-F1 0.854, 0.705, and 0.573, meeting a prespecified ordering that implicates acquisition/background shift. Strong augmentation raises mean cross-domain macro-F1 from 0.436 to 0.503 (14/18 pairs improve) and reduces the mean gap by 0.055; ResNet50 benefits most. LODO improves only 3 of 9 target/model comparisons and collapses on held-out BRRI (mean macro-F1 0.203). We conclude that strong augmentation is a more reliable first mitigation than naive multi-source training on this benchmark, while residual gaps and LODO failures underscore persistent domain dependence.

---

## 1. Introduction

Rice leaf disease classifiers are frequently evaluated within a single collection under fixed acquisition conditions. In Bangladesh, several public and research datasets now exist, but they differ in cameras, backgrounds (field vs studio/white), class inventories, and labeling conventions. A model that scores highly on one collection may fail when deployed on images from another farm, research station, or citizen-science source.

This paper treats cross-dataset generalization as a first-class empirical question rather than an afterthought. Our contribution is deliberately **application-plus-insight**, not a new architecture:

1. **Benchmark.** A frozen, shared-class transfer matrix and generalization gap across three Bangladeshi collections and three CNN backbones.
2. **Diagnosis.** Qualitative Grad-CAM evidence and a quantitative background-confound experiment that separates white-background, field-background, and cross-dataset field conditions.
3. **Mitigation.** One-variable tests of strong train-only augmentation versus leave-one-dataset-out (LODO) multi-source training.

We ask: How large is the gap between in-dataset and cross-dataset macro-F1? Does background/acquisition appear to contribute? Do simple, standard interventions close the gap?

---

## 2. Related work

Convolutional networks are widely applied to crop disease recognition, typically reporting strong accuracy when train and test images share a dataset. Domain shift studies in plant pathology and broader computer vision show that background, illumination, and sensor differences can dominate lesion cues. Attribution methods such as Grad-CAM are often used to check whether models attend to biologically plausible regions; they provide supporting qualitative evidence rather than causal proof.

Multi-source training and aggressive data augmentation are common practical responses to shift. We evaluate both under a controlled protocol: frozen splits, fixed seed, and one changed experimental factor at a time. Our goal is not to introduce a new domain-adaptation method, but to document what happens when standard practices are applied honestly to a Bangladeshi multi-dataset setting.

---

## 3. Datasets and preprocessing

### 3.1 Collections

We use three Bangladeshi rice leaf disease collections (originals only):

| Dataset | Role in study | Approx. original images used |
|---------|---------------|------------------------------|
| RiceLeafBD | Field-oriented leaf disease images | 1,560 |
| Dhan-Shomadhan | Field and white-background subsets | 1,106 |
| BRRI Rice Leaf Disease and Pest | Station/field originals (augmented archive excluded) | 2,753 |

After exclusions, the harmonized corpus contains **5,419** original images (Table: `paper/tables/table_indataset.csv` sample counts; manifest: `frozen_results/manifest.csv`).

### 3.2 Label harmonization

Raw folder names were mapped to a shared canonical taxonomy (healthy, bacterial leaf blight, brown spot, tungro, rice blast, scald, sheath blight, leaf folder, insect, stripes), including documented spelling variants (e.g., Browon Spot, Rice Turgro, Shath Blight). Unmatched labels halt the pipeline rather than being guessed. The BRRI `Rice` folder (16 images of ambiguous/general content) is **excluded** by design and documented in `notes/label_mapping.md`.

### 3.3 Splits and leakage controls

- Stratified **70/15/15** train/val/test splits per dataset and class, seed **42**, frozen after creation.
- Pre-augmented BRRI images are never used.
- Cross-dataset near-duplicate search with perceptual hashing (threshold 2) found **no** true shared copies (`notes/dedup_report.md`).

### 3.4 Shared-class transfer pairs

Transfer experiments use the largest non-degenerate pairwise overlaps:

| Train → Test | Shared classes |
|--------------|----------------|
| RiceLeafBD ↔ Dhan-Shomadhan | brown_spot, tungro |
| Dhan-Shomadhan ↔ BRRI | rice_blast, scald, tungro |
| RiceLeafBD ↔ BRRI | healthy, tungro |

All six ordered directions are evaluated. Dhan-Shomadhan’s dual backgrounds enable the later confound diagnosis.

---

## 4. Methods

### 4.1 Models and training

Backbones: **MobileNetV2** (`mobilenetv2_100`), **EfficientNet-B0**, and **ResNet50**, ImageNet-pretrained via `timm`, with the classifier head replaced for the active class set. Protocol (all experiments unless noted):

- Image size 224, batch size 32, Adam, learning rate 0.001
- 3-epoch head-only warm-up, then full fine-tuning up to 30 epochs
- Class-weighted cross-entropy; early stopping on validation macro-F1 (patience 7)
- Mixed precision on GPU; seed 42 for Python, NumPy, and PyTorch

Default training augmentation is mild geometric/photometric jitter. Evaluation transforms are deterministic.

### 4.2 Metrics

Primary metric: **macro-F1**. We also report accuracy and per-class F1. The **generalization gap** for a transfer run is:

\[
\text{gap} = \text{macro-F1}_{\text{source test}} - \text{macro-F1}_{\text{target test}}
\]

on the same shared-class subset.

### 4.3 Benchmark (Week 5)

For each of 6 ordered pairs × 3 models (18 runs), train on the source shared-class subset and evaluate on both source and target frozen test sets. Results: `frozen_results/crossdataset_matrix.csv`, `generalization_gap.csv`; Figures `fig02`, `fig03`.

### 4.4 Diagnosis (Week 6)

**Representative setting:** ResNet50 trained on Dhan-Shomadhan with shared classes `{brown_spot, tungro}`, evaluated toward RiceLeafBD.

- **Grad-CAM:** Overlays on correct and incorrect RiceLeafBD predictions; border-attention enrichment is a reproducible proxy (uniform heatmap = 1.0), not lesion segmentation.
- **Background confound:** Same checkpoint evaluated on (i) Dhan white-background test, (ii) Dhan field-background test, (iii) RiceLeafBD field test. Prespecified descriptive support criterion: white macro-F1 > field macro-F1 **and** white macro-F1 > cross-dataset field macro-F1.

Figure: `fig04_background_confound.png`. Table: `table_background_confound`.

### 4.5 Mitigation (Week 7)

Two interventions, each changing **one** factor versus the Week 5 baseline:

**Strong augmentation (train only).** RandomResizedCrop (scale 0.6–1.0), HorizontalFlip, Affine, RandomBrightnessContrast, HueSaturationValue, GaussianBlur, and CoarseDropout. All 18 transfer runs are repeated with an `__aug-strong` tag. Figures `fig05`–`fig07`.

**Leave-one-dataset-out (LODO).** Train on two datasets combined; evaluate on the held-out third. Label space = largest non-degenerate set present in the held-out set and at least one source:

| Held out | Train on | Classes |
|----------|----------|---------|
| RiceLeafBD | Dhan + BRRI | brown_spot, healthy, tungro |
| Dhan-Shomadhan | RiceLeafBD + BRRI | brown_spot, rice_blast, scald, tungro |
| BRRI | RiceLeafBD + Dhan | healthy, rice_blast, scald, tungro |

LODO uses **default** augmentation so source composition is the only changed factor. LODO and pairwise single-source scores use different class sets and are therefore **strategy-level** comparisons. Figures `fig08`, `fig09`.

### 4.6 Result freeze (Week 8)

All publication numbers are audited against per-run metrics JSON files, gap arithmetic, and re-derived mitigation tables (`python -m freeze_results`). Figures and LaTeX/CSV tables are regenerated deterministically (`python -m make_figures`).

---

## 5. Results

### 5.1 In-dataset baselines

Figure `fig01_indataset_macro_f1.png`; Table `table_indataset`.

Mean test macro-F1 across 9 model–dataset runs is **0.719**. By dataset (models pooled): RiceLeafBD **0.901**, BRRI **0.638**, Dhan-Shomadhan **0.618**. By model (datasets pooled): ResNet50 **0.734**, EfficientNet-B0 **0.726**, MobileNetV2 **0.697**. Strong in-dataset performance on RiceLeafBD confirms that the pipeline and labels are usable; weaker scores on Dhan and BRRI already hint at harder within-collection conditions (class imbalance, mixed backgrounds, or harder visual classes).

### 5.2 Cross-dataset transfer and gaps

Figures `fig02`, `fig03`; Tables `table_transfer_baseline`, `table_gap_baseline`.

Across 18 transfer runs:

| Quantity | Mean |
|----------|-----:|
| Source (in-subset) macro-F1 | 0.824 |
| Cross-dataset macro-F1 | **0.436** |
| Generalization gap | **0.387** |

Gaps range from about 0.195 to 0.629. Transfer is consistently weaker than matched-subset source evaluation, establishing a large and systematic cross-collection drop for all three architectures.

### 5.3 Diagnosis: background confound and Grad-CAM

Figure `fig04`; Table `table_background_confound`; notes in `notes/week6_retro.md`.

Same Dhan-trained ResNet50 (`brown_spot`, `tungro`):

| Condition | n | Accuracy | Macro-F1 |
|-----------|--:|---------:|---------:|
| Dhan white-bg | 29 | 0.862 | **0.854** |
| Dhan field-bg | 22 | 0.727 | **0.705** |
| RiceLeafBD field | 133 | 0.586 | **0.573** |

Brown-spot F1 falls monotonically: 0.818 (white) → 0.625 (field) → 0.495 (cross-dataset field). The prespecified ordering (white > field and white > cross) is **met**. Grad-CAM overlays (generated during Week 6; local overlay PNG not in the frozen figure set) show attention that often spreads into neighboring leaves, grass, and background rather than remaining on lesions. Border-attention enrichment does **not** cleanly separate correct from incorrect predictions; Grad-CAM is supporting qualitative evidence, not standalone proof.

### 5.4 Mitigation: strong augmentation

Figures `fig05`–`fig07`; Tables `table_mitigation_pairwise`, `table_summary_stats`.

| Statistic | Value |
|-----------|------:|
| Mean cross macro-F1 (baseline → aug) | 0.436 → **0.503** |
| Mean improvement | **+0.067** |
| Pairs improved (cross F1) | **14 / 18** |
| Mean gap (baseline → aug) | 0.387 → **0.333** |
| Mean gap reduction | **0.055** |
| Pairs with reduced gap | **13 / 18** |

Mean improvement by target: BRRI +0.092, RiceLeafBD +0.064, Dhan-Shomadhan +0.045. Under strong augmentation, ResNet50 achieves the best mean cross-domain macro-F1 (**0.609**) and smallest mean gap (**0.244**). Largest gain: ResNet50, BRRI → RiceLeafBD, 0.300 → **0.716** (gap reduction 0.343). Largest regression: MobileNetV2, BRRI → RiceLeafBD, 0.573 → 0.273. Augmentation helps most runs but is not universal.

### 5.5 Mitigation: LODO

Figures `fig08`, `fig09`; Tables `table_lodo`, `table_mitigation_strategy`.

LODO beats the baseline single-source aggregate in only **3 of 9** target/model cells:

- EfficientNet-B0 → held-out Dhan: 0.317 → 0.489 (+0.171)
- EfficientNet-B0 → held-out RiceLeafBD: 0.474 → 0.541 (+0.067)
- MobileNetV2 → held-out RiceLeafBD: 0.549 → 0.574 (+0.025)

Mean LODO macro-F1 by held-out target: RiceLeafBD **0.500**, Dhan-Shomadhan **0.402**, BRRI **0.203**. Best BRRI LODO result is only 0.267 (ResNet50). Adding a second source dataset does not reliably replace careful single-source training plus augmentation on this benchmark.

---

## 6. Discussion

**The gap is real.** Strong in-dataset RiceLeafBD scores coexist with a mean cross-domain macro-F1 near 0.44. Reporting only within-collection accuracy would overstate deployability across Bangladeshi sources.

**Background and acquisition matter.** The white → field → cross-dataset field ordering for a fixed Dhan-trained model supports the hypothesis that non-disease cues contribute to failure. Grad-CAM is consistent with this story but should not be oversold: border enrichment failed as a clean correctness separator.

**Strong augmentation is the more reliable simple mitigation.** Improving 14/18 matched transfers and cutting the mean gap fits a picture where randomizing crop, color, blur, and occlusion reduces dependence on dataset-specific appearance. ResNet50 benefits most, suggesting capacity and feature richness interact with augmentation.

**LODO is an honest negative result.** Multi-source training without domain alignment often underperforms, especially for BRRI. Source diversity alone is insufficient when collections differ in class appearance, balance, and capture conditions. Because LODO label spaces differ from pairwise overlaps, we interpret LODO as a strategy-level finding, not a matched-class effect.

**Practical implication.** For Bangladeshi rice disease tools that may see images from multiple collections, practitioners should (i) measure cross-dataset transfer explicitly, (ii) treat background/acquisition as a risk factor, and (iii) prefer aggressive train-time augmentation before assuming that pooling datasets will fix generalization.

---

## 7. Limitations

- **Single seed and single frozen split.** All numbers use seed 42; variance across seeds/splits is unquantified.
- **Three architectures, ImageNet initialization.** Results may not transfer to transformers or custom domain-adaptation methods.
- **Shared-class subsets only.** Classes unique to one dataset cannot appear in pairwise transfer by construction.
- **LODO vs single-source is strategy-level**, not class-matched.
- **Diagnosis scope.** Background confound and Grad-CAM focus on one representative transfer (Dhan → RiceLeafBD, ResNet50, two classes).
- **Grad-CAM overlay availability.** Sample records are frozen (`gradcam_records.csv`); the Week 6 overlay PNG was not re-bundled locally for Week 8 figures and is noted as a provenance gap.
- **No field deployment study.** Results are offline evaluations on public research collections.

---

## 8. Conclusion

We assembled a reproducible Bangladeshi rice leaf disease cross-dataset benchmark with frozen splits, a transfer matrix, a background-aware diagnosis, and two controlled mitigations. Cross-dataset macro-F1 is substantially lower than matched in-subset performance (mean 0.436 vs 0.824; mean gap 0.387). A background-confound experiment supports acquisition/background contribution to the failure. Strong augmentation improves most matched transfers (mean cross F1 0.436 → 0.503), while LODO usually does not. The study’s primary value is an evidence-backed warning against single-dataset optimism and a clear ranking of two simple mitigations on this frozen benchmark.

---

## 9. Reproducibility

| Artifact | Path |
|----------|------|
| Frozen CSVs + audit | `frozen_results/` |
| Figures | `paper/figures/fig01`–`fig09` |
| Tables (CSV/LaTeX) | `paper/tables/` |
| Freeze command | `python -m freeze_results` |
| Figure command | `python -m make_figures` |
| Code + Kaggle notes | `src/`, `run_*.py`, `notebooks/kaggle_week*.md` |

---

## Figure checklist (for camera-ready)

| ID | File | Use in |
|----|------|--------|
| Fig. 1 | `fig01_indataset_macro_f1.png` | §5.1 |
| Fig. 2 | `fig02_crossdataset_heatmap_baseline.png` | §5.2 |
| Fig. 3 | `fig03_generalization_gap_baseline.png` | §5.2 |
| Fig. 4 | `fig04_background_confound.png` | §5.3 |
| Fig. 5 | `fig05_crossdataset_heatmap_aug.png` | §5.4 |
| Fig. 6 | `fig06_generalization_gap_aug.png` | §5.4 |
| Fig. 7 | `fig07_augmentation_f1_delta.png` | §5.4 |
| Fig. 8 | `fig08_mitigation_by_target.png` | §5.5 |
| Fig. 9 | `fig09_lodo_heldout.png` | §5.5 |

Optional: restore Grad-CAM grid from Kaggle Week 6 bundle as Fig. 10 if available.
