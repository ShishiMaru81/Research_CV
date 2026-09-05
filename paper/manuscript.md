# Cross-Dataset Generalization of Bangladeshi Rice Leaf Disease Classifiers: Benchmark, Diagnosis, and Mitigation

**Camera-ready draft (Phase 5 revision)**  
Venue-agnostic Markdown. Week 5–7 core CSVs are immutable under `frozen_results/` (SHA-256 in `freeze_manifest.json`). AdaBN, ablation, multi-seed aggregates, and stats use `frozen_results_v2/`. Primary transfer/augmentation headlines use the **three-train-seed** pair (0.445 → 0.502, Δ +0.063); seed-42-only matrices are retained for reference and are not mixed into that headline. **12 of 18** strong-aug cells still lack seed 2024 locally.

---

## Abstract

Deep learning models for rice leaf disease recognition often report strong in-dataset accuracy, yet their reliability across independently collected Bangladeshi datasets remains poorly characterized. We study cross-dataset generalization over three original-image collections—RiceLeafBD, Dhan-Shomadhan, and the BRRI Rice Leaf Disease and Pest dataset—after label harmonization, exclusion of pre-augmented images, and freezing of stratified 70/15/15 splits (5,419 images; seed 42). Using MobileNetV2, EfficientNet-B0, and ResNet50, we (i) measure a transfer benchmark and generalization gap on six ordered shared-class pairs, (ii) diagnose failure via Grad-CAM inspection and a background-confound evaluation, and (iii) test three standard mitigations: strong train-only augmentation, leave-one-dataset-out (LODO) training, and Adaptive Batch Normalization (AdaBN).

In-dataset mean macro-F1 is 0.719. Across three train seeds (split seed fixed at 42), mean cross-dataset macro-F1 is 0.445 under default augmentation and 0.502 under strong augmentation (mean paired Δ +0.063; Wilcoxon *p* ≈ 0.0077 on 18 cell means; 14/18 positive), comparable to the baseline across-seed noise floor ±0.057. A Dhan-trained ResNet50 evaluated on white-background, field-background, and cross-dataset field tests yields macro-F1 0.854, 0.705, and 0.573, meeting a prespecified ordering that implicates acquisition/background shift. ResNet50 benefits most under full strong augmentation (0.609 mean cross F1 on six pairs at seed 42). A bucket ablation on those pairs shows geometric/background transforms alone yield the largest gain (mean cross F1 0.567, Δ +0.085 vs baseline), recovering about two-thirds of the full strong-aug improvement; photometric and occlusion buckets contribute less on average. LODO improves only 3 of 9 target/model comparisons and collapses on held-out BRRI (mean macro-F1 0.203). AdaBN improves only 5/18 pairs (mean Δ −0.055, *p* ≈ 0.10) and never helps ResNet50. We conclude that strong augmentation—especially geometric randomization—is the most reliable simple mitigation among those tested, while LODO and AdaBN are largely unsuccessful on this benchmark and residual gaps underscore persistent domain dependence.

---

## 1. Introduction

Rice leaf disease classifiers are frequently evaluated within a single collection under fixed acquisition conditions. In Bangladesh, several public and research datasets now exist, but they differ in cameras, backgrounds (field vs studio/white), class inventories, and labeling conventions. A model that scores highly on one collection may fail when deployed on images from another farm, research station, or citizen-science source.

This paper treats cross-dataset generalization as a first-class empirical question rather than an afterthought. Our contribution is deliberately **application-plus-insight**, not a new architecture:

1. **Benchmark.** A frozen, shared-class transfer matrix and generalization gap across three Bangladeshi collections and three CNN backbones, with multi-seed replication (train seeds 42, 7, 2024; split seed fixed at 42).
2. **Diagnosis.** Qualitative Grad-CAM evidence and a quantitative background-confound experiment that separates white-background, field-background, and cross-dataset field conditions on a representative Dhan-trained ResNet50.
3. **Mitigation.** One-variable tests of strong train-only augmentation, leave-one-dataset-out (LODO) multi-source training, and Adaptive Batch Normalization (AdaBN) on frozen source checkpoints.
4. **Mechanism and rigour.** An augmentation bucket ablation (ResNet50, six pairs) and paired inferential statistics (Wilcoxon signed-rank, bootstrap CIs, across-seed variance) that contextualize headline gains against seed noise.

We ask: How large is the gap between in-dataset and cross-dataset macro-F1? Does background/acquisition appear to contribute in the settings we can isolate? Do simple, standard interventions close the gap—and which ingredients of strong augmentation matter most?

---

## 2. Related work

**Plant disease recognition.** Convolutional networks achieve strong in-collection accuracy on curated plant-disease benchmarks such as PlantVillage (Mohanty et al., 2016) and large multi-crop corpora (Ferentinos, 2018). Reviews emphasize that reported performance often assumes a single acquisition regime and that real-field deployment faces illumination, occlusion, and background clutter (Barbedo, 2018; Liu & Wang, 2021). Bangladeshi rice collections—including RiceLeafBD (Rimi et al., 2025), Dhan-Shomadhan (Hossain et al., 2021), and the BRRI rice leaf disease and pest archive (Hasan et al., 2025)—differ in cameras, backgrounds, and label inventories, making them a natural testbed for cross-collection evaluation rather than within-dataset leaderboard chasing.

**Domain shift and generalization.** Domain shift in vision arises from covariate and label distribution differences between train and test environments (Quinonero-Candela et al., 2009). Domain generalization surveys catalog mitigation families—data augmentation, multi-source training, representation learning, and test-time adaptation—and stress that simple baselines are often under-reported relative to complex methods (Zhou et al., 2022). In agricultural imaging, shift from controlled to field conditions remains a recurring failure mode even when in-lab accuracy is high (Too et al., 2019).

**Diagnosis and attribution.** Grad-CAM (Selvaraju et al., 2017) and related saliency tools are widely used to inspect whether models attend to lesions versus background; they provide qualitative support but not causal identification of confounds. Our background-confound design follows this literature by holding the checkpoint fixed while varying evaluation background (white vs field vs cross-dataset field).

**Mitigations we test.** Train-time augmentation is a standard first response to appearance shift (Shorten & Khoshgoftaar, 2019). Multi-source training pools heterogeneous collections but does not guarantee alignment of class appearance or balance. Adaptive Batch Normalization (AdaBN; Li et al., 2016) recalibrates BatchNorm running statistics on unlabeled target data without gradient updates and is a lightweight domain-adaptation baseline often paired with deep CNNs. Agriculture-focused workshops (e.g., CVPR Agriculture-Vision; Rust et al., 2020) similarly highlight robustness and cross-condition evaluation as open problems.

**Positioning.** Prior rice-disease work in Bangladesh typically reports strong within-dataset metrics. We instead freeze splits, harmonize labels, and change **one** experimental factor at a time while measuring cross-dataset macro-F1, generalization gap, and seed variability. Our goal is not a new architecture but an evidence-backed ranking of standard mitigations—and an honest account of where they fail—on three public Bangladeshi collections.

---

## 3. Datasets and preprocessing

### 3.1 Collections

We use three Bangladeshi rice leaf disease collections (originals only):

| Dataset | Role in study | Approx. original images used |
|---------|---------------|------------------------------|
| RiceLeafBD | Field-oriented leaf disease images | 1,560 |
| Dhan-Shomadhan | Field and white-background subsets | 1,106 |
| BRRI Rice Leaf Disease and Pest | Station/field originals (augmented archive excluded) | 2,753 |

RiceLeafBD contributes **1,560** original images in the released archive we use (422+356+252+530 by class inventory); the source descriptor reports 1,555—we use the archive count after file-level verification.

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

Primary metric: **macro-F1** on the evaluation split. For cross-dataset transfer we report **target-test macro-F1** (cross-dataset macro-F1) as the headline transfer score because it measures absolute performance on the deployment collection. The **generalization gap** is reported secondarily and always alongside the matched source-test macro-F1:

\[
\text{gap} = \text{macro-F1}_{\text{source test}} - \text{macro-F1}_{\text{target test}}
\]

on the same shared-class subset. Gap can partly reflect source-task difficulty (in-dataset F1 ranges 0.638–0.99 across sources) rather than transfer failure alone; cross-F1 is therefore preferred for comparing transfer arms and mitigations.

We also report accuracy and per-class F1. On RiceLeafBD, macro-F1 often exceeds accuracy (e.g., ResNet50: 0.913 vs 0.898) because class-weighted cross-entropy lifts recall on minority classes more than it lifts overall accuracy.

### 4.3 Benchmark (Week 5)

For each of 6 ordered pairs × 3 models (18 runs), train on the source shared-class subset and evaluate on both source and target frozen test sets. Results: `frozen_results/crossdataset_matrix.csv`, `generalization_gap.csv`; Figures `fig02`, `fig03`.

### 4.4 Diagnosis (Week 6)

**Representative setting:** ResNet50 trained on Dhan-Shomadhan with shared classes `{brown_spot, tungro}`, evaluated toward RiceLeafBD.

- **Grad-CAM:** Overlays on correct and incorrect RiceLeafBD predictions. **Border-attention enrichment** is computed from a normalized Grad-CAM map \(H\) (non-negative, summing to 1 over spatial locations). Let \(B\) be the border band comprising the outer **20%** of height and width on each side. Then \(\text{enrichment} = \mathrm{mean}(H[B]) / \mathrm{mean}(H)\); a uniform map yields 1.0. This is a reproducible background-attention proxy, not lesion segmentation.
- **Background confound:** Same checkpoint evaluated on (i) Dhan white-background test, (ii) Dhan field-background test, (iii) RiceLeafBD field test. Prespecified descriptive support criterion: white macro-F1 > field macro-F1 **and** white macro-F1 > cross-dataset field macro-F1.

Figure: `fig04_background_confound.png`. Table: `table_background_confound`.

### 4.5 Mitigation (Week 7–12)

Three interventions, each changing **one** factor versus the Week 5-style baseline transfer protocol:

**Strong augmentation (train only).** RandomResizedCrop (scale 0.6–1.0), HorizontalFlip, Affine, RandomBrightnessContrast, HueSaturationValue, GaussianBlur, and CoarseDropout. All 18 transfer runs are repeated with an `__aug-strong` tag. Figures `fig05`–`fig07`.

**Leave-one-dataset-out (LODO).** Train on two datasets combined; evaluate on the held-out third. Label space = largest non-degenerate set present in the held-out set and at least one source:

| Held out | Train on | Classes |
|----------|----------|---------|
| RiceLeafBD | Dhan + BRRI | brown_spot, healthy, tungro |
| Dhan-Shomadhan | RiceLeafBD + BRRI | brown_spot, rice_blast, scald, tungro |
| BRRI | RiceLeafBD + Dhan | healthy, rice_blast, scald, tungro |

LODO uses **default** augmentation so source composition is the only changed factor. LODO and pairwise single-source scores use different class sets and are therefore **strategy-level** comparisons. Figures `fig08`, `fig09`.

**Adaptive Batch Normalization (AdaBN).** For each of the 18 source-trained baseline checkpoints (default augmentation, seed 42), we recalibrate BatchNorm running mean/variance using forward passes over the **target dataset train split** only (labels unused during adaptation; non-BN modules remain in eval mode). We then evaluate on the target frozen test set and compare to an unrecalibrated evaluation of the **same** checkpoint. Adaptation never sees the target test split. This isolates BN-statistic shift as the sole changed factor. Table: `table_adabn`.

### 4.6 Augmentation bucket ablation (Week 13)

To attribute the strong-augmentation gain without a full leave-one-transform-out design, we group the strong pipeline into three **mechanism buckets** and run each bucket **alone**:

| Bucket | Transforms | Hypothesis |
|--------|------------|------------|
| Geometric / background | RandomResizedCrop, HorizontalFlip, Affine | Background randomization is the active ingredient |
| Photometric | RandomBrightnessContrast, HueSaturationValue | Dataset colour/lighting signature is the active ingredient |
| Occlusion | GaussianBlur, CoarseDropout | Forcing multi-region reliance is the active ingredient |

Scope (budgeted): **ResNet50 × 6 transfer pairs × seed 42** (18 runs). ResNet50 is chosen because it showed the largest Week-7 augmentation response, making mechanism differences most detectable; restricting to one model/seed is stated as a limitation. Outputs: `augmentation_ablation.csv`; figure `fig11_ablation_buckets.png`.

### 4.7 Statistics and freeze discipline (Week 11–14)

**Train seed vs split seed.** Stratified 70/15/15 splits remain frozen at **split_seed = 42**. Multi-seed replication varies only **train_seed** ∈ {42, 7, 2024} (initialization, shuffling, augmentation sampling).

**Inferential tests.** For the paired augmentation claim we use a Wilcoxon signed-rank test on matched baseline vs strong cross-dataset macro-F1 (no normality assumption; *n* = 18 cells, optionally aggregated as seed means). AdaBN uses the same paired test on Δ macro-F1. Across-seed variability is reported as mean ± std per cell. Per-run uncertainty uses bootstrap resampling of test samples (1,000 resamples) from full per-sample prediction CSVs. We avoid uncorrected per-cell *t*-tests.

**Replacement freeze.** Seed-42 Week 5–7 CSVs remain the immutable files under `frozen_results/` (SHA-256 digests in `freeze_manifest.json`). Multi-seed, AdaBN, ablation, and statistics overlays are released under `frozen_results_v2/` with `notes/freeze_v2_changelog.md` (`python -m freeze_results_v2`). The v2 assembler **copies** those v1 core files and checks file-copy / hash integrity; it does not independently re-derive metrics from training logs. Numerical checks that can fail (gap arithmetic, seed-42 identity in the multi-seed table, Wilcoxon recomputation) are run separately via `python scripts/numerical_freeze_audit.py`. Figures: `python -m make_figures` (core) and `python -m make_figures --revision` (Phase-4 plots).

### 4.8 Result freeze (Week 8)

Week 5–7 publication CSVs were frozen under `frozen_results/` after checks against per-run metrics JSON files, gap arithmetic, and re-derived mitigation tables (`python -m freeze_results`; 23 checks in `audit_report.md`). Recorded digests, not the obsolete git SHA stamped at freeze time, are the integrity ground truth after a later history rewrite that removed Cursor co-author trailers (`notes/missing_commit_investigation.md`). Figures and LaTeX/CSV tables are regenerated deterministically (`python -m make_figures`). AdaBN numbers (§5.6) are archived under `week12_results/adabn/` and `paper/tables/table_adabn.csv`.

### 4.9 Disclosure of AI assistance

Large language model assistants (Anthropic Claude, used through Claude Code and Cursor) were
used throughout this project for code authoring and review, analysis and figure scripting,
audit tooling, and manuscript drafting and copy-editing; the repository carries this standing
context in `CLAUDE.md` / `.cursorrules`. No experimental result was produced by a language
model. All reported metrics are computed by the versioned training and evaluation code in this
repository from the frozen splits, and every headline number is recomputed from the released
CSVs by `python scripts/numerical_freeze_audit.py` (12 checks) and `python
scripts/audit_writing.py` (22 checks). The author designed the study, ran the experiments, and
takes full responsibility for the content and for every claim made here.

Data, code, and reproducibility artifacts:

- Zenodo (preprint and archived artifacts): https://doi.org/10.5281/zenodo.21787018
- GitHub: https://github.com/ShishiMaru81/Research_CV

Model checkpoints are **not** included. They were produced on Kaggle and local GPUs and were
not archived to persistent storage (`notes/kaggle_checkpoint_verification.md`). Reported
metrics and the split manifest are therefore reproducible from code, but exact re-scoring of
stored predictions is limited to the 19 runs whose per-sample prediction CSVs survive
(§5.8, §7).

---

## 5. Results

### 5.1 In-dataset baselines

Figure `fig01_indataset_macro_f1.png`; Table `table_indataset`.

Mean test macro-F1 across 9 model–dataset runs is **0.719**. By dataset (models pooled): RiceLeafBD **0.901**, BRRI **0.638**, Dhan-Shomadhan **0.618**. By model (datasets pooled): ResNet50 **0.734**, EfficientNet-B0 **0.726**, MobileNetV2 **0.697**. Strong in-dataset performance on RiceLeafBD confirms that the pipeline and labels are usable; weaker scores on Dhan and BRRI already hint at harder within-collection conditions (class imbalance, mixed backgrounds, or harder visual classes).

### 5.2 Cross-dataset transfer and gaps

Figures `fig02`, `fig03`; Tables `table_transfer_baseline`, `table_gap_baseline`, `table_transfer_multiseed`.

**Primary reporting uses the three-train-seed aggregate** (train seeds 42, 7, 2024; split seed 42; Table `table_transfer_multiseed`). Mean cross-dataset macro-F1 over 18 cells is **0.445 ± 0.057** (mean ± mean across-seed std per cell). Transfer is consistently weaker than matched-subset source evaluation for all three architectures.

For reference only, the frozen seed-42 matrix alone (not mixed into the multi-seed headline) has mean source-subset macro-F1 0.824, mean cross macro-F1 0.436, and mean gap 0.387.

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

Figures `fig05`–`fig07`; Tables `table_mitigation_pairwise`, `table_summary_stats`, `table_transfer_multiseed`, `table_summary_stats_multiseed`.

**Multi-seed aggregates (primary).** Baseline mean cross macro-F1 **0.445**, strong augmentation **0.502**; mean paired Δ **+0.063** with **14 / 18** cell means positive (Table `table_summary_stats_multiseed`). Wilcoxon signed-rank on the 18 cell-mean paired deltas: *W* = 26, *p* ≈ 0.0077. The mean gain is on the same order as the baseline across-seed noise floor (±0.057) but remains statistically significant under the paired test. Twelve of 18 strong-aug cells still lack seed 2024 locally, so strong-aug cell means use 2–3 seeds as available.

Under strong augmentation at seed 42, ResNet50 achieves the best mean cross-domain macro-F1 (**0.609**) and smallest mean gap (**0.244**). Largest single-seed gain: ResNet50, BRRI → RiceLeafBD, 0.300 → **0.716**. The seed-42 MobileNetV2 BRRI → RiceLeafBD swing (0.573 → 0.273) is **not seed-stable** (two-seed strong cell 0.431 ± 0.224 vs baseline 0.536 ± 0.086) and is not treated as a finding.

### 5.5 Mitigation: LODO

Figures `fig08`, `fig09`; Tables `table_lodo`, `table_mitigation_strategy`, `table_lodo_multiseed`.

**Seed 42.** LODO beats the baseline single-source aggregate in only **3 of 9** target/model cells:

- EfficientNet-B0 → held-out Dhan: 0.317 → 0.489 (+0.171)
- EfficientNet-B0 → held-out RiceLeafBD: 0.474 → 0.541 (+0.067)
- MobileNetV2 → held-out RiceLeafBD: 0.549 → 0.574 (+0.025)

Mean LODO macro-F1 by held-out target (seed 42): RiceLeafBD **0.500**, Dhan-Shomadhan **0.402**, BRRI **0.203**. Best BRRI LODO result is only 0.267 (ResNet50).

**Multi-seed LODO** (three train seeds; `lodo_cell_mean_std.csv`): mean held-out macro-F1 by target—RiceLeafBD **0.464 ± 0.045**, Dhan-Shomadhan **0.380 ± 0.022**, BRRI **0.230 ± 0.031** (models pooled within target). BRRI remains the hardest held-out collection under multi-source training. Adding a second source dataset does not reliably replace careful single-source training plus augmentation on this benchmark.

### 5.6 Mitigation: AdaBN

Table `table_adabn` (full 18 pairs); summary `table_adabn_by_model`.

On the matched unrecalibrated vs AdaBN evaluations of the same seed-42 baseline checkpoints:

| Statistic | Value |
|-----------|------:|
| Mean Δ macro-F1 (AdaBN − baseline) | **−0.055** |
| Pairs improved (Δ > 0) | **5 / 18** |
| Mean Δ, MobileNetV2 | **+0.013** (3/6 improve) |
| Mean Δ, EfficientNet-B0 | **−0.060** (2/6 improve) |
| Mean Δ, ResNet50 | **−0.119** (0/6 improve) |

Largest gains: MobileNetV2 RiceLeafBD → BRRI (+0.190), MobileNetV2 Dhan → BRRI (+0.141), MobileNetV2 Dhan → RiceLeafBD (+0.124). Largest losses include EfficientNet-B0 Dhan → RiceLeafBD (−0.270) and ResNet50 BRRI → Dhan (−0.201). AdaBN does **not** systematically recover cross-dataset macro-F1 on this benchmark; improvements are sparse and architecture-specific (MobileNet only), while ResNet50 never improves.

### 5.7 Augmentation bucket ablation

Figure `fig11_ablation_buckets.png`; Tables `table_ablation`, `table_ablation_summary`; note `notes/ablation_interpretation.md`; source `results/ablation/augmentation_ablation.csv` (archived in `frozen_results_v2/`).

Protocol: ResNet50, six transfer pairs, train seed 42, each strong-augmentation bucket applied alone (§4.6). For reference, matched ResNet50 baseline and full strong-aug means over the same six pairs are **0.482** and **0.609** (Δ **+0.127**).

| Bucket | Mean cross macro-F1 | Mean Δ vs baseline | Pairs improved (Δ > 0) |
|--------|--------------------:|-------------------:|------------------------:|
| Geometric (crop, flip, affine) | **0.567** | **+0.085** | 3 / 6 |
| Occlusion (blur, CoarseDropout) | 0.512 | +0.030 | **5 / 6** |
| Photometric (brightness, hue/sat) | 0.508 | +0.026 | 3 / 6 |

The **geometric/background** bucket alone recovers the largest mean gain (+0.085), about **67%** of the full strong-augmentation improvement on these six pairs (+0.127). Photometric and occlusion buckets help less on average when used in isolation, though occlusion improves more individual pairs (5/6) without raising the mean as much. Full strong augmentation still exceeds any single bucket, indicating the bundled transforms are **complementary** rather than redundant.

Largest geometric gain: ResNet50 BRRI → RiceLeafBD, 0.300 → **0.874** (baseline cross F1 from Table baseline). Largest geometric regression vs baseline: Dhan → RiceLeafBD, 0.596 → 0.548. Mechanism claims should emphasize the **mean geometric advantage** and background randomization, not single-pair swings.

### 5.8 Multi-seed variability and inferential statistics

Figures `fig10_aug_paired_seed.png`, `fig13_seed_std_heatmap.png`; Tables `table_summary_stats_multiseed`, `table_transfer_multiseed`; artifacts `results/stats/` (`python -m run_stats`).

Headline inferential results (full detail in `table_summary_stats_multiseed` and `stats_tests.csv`):

| Test | Result |
|------|--------|
| Augmentation vs baseline (18 cell-mean paired Δ) | mean **+0.063**, 14/18 positive, Wilcoxon *W* = 26, *p* ≈ 0.0077 |
| Baseline cross macro-F1 across-seed noise | mean ± **0.057** per cell |
| AdaBN vs baseline (18 seed-42 pairs) | mean **−0.055**, 5/18 positive, *W* = 47, *p* ≈ 0.099 |

Bootstrap 95% CIs for per-run macro-F1 are computed from full prediction CSVs when present (`bootstrap_ci.csv`; 19 runs covered at time of writing). We report the three-seed pair **0.445 → 0.502** (Δ +0.063) as the primary augmentation claim and do not mix seed-42-only means into that headline. Twelve strong-aug cells still lack seed 2024 locally.

---

## 6. Discussion

**The gap is real.** Strong in-dataset RiceLeafBD scores coexist with a mean cross-domain macro-F1 near 0.44. Reporting only within-collection accuracy would overstate deployability across Bangladeshi sources.

**Background and acquisition conditions matter in the settings we could isolate.** The white → field → cross-dataset field ordering for a fixed Dhan-trained model supports the hypothesis that non-disease cues contribute to failure. Grad-CAM is consistent with this story but should not be oversold: border enrichment failed as a clean correctness separator, and white/field labels are available only for Dhan-Shomadhan.

**Strong augmentation is the more reliable simple mitigation, with seed noise acknowledged.** Across three train seeds, mean cross macro-F1 rises from 0.445 to 0.502 (paired Δ +0.063; Wilcoxon *p* ≈ 0.0077; 14/18 cell means positive)—statistically significant despite a baseline noise floor of ±0.057. ResNet50 benefits most under the full strong pipeline. Large single-seed swings (e.g. MobileNetV2 BRRI → RiceLeafBD at seed 42) should not be treated as definitive findings.

**Bucket ablation isolates geometric/background randomization as the dominant single ingredient.** On ResNet50 × six pairs (seed 42), geometric transforms alone raise mean cross macro-F1 from 0.482 to **0.567** (Δ +0.085), recovering about two-thirds of the full strong-aug gain (+0.127 to 0.609). Photometric (+0.026) and occlusion (+0.030) buckets contribute less on average, while full strong augmentation still wins—consistent with complementary mechanisms rather than one transform doing all the work. This supports—but does not prove—the background-randomization account tied to the confound experiment, within the ablation’s single-model, single-seed scope.

**LODO is an honest negative result.** Multi-source training without domain alignment often underperforms, especially for BRRI. Source diversity alone is insufficient when collections differ in class appearance, balance, and capture conditions. Because LODO label spaces differ from pairwise overlaps, we interpret LODO as a strategy-level finding, not a matched-class effect.

**AdaBN is likewise an honest weak/negative baseline.** Recalibrating BatchNorm statistics on unlabeled target-train images fails on average (mean Δ −0.055; only 5/18 pairs improve) and never helps ResNet50. If domain shift were primarily a matter of mismatched BN appearance statistics, AdaBN should have recovered a substantial fraction of the gap. The observed pattern instead suggests that cross-dataset failure on this benchmark is not reducible to BatchNorm covariate shift alone—or that source features after standard fine-tuning do not transfer under BN recalibration. Together with LODO, AdaBN strengthens the ranking that **train-time strong augmentation remains the best of the three simple interventions** we tested.

**Practical implication.** For Bangladeshi rice disease tools that may see images from multiple collections, practitioners should (i) measure cross-dataset transfer explicitly, (ii) treat background/acquisition as a risk factor, and (iii) prefer aggressive train-time augmentation before assuming that pooling datasets or lightweight BN adaptation will fix generalization.

### 6.1 Future work: testing background dependence causally

Both strands of our background evidence are **associational**. The confound experiment
(§5.3) varies the evaluation set rather than the image content, and the bucket ablation
(§5.7) shows that geometric/background randomization is the largest single contributor to
the augmentation gain (Δ +0.085 of +0.127 on six ResNet50 pairs) without isolating
background as the manipulated variable. Two experiments would convert this into a causal
test, and we flag them as the natural next phase rather than as results:

**Experiment A — background substitution.** Segment leaf and lesion regions from a
class-balanced sample of field images and composite them onto (i) a uniform white
background and (ii) a resampled field background, holding the foreground fixed. Evaluating
a fixed source-trained checkpoint across the resulting matched sets varies background while
holding lesion content constant, which the present design cannot do. Recovering the
white > field ordering of §5.3 under substitution would implicate background causally;
failing to recover it would indicate that the §5.3 ordering reflects correlated acquisition
differences (camera, focal distance, capture protocol) rather than background alone.

**Experiment B — background-targeted augmentation.** Compare the geometric bucket against
an augmentation that explicitly randomizes background while leaving the leaf region
untouched, using the segmentation masks from Experiment A. Because the geometric bucket
also perturbs scale, translation, and framing, it does not separate background
randomization from general spatial invariance; a background-only arm would. We state no
expected effect size: the ablation constrains which mechanism is *largest*, not how much
of the residual gap a targeted intervention could close.

Both experiments require segmentation masks that the current corpus does not include, and
Experiment A additionally assumes segmentation quality sufficient not to introduce its own
artifacts — a confound that would need its own control.


---

## 7. Limitations

- **Frozen splits; partial strong-aug seed coverage.** Stratified splits remain frozen at split seed 42. Train-seed replication (42, 7, 2024) is complete for baseline/default transfer and LODO; **12 of 18** strong-augmentation cells still lack seed 2024 locally, so strong-aug mean ± std uses 2–3 seeds per cell. Bootstrap CIs from full per-sample predictions depend on restoring all prediction bundles.
- **Three architectures, ImageNet initialization.** Results may not transfer to transformers or heavier domain-adaptation methods (e.g., adversarial alignment).
- **Shared-class subsets only.** Classes unique to one dataset cannot appear in pairwise transfer by construction.
- **LODO vs single-source is strategy-level**, not class-matched.
- **Diagnosis scope.** Background confound and Grad-CAM focus on one representative transfer (Dhan → RiceLeafBD, ResNet50, two classes). White/field background labels exist only for Dhan-Shomadhan, so the confound cannot be extended symmetrically to all pairs without new annotations.
- **Grad-CAM overlay availability.** Sample records are frozen (`gradcam_records.csv`); the Week 6 overlay PNG was not re-bundled locally for Week 8 figures and is noted as a provenance gap.
- **AdaBN scope.** AdaBN uses default-augmentation baseline checkpoints only (not strong-aug or LODO models) and seed 42 only; positive MobileNet cells should not be over-generalized.
- **Ablation scope.** Bucket ablation is ResNet50 × seed 42 only; bucket rankings may not transfer to MobileNet/EfficientNet or other seeds.
- **No field deployment study.** Results are offline evaluations on public research collections.

---

## 8. Conclusion

We assembled a reproducible Bangladeshi rice leaf disease cross-dataset benchmark with frozen splits, a transfer matrix, a background-aware diagnosis, and three controlled mitigations. Across three train seeds, mean cross-dataset macro-F1 is 0.445 under default augmentation versus 0.502 under strong augmentation (paired Δ +0.063; Wilcoxon *p* ≈ 0.008; 14/18 cell means positive)—statistically detectable but modest relative to the baseline across-seed noise floor (±0.057). A background-confound experiment supports acquisition/background contribution to the failure. LODO usually does not help and AdaBN fails on average (mean Δ −0.055; 5/18 pairs improve). A ResNet50 bucket ablation attributes most of the strong-aug mean gain to geometric/background randomization alone (Δ +0.085 vs baseline on six pairs). The study’s primary value is an evidence-backed warning against single-dataset optimism and a clear ranking of simple mitigations on this frozen benchmark.

---

## 9. Reproducibility

| Artifact | Path |
|----------|------|
| Frozen CSVs + audit (v1) | `frozen_results/` |
| Replacement freeze (v2) | `frozen_results_v2/` + `notes/freeze_v2_changelog.md` |
| Figures | `paper/figures/fig01`–`fig13` |
| Tables (CSV/LaTeX) | `paper/tables/` (including `table_transfer_multiseed`, `table_lodo_multiseed`, `table_summary_stats_multiseed`) |
| Multi-seed aggregates | `week11_results/multiseed/` |
| AdaBN results | `week12_results/adabn/`, `paper/tables/table_adabn.csv` |
| Ablation | `results/ablation/`, `notebooks/kaggle_week13.md` |
| Statistics | `results/stats/`, `notebooks/kaggle_week14.md` |
| Freeze commands | `python -m freeze_results`, `python -m freeze_results_v2`, `python scripts/numerical_freeze_audit.py` |
| Commit-hash note | `notes/missing_commit_investigation.md` |
| Checkpoint status | `notes/kaggle_checkpoint_verification.md` |
| Figure commands | `python -m make_figures`, `python -m make_figures --revision` |
| Runners | `python -m run_adabn`, `python -m run_ablation`, `python -m run_stats` |
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
| Fig. 10 | `fig10_aug_paired_seed.png` | §5.8 |
| Fig. 11 | `fig11_ablation_buckets.png` | §5.7 |
| Fig. 12 | `fig12_adabn_delta.png` | §5.6 |
| Fig. 13 | `fig13_seed_std_heatmap.png` | §5.8 |

Optional: restore Grad-CAM grid from Kaggle Week 6 bundle as Fig. 14 if available.

---

## References

Barbedo, J. G. A. (2018). Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification. *Computers and Electronics in Agriculture*, 153, 46–53.

Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311–318.

Hasan, A., Layes, T. A., Afridi, A. S., Rifat, S. H., Nur, F. N., & Moon, N. N. (2025). A comprehensive dataset of rice leaf images for disease detection using machine learning. *Data in Brief*, 62, 111977. https://doi.org/10.1016/j.dib.2025.111977

Hossain, M. F., Abujar, S., Noori, S. R. H., & Hossain, S. A. (2021). Dhan-Shomadhan: A dataset of rice leaf disease classification for Bangladeshi local rice. Mendeley Data, V1. https://doi.org/10.17632/znsxdctwtt.1

Li, Y., et al. (2016). Revisiting batch normalization for practical domain adaptation. *arXiv:1603.04779*.

Liu, B., & Wang, Y. (2021). Plant disease detection and classification using deep learning: A review. *IEEE Access*, 9, 56683–56698.

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.

Quinonero-Candela, J., et al. (2009). *Dataset Shift in Machine Learning*. MIT Press.

Rimi, S. A., Chowdhury, M. J. U., Abdullah, R., Ahmed, I., Mim, M. A., & Rahman, M. S. (2025). RiceLeafBD: A real-field image dataset for rice leaf disease detection and classification in Bangladesh. Mendeley Data, V1. https://doi.org/10.17632/kx9rx8p2mz.1

Rust, F., et al. (2020). Agriculture-Vision: A large aerial image database for agricultural pattern analysis. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, 57–66.

Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision*, 618–626.

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), 60.

Too, E. C., et al. (2019). A comparative study of fine-tuning deep learning models for plant disease identification. *Computers and Electronics in Agriculture*, 161, 272–279.

Zhou, K., et al. (2022). Domain generalization: A survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(4), 4396–4415.
