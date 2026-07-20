# Weeks 10–14: Post-Review Revision Workflow

**Goal:** take the manuscript from "borderline, likely major revision" (54/100) to "credible accept at applied venue" (~68–72/100), without redesigning the study.

**Constraint reality:** Kaggle free tier ≈ 30 GPU-hrs/week. Everything below is budgeted against that. Solo, no co-authors, no mentor.

---

## PART 0 — The logic of the ordering (read this first)

You have 10 improvement items from the review. They are **not** independent, and doing them in the wrong order wastes GPU hours you can't get back. Three rules determine the sequence:

**Rule 1 — Instrumentation before compute.**
Bootstrap confidence intervals require *per-sample predictions* for every evaluation run. If you re-run 90 training jobs and only save summary metrics, you cannot compute CIs afterward without re-running everything. So the logging fix must land **before** the first multi-seed run. This is the single most expensive mistake available to you right now.

**Rule 2 — Free evidence before paid evidence.**
Some review items need no training at all:
- Extending the background-confound diagnosis = *evaluation only* on existing checkpoints.
- AdaBN (BatchNorm statistic recalibration) = *forward passes only*, no gradient updates.
- The 1,555/1,560 fix, the border-attention formula, the related-work expansion = pure writing.

These cost near-zero GPU and close three reviewer objections. Do them while multi-seed jobs are queued, not after.

**Rule 3 — Freeze discipline is non-negotiable.**
Your `data_rules.md` and Week 8 freeze policy say frozen CSVs are immutable and corrections require a *documented replacement freeze*. You are about to add seeds, a new baseline, and an ablation. That is exactly a replacement freeze. Do **not** append rows to `frozen_results/`. Create `frozen_results_v2/` and run a fresh audit. Your reviewers (and your future self) will care.

---

## PART 1 — Phase map

| Phase | Days | GPU cost | Closes review item |
|---|---|---|---|
| 0. Instrumentation + writing fixes | 2–3 | ~0 hrs | #2, #9, #7, #10 |
| 1. Multi-seed replication | 7–10 | ~20–26 hrs | **#1 (critical)** |
| 2. AdaBN baseline + diagnosis extension | 3–4 | ~2–3 hrs | #3, #4 |
| 3. Augmentation bucket ablation | 4–5 | ~6–8 hrs | #6 |
| 4. Stats layer + freeze v2 + figures | 3–4 | ~0 hrs | #1, #5, #8 |
| 5. Manuscript revision + submission | 5–7 | 0 hrs | all writing items |

Total: ~4–5 weeks part-time, ~30–37 GPU-hrs. Fits comfortably in the free tier if spread across 2–3 Kaggle weeks.

---

## PHASE 0 — Instrumentation and free wins (Days 1–3)

### 0.1 Why this phase exists
You are about to spend 25+ GPU hours. Everything those hours produce is only as useful as what `eval.py` writes to disk. Right now it writes a metrics JSON and *sample* predictions. You need **all** predictions.

### 0.2 What to change

**(a) `src/eval.py` — dump full per-sample predictions.**
For every eval run, write `predictions/{run_id}.csv` with columns:
`image_path, dataset, true_index, pred_index, true_label, pred_label, correct, prob_0..prob_k`

Why: this one file lets you compute, *after the fact and with zero extra GPU*, bootstrap CIs, per-class F1, confusion matrices, McNemar tests between two models on the same test set, and calibration curves. Without it, each of those needs a re-run.

**(b) `src/train.py` / `src/eval.py` — seed plumbing.**
Confirm `--seed` propagates to: Python `random`, NumPy, `torch`, `torch.cuda`, the DataLoader worker seeding (`worker_init_fn`), and the shuffle generator. A partially-seeded pipeline produces fake variance that will corrupt your entire statistics story.

Important: the **data split stays frozen at seed 42.** You are varying *training* stochasticity (init, shuffling, augmentation sampling), not the split. Say this explicitly in the paper — otherwise a reviewer will think you re-split and reintroduced leakage risk. Add a `--split_seed` (fixed 42) separate from `--train_seed` if the current code conflates them.

**(c) Run registry.**
`results/run_registry.csv`: one row per run with `run_id, experiment_type, model, train_datasets, eval_dataset, classes, train_seed, augmentation, status, checkpoint_path, predictions_path`. Every script writes to it, and every script checks it before launching. This is what makes a 90-run campaign resumable across Kaggle session timeouts instead of a guessing game.

**(d) Naming convention.** Extend existing tags: `{model}__train-{src}__seed{N}__aug-{none|strong|bucket-geo|...}`. Collision with Week 5/7 artifacts must be impossible.

### 0.3 Writing fixes (do them now, they take an hour)
- RiceLeafBD count: reconcile 1,555 (Section II) vs 1,560 (Table I / Section III). Your inventory gives 422+356+252+530 = 1,560. Recommended text: *"RiceLeafBD contributes 1,560 original images in the released archive (the source descriptor reports 1,555; we use the archive count)."*
- Border-attention enrichment: write the actual formula. Something like — let `H` be the normalized Grad-CAM map, `B` the border region (state the width, e.g. outer 15% of each dimension), then `enrichment = mean(H[B]) / mean(H)`. State the band width explicitly.
- Add a sentence on the RiceLeafBD macro-F1 > accuracy pattern (class-weighted CE lifting minority-class recall relative to overall accuracy).

### Definition of done — Phase 0
- [ ] One test eval run produces a complete `predictions/*.csv`
- [ ] Same seed twice → byte-identical metrics; different seed → different metrics
- [ ] `run_registry.csv` populated and resume-checked
- [ ] Three text fixes committed
- [ ] Tag: `week10-instrumentation`

---

## PHASE 1 — Multi-seed replication (Days 4–13)

### 1.1 Why this is the critical item
Every headline claim in your abstract is a single point estimate. "Augmentation improves 14/18 pairs" is currently indistinguishable from "augmentation improves 14/18 pairs *at seed 42*." A reviewer cannot accept a comparison claim without knowing the noise floor. This one phase moves your score more than the other nine items combined.

### 1.2 Scope decision (this is where you save your GPU budget)

You do **not** need every experiment at every seed. Prioritize by whether the claim is *comparative*:

| Experiment | Seeds needed | Why |
|---|---|---|
| Cross-dataset baseline (18 runs) | 3 | Feeds the gap claim AND is the reference for aug |
| Strong augmentation (18 runs) | 3 | The core comparative claim |
| LODO (9 runs) | 3 | The negative result — needs variance most of all |
| In-dataset baselines (9 runs) | 1 (existing) | Descriptive context, not a contested comparison |

**Seeds: 42 (done), 7, 2024.** Three is the minimum that lets you report mean ± std honestly. Five is better if budget allows; add seed 123 and 999 later only if Phase 3 comes in under budget.

**Run count:** (18 + 18 + 9) × 2 new seeds = **90 training runs.**
At ~12–15 min/run on T4 → **18–23 GPU-hrs.** Spread over 2 Kaggle weeks.

### 1.3 Execution order
Run **seed-major, not experiment-major**: finish all of seed 7, then all of seed 2024. If you run out of budget halfway, you have two complete seeds (usable: mean ± std) rather than three partial ones (usable: nothing).

### 1.4 The honest-reporting trap to avoid
When you get three seeds, some of your Week 7 headline numbers **will move**, and some may reverse sign. The BRRI→RiceLeafBD MobileNetV2 regression (0.573 → 0.273) is a prime candidate — a 0.30 swing on a small target set is exactly what seed noise looks like. If it turns out to be noise, you must say so and remove it as a "finding." Rewriting a result you were proud of is the job. A reviewer who catches you keeping a seed-42 anecdote after you had the variance data will discount the whole paper.

### Definition of done — Phase 1
- [ ] 45 runs × 3 seeds present in `run_registry.csv`, status=complete
- [ ] Full predictions CSV for every run
- [ ] Quick sanity check: seed-42 rows reproduce the frozen Week 5/7 values
- [ ] Tag: `week11-multiseed`

---

## PHASE 2 — AdaBN baseline + diagnosis extension (Days 14–17)

### 2.1 AdaBN — why this is the best value in the whole plan
The reviewer wants a domain-adaptation baseline. Most DA methods require retraining. **AdaBN does not.** You take an existing trained checkpoint, run forward passes over the *unlabeled target training images* with BatchNorm layers in training mode so their running mean/variance update to target statistics, then freeze and evaluate. No labels, no gradients, no optimizer.

Cost: ~1–2 min per checkpoint. For 18 baseline checkpoints × 3 seeds = 54 evaluations ≈ **1.5 hrs.**

Why it strengthens the paper conceptually, not just procedurally: AdaBN adapts *only* the feature normalization statistics. If your background-confound hypothesis is right — that the model is keying on dataset-level appearance statistics — then AdaBN should recover a meaningful chunk of the gap. If it recovers nothing, that is *evidence against your own hypothesis* and you should report it. Either outcome is a real result. This turns a box-ticking exercise into a genuine third mitigation arm.

Implementation caution: use the target dataset's **train** split for statistic estimation, never the test split. Document this. A reviewer will check.

### 2.2 Diagnosis extension — and its hard limit
The reviewer asked you to extend the background-confound experiment beyond one setting. **You cannot fully comply, and you should say why.** The white/field background labels exist only for Dhan-Shomadhan. So the confound experiment is structurally restricted to Dhan-trained checkpoints.

What you *can* do:
1. Same experiment for MobileNetV2 and EfficientNet-B0 (currently only ResNet50) → tests whether the ordering is architecture-specific. Cost: eval-only on existing checkpoints.
2. Same experiment for the Dhan → BRRI direction (classes: rice_blast, scald, tungro) → tests whether the ordering is transfer-direction-specific.
3. Across all 3 seeds → tests whether the ordering is seed-specific.

That gives you up to 3 models × 2 directions × 3 seeds = 18 confound evaluations instead of 1. If the white > field > cross ordering holds in most of them, your Discussion claim becomes defensible as stated. If it holds only for ResNet50, you must narrow the claim — and that's still a better paper than the current one.

3. Grad-CAM: regenerate overlays (your Week 6 overlay PNG is a known provenance gap) and extend border-enrichment to the *full* target test set, not 12 curated samples. Then report the correct-vs-incorrect enrichment distributions with a Mann–Whitney U test. Currently you say it "does not cleanly separate" — with full-set numbers you can say *how much* it fails to separate, which is a much stronger honest claim.

### Definition of done — Phase 2
- [ ] `adabn_results.csv` — 54 rows
- [ ] `background_confound_extended.csv` — up to 18 rows
- [ ] `gradcam_full_enrichment.csv` — all target test samples, both directions
- [ ] Tag: `week12-adabn-diagnosis`

---

## PHASE 3 — Augmentation bucket ablation (Days 18–22)

### 3.1 Why bucket, not leave-one-out
Full leave-one-transform-out on 6 transforms × 18 pairs × 3 seeds = 324 runs. Impossible on your budget.

Group the pipeline into **three mechanism buckets** that map onto competing explanations:

| Bucket | Transforms | Hypothesis it tests |
|---|---|---|
| **Geometric/background** | RandomResizedCrop, HorizontalFlip, Affine | Background randomization is the active ingredient |
| **Photometric** | BrightnessContrast, HueSaturation | Dataset colour signature is the active ingredient |
| **Occlusion** | GaussianBlur, CoarseDropout | Forcing multi-region reliance is the active ingredient |

Run each bucket alone. This directly tests your own causal story — you claim in Section VI that augmentation works by "randomizing crop, color, blur, and occlusion," but you never established *which*. If geometric alone recovers most of the gain, your background hypothesis gets independent support from a second experiment. That's a much stronger paper than "we bundled six things and it helped."

### 3.2 Scope
3 buckets × 6 pairs × ResNet50 only × 1 seed (42) = **18 runs ≈ 4 hrs.**
If budget allows, add a second seed (+4 hrs). Restrict to ResNet50 because it showed the largest augmentation response — the effect is most detectable there. State this scoping choice explicitly as a limitation.

### Definition of done — Phase 3
- [ ] `augmentation_ablation.csv` — 18+ rows
- [ ] A one-paragraph interpretation naming which bucket dominates
- [ ] Tag: `week13-ablation`

---

## PHASE 4 — Statistics, freeze v2, figures (Days 23–26)

### 4.1 The statistics layer — what test for what claim

| Claim | Test | Why this one |
|---|---|---|
| Augmentation > baseline (18 pairs) | Wilcoxon signed-rank, paired | Same pairs before/after; n=18; no normality assumption |
| LODO > single-source (9 cells) | Wilcoxon signed-rank + report effect size | n=9 is underpowered — report the effect size and say so |
| Per-run macro-F1 uncertainty | Bootstrap over test samples, 1000 resamples | Uses your new per-sample predictions; no retraining |
| Across-seed variability | mean ± std over 3 seeds, per cell | The headline "noise floor" number |
| Confound ordering robustness | Count of settings where ordering holds / total | Simple, honest, no test needed |

**The number that matters most:** compute the mean across-seed standard deviation of cross-dataset macro-F1. Then state in the paper: *"the mean augmentation improvement (+0.067) is [larger/comparable] to the mean across-seed standard deviation (±X)."* If +0.067 is smaller than your seed noise, you must soften the augmentation claim. That sentence, whichever way it lands, is what converts this from an undergraduate report into a paper a reviewer trusts.

Avoid: running a t-test per cell and reporting 18 p-values without correction. If you go per-cell, apply Benjamini–Hochberg and say so.

### 4.2 Freeze v2
New directory `frozen_results_v2/`, fresh `freeze_results.py` run, fresh audit report. The v2 audit should additionally check: seed coverage complete, predictions file present for every run, v1 seed-42 values reproduced within tolerance. Record both v1 and v2 commit hashes. Write `notes/freeze_v2_changelog.md` explaining exactly what changed and why — that document is your defence if anyone questions result provenance.

### 4.3 Figure updates
- Fig. 1, 2, 3, 6, 7, 8: add error bars / annotate cells with ± std
- New: augmentation-vs-baseline paired dot plot with per-seed points visible (this single figure communicates your variance story better than any table)
- New: bucket ablation bar chart
- New: AdaBN as a fourth bar in the mitigation comparison (Fig. 9)
- Replace: full-test-set border-enrichment distribution (correct vs incorrect) instead of 12 curated samples
- Restore: Grad-CAM overlay grid, properly bundled this time

### Definition of done — Phase 4
- [ ] `stats_tests.csv`, `seed_variance.csv`, `bootstrap_ci.csv`
- [ ] `frozen_results_v2/` audit PASS
- [ ] Figures regenerate deterministically twice with identical hashes
- [ ] Tag: `week14-freeze-v2`

---

## PHASE 5 — Manuscript revision (Days 27–33)

Rewrite order (most-dependent-on-new-data first):

1. **Results** — every table gets mean ± std; add AdaBN subsection; add ablation subsection; rewrite any claim the seed data overturned.
2. **Methods** — add AdaBN protocol, ablation protocol, statistics protocol, the split-seed-vs-train-seed clarification.
3. **Discussion** — retitle *"Background and acquisition conditions appear to matter"* to reflect actual scope, e.g. *"Background and acquisition conditions matter in the settings we could isolate."* Fold in what AdaBN and the ablation say about the mechanism.
4. **Limitations** — remove the single-seed limitation (you fixed it), keep and sharpen the rest, add the new ones (ablation on one model/seed; confound restricted to Dhan by data availability).
5. **Related work** — add Mohanty 2016, Ferentinos 2018, Barbedo 2018, Zhou 2022 (DG survey), Li 2016 (AdaBN), plus 1–2 Agriculture-Vision workshop papers. Target ~15–18 references.
6. **Abstract + Intro** — last. Update headline numbers; they will have changed.
7. **Trim** the Results/Discussion redundancy to make room.

Then: arXiv preprint immediately, Agriculture-Vision submission, IEEE Access / PeerJ CS as fallback.

---

## PART 2 — Cursor execution plan

See prompts 0.1–4.3 in the original revision brief. Use one phase per branch; never edit `frozen_results/`; always run `--dry_run` before multi-seed campaigns.

---

## PART 3 — Guardrails

- Kaggle sessions die at ~9 hours — verify resume before the real campaign.
- Download `results/` (including checkpoints) every session.
- Never edit `frozen_results/` — use `frozen_results_v2/` + changelog.
- Re-verify every manuscript number against v2 before submitting.

---

## PART 4 — If you run out of time

Cut order: Phase 3 (ablation) → third seed → extended diagnosis (narrow Discussion instead).

**Never cut:** Phase 0 instrumentation, ≥2 seeds, AdaBN, statistics layer (~12 GPU-hrs minimum viable revision).

**Never add:** fourth dataset or transformer backbone before fixing rigour on existing claims.
