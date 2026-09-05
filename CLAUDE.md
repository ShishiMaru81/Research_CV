# PROJECT WORKFLOW — RICE LEAF CROSS-DATASET GENERALIZATION

**Repository:** `Research_CV`
**Author:** Anindya Paul, BRAC University
**Paper:** Cross-Dataset Generalization of Bangladeshi Rice Leaf Disease Classifiers:
Benchmark, Diagnosis, and Mitigation
**Preprint DOI:** 10.5281/zenodo.21787018
**Status:** NOT submitted. Camera-ready pass complete; venue still open (see §8).
Corrected 2026-08-28: the previous "Submitted" line was stale and was the premise
behind two task briefs that proposed a second, concurrent submission.

> **Place this file at the repo root as `CLAUDE.md` (Claude Code) or `.cursorrules`
> (Cursor) so it loads automatically as standing context.**

---

# 0. READ THIS FIRST — HARD RULES

These exist because each one was learned from an actual failure in this project.

### 0.1 Never trust a stated fact without verifying it

In August 2026, three "facts" supplied in a task brief were wrong:
- A frozen commit hash that did not exist in the repository
- A git tag `week-9-manuscript` that did not exist (repo has no tags)
- A manifest structure that did not match the actual file

**Rule:** paths, hashes, commits, tags, column names, and row counts stated in any brief
are *claims to check*, not facts. Verify before building on them. If a stated premise is
wrong, **stop and report it** rather than working around it silently.

### 0.2 Never trust a number handed to you — recompute it

In the same session, corrected statistics were supplied that had themselves been derived
from a corrupted file. Pasting them would have shipped the exact defect being fixed.

**Rule:** any statistic entering the manuscript is recomputed from authoritative source
files in this repo. Numbers quoted in conversation are starting points, never sources.

### 0.3 Frozen directories are read-only

`frozen_results/`, `frozen_results_v2/`, and `paper/tables/` are never modified by
analysis code. Analysis writes to its own new directory.

To change a frozen result you must: re-run the generating script, re-freeze, update the
manifest, and record the change in the changelog. Never edit in place.

### 0.4 An audit that cannot fail is not an audit

`freeze_results_v2.py` once ran `shutil.copy2(src, dst)` then compared `src` to `dst` —
comparing a file to its own copy. Eleven checks reported PASS and could not have done
otherwise. A 0.23 discrepancy sat inside a green audit for weeks.

**Rule:** every check must have a construction under which it fails. When writing a
check, state what would make it fail. If nothing would, it isn't a check.

### 0.5 Report negative and weakening results

This paper's credibility rests on keeping results that don't flatter it: LODO fails
(3/9 cells), AdaBN fails (mean −0.055, 0/6 on ResNet50), augmentation clears the noise
floor only narrowly. **Never soften, omit, or round these away.**

---

# 1. AUTHORITATIVE SOURCES

## 1.1 Which file wins

| Purpose | Authoritative file | Notes |
|---|---|---|
| Manifest | `frozen_results_v2/manifest.csv` | 5419 rows; 8 copies exist, 4 byte-identical |
| Baseline transfer | `frozen_results_v2/generalization_gap.csv` | Full precision (11–12 dp) |
| Augmented transfer | `frozen_results_v2/generalization_gap_aug.csv` | 18 rows |
| Multi-seed | `paper/tables/table_transfer_multiseed.csv` | 42 rows; **post-fix only** |
| In-dataset | `frozen_results_v2/indataset_results.csv` | 9 rows |

**Do not use** the repo-root `generalization_gap_aug.csv` — it is a 5-row Week-11
partial.

## 1.2 Precision as a provenance signal

A real diagnostic discovered in this project:

| Series | Decimal places | Origin |
|---|---|---|
| Frozen `cross_macro_f1` | 11–12 | Computed from predictions |
| Log-scraped seed-42 default | 2–4 | **Read from printed log text** |
| Re-evaluated seed-42 strong | 4–6 | Computed from checkpoints |

**Low decimal precision means a value was scraped from a log, not computed.**
If `gap != in_dataset_f1 - cross_f1` at ~1e-4, that's rounding residue from scraping.

Check this whenever two files disagree. It settles provenance faster than timestamps.

## 1.3 The `keep="first"` invariant

`scripts/rebuild_multiseed_summary.py:442-447` concatenates frozen seed-42 rows **first**,
then deduplicates with `keep="first"`.

It previously used `keep="last"`, which silently discarded the frozen rows the comment
claimed to be attaching, replacing them with log scrapings. Effect: BRRI→RiceLeafBD
MobileNet seed 42 read 0.448 instead of the true 0.573.

**Any change to this dedup logic must be justified in a commit message and verified by
`numerical_freeze_audit.py`.**

---

# 2. CURRENT HEADLINE NUMBERS (post-fix, use these)

## 2.1 Primary metric: cross-dataset macro-F1

**Report cross-dataset macro-F1 as primary. Generalization gap is secondary and must
always appear alongside its in-dataset baseline.**

Reason: gap = in − cross, and in-dataset F1 ranges 0.638–0.99 across sources, so gap
partly measures source task difficulty rather than transfer failure. Cross-F1 spread
across sources is 0.035; gap spread is 0.203 (5.7×), and the rankings invert —
RiceLeafBD is worst by gap but best by cross-F1. 36.6% of pairwise comparisons are
discordant between the two orderings.

## 2.2 The augmentation result (3-seed, authoritative)

```
Baseline cross-dataset macro-F1   0.445
Strong-aug cross-dataset macro-F1 0.502
Mean paired delta                 +0.063
Wilcoxon                          W = 26, p ≈ 0.0077
Cells improved                    14 / 18
Across-seed noise floor           ±0.057
```

**Framing constraint — this matters.** Δ +0.063 against a noise floor of ±0.057 means
the effect *barely clears seed variability*.

Permitted phrasing:
> "improves cross-dataset macro-F1 by +0.063 on average (14/18 pairs, W=26, p≈0.008),
> a margin comparable to across-seed variability (±0.057)"

**Forbidden phrasing:** "substantially improves", "consistently improves", "large gain",
or any wording implying the effect is comfortably above noise.

The Discussion must state plainly that the improvement is statistically detectable but
modest relative to seed noise.

## 2.3 Seed-42 reference values (do not mix into headline)

```
Baseline 0.436 → Strong 0.503, delta +0.0669, Wilcoxon p = 0.0139
```

These are internally consistent and may be cited as seed-42 reference. **Never combine
seed-42 means with 3-seed test statistics** — that mismatch is what produced the original
arithmetic incoherence (stated Δ 0.067 vs tested Δ 0.0701).

## 2.4 Negative results — preserve exactly

```
LODO         improves only 3/9 cells; BRRI collapses to 0.203 (0.230 ± 0.031 over 3 seeds)
AdaBN        mean delta −0.055; 5/18 pairs positive; ResNet50 0/6; W=47, p≈0.099
Background   white 0.854 > field 0.705 > cross 0.573 (prespecified ordering met)
Ablation     geometric bucket alone: cross-F1 0.567, Δ+0.085, ~67% of full aug gain
```

---

# 3. REPOSITORY MAP

```
Research_CV/
├── CLAUDE.md                          # this file
├── .cursorrules                       # same standing context for Cursor
│
├── frozen_results/                    # v1 freeze — READ-ONLY, 11/11 hashes verified
├── frozen_results_v2/                 # v2 freeze — READ-ONLY, 19/19 hashes verified
│   ├── manifest.csv                   # AUTHORITATIVE (5419 rows)
│   ├── generalization_gap.csv         # AUTHORITATIVE baseline
│   ├── generalization_gap_aug.csv     # AUTHORITATIVE augmented
│   ├── crossdataset_matrix.csv
│   ├── indataset_results.csv
│   ├── lodo_results.csv
│   ├── background_confound.csv
│   ├── gradcam_records.csv
│   └── freeze_manifest.json
│
├── paper/
│   ├── manuscript.md                  # §4.7–4.8 contain corrected audit wording
│   ├── figures/fig01–fig13.png
│   └── tables/*.csv, *.tex
│
├── scripts/
│   ├── rebuild_multiseed_summary.py   # keep="first" invariant at :442-447
│   └── numerical_freeze_audit.py      # 12 checks that CAN fail
│
├── freeze_results_v2.py               # audit wording corrected (repo root)
├── run_stats.py                       # Wilcoxon on cross_macro_f1 (repo root)
│
├── notes/
│   ├── missing_commit_investigation.md
│   ├── kaggle_checkpoint_verification.md
│   ├── numerical_freeze_audit.md
│   └── week*_plan.md, week*_retro.md
│
├── feasibility_check/                 # planned / optional — segmentation viability
├── paradigm_analysis/                 # planned / optional — NEGATIVE result — see §6.1
├── metric_validity/                   # planned / optional — gap metric investigation
└── discrepancy_investigation/         # planned / optional — baseline provenance trace
```

---

# 4. STANDARD WORKFLOWS

## 4.1 Any analysis task

```
1. Locate files — do not assume paths. Report which copy you used and whether
   duplicates agree.
2. Print columns and row counts for confirmation before analysing.
3. Verify every premise in the brief. Report mismatches; do not work around them.
4. Write outputs ONLY to a new directory. Never touch frozen dirs.
5. Flag every subgroup with n < 5, inline, every time the statistic appears.
6. State the strongest counterargument to the conclusion.
7. If evidence is ambiguous, say so. Do not manufacture a verdict.
```

## 4.2 Statistics

- **Wilcoxon signed-rank**, not t-test (n=18, normality not assumable)
- Run on **`cross_macro_f1`**, not on gap
- Always report: mean delta, n positive / n total, W, p, **and the noise floor**
- Report effect size alongside p; with n<5 treat p as secondary to the distribution
- Multiple comparisons: report p with effect sizes, note Bonferroni threshold as a
  robustness check rather than the primary criterion

## 4.3 Changing anything in the manuscript

```
1. Recompute the number from an authoritative file (§1.1)
2. Grep the ENTIRE manuscript for every instance of the old value
3. Update abstract, results, discussion, conclusion, AND tables together
4. Re-run scripts/numerical_freeze_audit.py  → must stay 12/12
5. Re-run the writing audit                  → must stay 22/22
6. Record in the changelog
```

Never update one location. The 0.436/0.4408 incoherence came from partial updates.

## 4.4 GPU work

All GPU work runs on **Kaggle** (30 hrs/week free). Local machine is CPU-only for
analysis. Checkpoints live on Kaggle; `results/checkpoints/` is gitignored and empty
locally — **0/144 verifiable locally, 90/90 complete per registry.**

Kaggle scripts must be resumable and idempotent:
```python
out = Path(f"cache/{run_id}.json")
if out.exists():
    continue        # skip completed work
```

For large sweeps use **extract-then-delete**: compute all metrics from a checkpoint,
save the numbers, delete the weights. Never accumulate checkpoints.

---

# 5. OPEN ITEMS

| Priority | Item | Notes |
|---|---|---|
| **HIGH** | Run Kaggle checkpoint verification cell | 0/144 verified locally; procedure in `notes/kaggle_checkpoint_verification.md`; save `checkpoint_verification.csv` |
| MEDIUM | Confirm §2.2 framing applied everywhere | Noise-floor caveat must appear in Abstract AND Discussion |
| MEDIUM | Decide re-framing around cross-F1 as primary metric | See §6.2 |
| LOW | Rebuild v1 audit with fail-capable checks | v2 done; v1 still uses old style |

---

# 6. INVESTIGATION HISTORY

Recorded so these are not repeated.

## 6.1 Acquisition-paradigm hypothesis — REJECTED

Hypothesis: cross-paradigm transfers (studio ↔ canopy) fail worse than within-paradigm.

**Rejected. Two fatal confounds:**
1. `within_studio` has **n=0** — the control condition does not exist in this design
   (no self-pairs, only one studio dataset)
2. `pairing_type` is **perfectly confounded with class set** — all cross-paradigm pairs
   are `healthy|tungro`; all mixed pairs are the 2- or 3-class sets

Also: gap differed +0.104 between groups while cross-F1 differed +0.003 — the gap metric
manufactured the apparent effect. This is what triggered §2.1.

**Do not revive without new datasets.** With 3 datasets, paradigm ≡ dataset identity.

## 6.2 Gap metric — COMPROMISED as primary

Note: the *correlation* argument for this is **invalid** — since gap = in − cross,
correlation with in-dataset F1 is algebraically guaranteed (independence floor +0.771;
observed +0.648 is *below* it).

The valid argument is §2.1: spread ratio 5.7×, rank inversion, 36.6% discordance.

## 6.3 Segmentation feasibility — ASYMMETRIC DESIGN ONLY

```
dataset          coverage  bg_uniformity  IoU   verdict    role
brri              11.6%        16.1       0.62  MODERATE   GOOD leaf source
dhan_shomadhan    14.3%        30.7       0.45  HARD       white=usable, field=fails
riceleafbd        30.7%        47.9       0.31  HARD       target background ONLY
```

- Standard 30–70% coverage thresholds **do not fit rice morphology** (thin blades →
  real coverage 8–31%). Coverage is not the constraint; every dataset has 69–88%
  non-leaf frame.
- **RiceLeafBD is in-canopy** — background is more rice foliage (39% border vegetation).
  A background swap there is conceptually ill-defined: substituting one canopy for
  another isolates nothing.
- Dhan-Shomadhan's aggregate HARD is a **mixing artifact**: white (n=14) IoU 0.62 vs
  field (n=6) IoU 0.05.
- **Design implication:** cut leaves from BRRI/Dhan-white onto field backgrounds.
  Asymmetric, not a symmetric swap.
- **Prefer leaf REMOVAL over leaf SWAP.** In a swap, mask error leaves leaf fragments
  in the "background" and inflates the effect — segmentation error works *for* you,
  which reviewers will attack. In removal, dilate generously and over-erase; residual
  signal is unambiguously background. Error works *against* you. Given IoU 0.31–0.74,
  that conservatism matters.
- `background` column is populated **only** for Dhan-Shomadhan (769 white / 337 field).
  Empty for all 2753 BRRI and 1560 RiceLeafBD rows.

## 6.4 Dhan 2×2 split — NOT VIABLE per-class

- χ²=19.12, p=0.00075, Cramér's V=0.117 (small but real class/background dependence)
- Dhan-white test: only 1 of 5 classes reaches n≥30. Dhan-field test: **0 of 5**;
  largest class n=14
- Only `tungro` is universal across all four sources; base rates 8.7%→34% make macro-F1
  non-comparable even on the "shared" task
- Breaks the paradigm confound but **not** the class-set confound
- Viable variant only: pool 337 field images as a single fixed evaluation target,
  restricted to `rice_blast|scald|tungro`

## 6.5 Missing commit — RESOLVED

Not a repo re-initialization. History was `filter-branch` rewritten on 2026-08-02 to
strip Cursor co-authors; old SHAs died.

- Freeze stamp `13b8552` = HEAD when freeze ran (Week-7 tip)
- Week-8 `88f8c5d` → now `ac77dc2`
- Full trace: `notes/missing_commit_investigation.md`

---

# 7. WRITING RULES

## 7.1 Framing

- This is an **empirical benchmark and diagnostic study**, not a novel-methods paper.
  Frame it that way. Do not claim methodological novelty.
- "First" claims are defensible **only** for this specific triplet of Bangladeshi
  datasets. Nothing broader.
- Cite and differentiate from DG-PLDR / SDPM. Positioning: low-resource, interpretable,
  train-time-only intervention outperforming lightweight test-time adaptation (AdaBN).

## 7.2 Forbidden

- Any claim exceeding what the data supports
- Numbers not traceable to an authoritative file
- SOTA claims
- Describing the audit as "numerical reproduction" — it verifies file-copy and SHA-256
  integrity only
- "Substantially/consistently improves" for the augmentation result (§2.2)
- Mixing seed-42 means with 3-seed test statistics

## 7.3 Required

- Every headline number traceable to §1.1
- Noise floor reported alongside every improvement claim
- All negative results retained (§2.4)
- Limitations section genuinely uncomfortable to write
- AI writing assistance disclosed per venue policy

---

# 8. VENUES

| Venue | Cost | Fit |
|---|---|---|
| CVPR Agriculture-Vision Workshop | Free submit; registration to present | Current submission |
| Computers and Electronics in Agriculture | Free (subscription route) | Best journal fit; where Barbedo/Ferentinos/Too published |
| Pattern Recognition | Free (subscription route) | If framed as domain generalization |
| TMLR | **Fully free**, rolling, no registration | Emphasizes correctness over significance — fits this paper's profile |

**Avoid:** IJCV/TPAMI (wrong scope, needs methodological novelty); MDPI *Remote Sensing*
(~2500 CHF APC).

**Note:** TMLR forbids overlap with previously published work. Decide TMLR *before*
submitting elsewhere.

---

# 9. THE STANDING INSTRUCTION

> Be skeptical of the brief, not supportive of it. A negative result found in an hour is
> worth more than a confirmation that costs three weeks. If the evidence doesn't support
> the hypothesis, say so plainly. Do not soften a bad finding.

Every significant advance in this project came from a check that returned a negative:
the paradigm hypothesis died in an hour, the gap metric problem surfaced from a
discarded subgroup analysis, and the `keep="last"` bug was found behind an audit that
reported green.
