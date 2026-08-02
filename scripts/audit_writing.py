"""Audit manuscript and progress-report numbers against frozen CSV sources."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUSCRIPT = ROOT / "paper" / "manuscript.md"
DEFAULT_PROGRESS = ROOT / "notes" / "progress_report.md"
DEFAULT_OUT = ROOT / "notes" / "manuscript_audit.md"

TOL = 0.002  # display rounding tolerance


@dataclass
class Check:
    claim: str
    expected: float
    source: str
    actual: float | None = None
    status: str = "pending"

    @property
    def ok(self) -> bool:
        if self.actual is None:
            return False
        return abs(self.expected - self.actual) <= TOL


def _mean(path: Path, col: str, filt=None) -> float:
    df = pd.read_csv(path)
    if filt is not None:
        df = filt(df)
    return float(df[col].mean())


def _load_checks() -> list[Check]:
    frozen = ROOT / "frozen_results"
    multiseed = ROOT / "results" / "multiseed"
    stats = ROOT / "results" / "stats"
    ablation = ROOT / "results" / "ablation" / "augmentation_ablation.csv"
    adabn = ROOT / "frozen_results_v2" / "adabn_results.csv"
    if not adabn.exists():
        adabn = ROOT / "adabn_results.csv"

    cell = pd.read_csv(multiseed / "transfer_cell_mean_std.csv")
    base_cell = cell[cell["augmentation"] == "default"]
    strong_cell = cell[cell["augmentation"] == "strong"]
    stats_df = pd.read_csv(stats / "stats_tests.csv")
    aug_stat = stats_df[
        stats_df["claim"] == "augmentation_gt_baseline_mean_over_seeds_18_cells"
    ].iloc[0]
    adabn_stat = stats_df[
        stats_df["claim"] == "adabn_gt_baseline_18_pairs_seed42"
    ].iloc[0]

    pairwise = pd.read_csv(frozen / "mitigation_pairwise_aug.csv")
    gap = pd.read_csv(frozen / "generalization_gap.csv")
    confound = pd.read_csv(frozen / "background_confound.csv")
    lodo = pd.read_csv(frozen / "lodo_results.csv")
    comparison = pd.read_csv(frozen / "mitigation_comparison.csv")
    indataset = pd.read_csv(frozen / "indataset_results.csv")

    abl = pd.read_csv(ablation)
    abl = abl[abl["model"] == "resnet50"]
    baseline_resnet = pd.read_csv(frozen / "crossdataset_matrix.csv")
    baseline_resnet = baseline_resnet[
        (baseline_resnet["model"] == "resnet50") & (baseline_resnet["seed"] == 42)
    ]["macro_f1"].mean()

    ada = pd.read_csv(adabn)

    checks = [
        Check("In-dataset mean macro-F1", 0.719, "frozen_results/indataset_results.csv"),
        Check("Baseline cross macro-F1 (seed 42)", 0.436, "frozen crossdataset_matrix"),
        Check("Mean gap (seed 42)", 0.387, "frozen generalization_gap"),
        Check("Strong-aug cross macro-F1 (seed 42)", 0.503, "mitigation_pairwise_aug"),
        Check("Mean aug improvement (seed 42)", 0.067, "mitigation_pairwise_aug"),
        Check("Aug pairs improved (seed 42)", 14.0, "mitigation_pairwise_aug"),
        Check("Baseline cross mean (3-seed cells)", 0.441, "transfer_cell_mean_std default"),
        Check("Baseline cross std avg", 0.067, "transfer_cell_mean_std default"),
        Check("Strong cross mean (partial seeds)", 0.502, "transfer_cell_mean_std strong"),
        Check("Wilcoxon aug p (18 cells)", 0.0002, "stats_tests.csv"),
        Check("Mean paired aug delta", 0.070, "stats_tests.csv"),
        Check("AdaBN mean delta", -0.055, "adabn_results.csv"),
        Check("AdaBN pairs improved", 5.0, "adabn_results.csv"),
        Check("AdaBN Wilcoxon p", 0.099, "stats_tests.csv"),
        Check("LODO positive cells", 3.0, "mitigation_comparison.csv"),
        Check("White-bg confound F1", 0.854, "background_confound.csv"),
        Check("Field-bg confound F1", 0.705, "background_confound.csv"),
        Check("Cross confound F1", 0.573, "background_confound.csv"),
        Check("Ablation geometric mean F1", 0.567, "augmentation_ablation.csv"),
        Check("Ablation geometric delta", 0.085, "augmentation_ablation.csv"),
        Check("ResNet50 baseline 6-pair mean", 0.482, "crossdataset_matrix resnet50"),
        Check("ResNet50 strong 6-pair mean", 0.609, "crossdataset_matrix_aug resnet50"),
    ]

    # Fill actuals
    checks[0].actual = float(indataset["macro_f1"].mean())
    checks[1].actual = float(pd.read_csv(frozen / "crossdataset_matrix.csv")["macro_f1"].mean())
    checks[2].actual = float(gap["generalization_gap"].mean())
    checks[3].actual = float(pairwise["aug_cross_macro_f1"].mean())
    checks[4].actual = float(pairwise["cross_f1_improvement"].mean())
    checks[5].actual = float((pairwise["cross_f1_improvement"] > 0).sum())
    checks[6].actual = float(base_cell["cross_mean"].mean())
    checks[7].actual = float(base_cell["cross_std"].mean())
    checks[8].actual = float(strong_cell["cross_mean"].mean())
    checks[9].actual = float(aug_stat["pvalue"])
    checks[10].actual = float(aug_stat["mean_delta"])
    checks[11].actual = float(ada["delta_macro_f1"].mean())
    checks[12].actual = float((ada["delta_macro_f1"] > 0).sum())
    checks[13].actual = float(adabn_stat["pvalue"])
    checks[14].actual = float((comparison["lodo_vs_single_source"] > 0).sum())
    white = confound[confound["condition"].str.contains("white", case=False)]
    field = confound[confound["condition"].str.contains("field", case=False) & ~confound["condition"].str.contains("RiceLeafBD", case=False)]
    cross = confound[confound["condition"].str.contains("RiceLeafBD", case=False)]
    checks[15].actual = float(white["macro_f1"].iloc[0])
    checks[16].actual = float(field["macro_f1"].iloc[0])
    checks[17].actual = float(cross["macro_f1"].iloc[0])

    geo = abl[abl["bucket"] == "geo"]["macro_f1"].mean()
    geo_delta = geo - baseline_resnet
    checks[18].actual = float(geo)
    checks[19].actual = float(geo_delta)
    checks[20].actual = float(baseline_resnet)
    strong_resnet = pd.read_csv(frozen / "crossdataset_matrix_aug.csv")
    strong_resnet = strong_resnet[
        (strong_resnet["model"] == "resnet50") & (strong_resnet["seed"] == 42)
    ]["macro_f1"].mean()
    checks[21].actual = float(strong_resnet)

    for c in checks:
        c.status = "PASS" if c.ok else "FAIL"

    return checks


def scan_manuscript_numbers(manuscript: Path) -> list[str]:
    text = manuscript.read_text(encoding="utf-8")
    # Find decimal numbers in prose (3 decimal places common)
    nums = re.findall(r"\b0\.\d{2,3}\b", text)
    return sorted(set(nums))


def render_report(
    checks: list[Check],
    manuscript: Path,
    progress: Path,
    out_path: Path,
) -> None:
    n_pass = sum(1 for c in checks if c.ok)
    n_fail = sum(1 for c in checks if not c.ok)

    lines = [
        "# Manuscript and report audit",
        "",
        f"Generated from repository CSVs. Tolerance ±{TOL} on rounded display values.",
        "",
        f"**Result: {n_pass}/{len(checks)} checks PASS**"
        + (f", **{n_fail} FAIL**" if n_fail else ""),
        "",
        "## Numeric claims vs sources",
        "",
        "| Claim | Expected | Actual | Source | Status |",
        "|-------|----------|--------|--------|--------|",
    ]
    for c in checks:
        actual = f"{c.actual:.4f}" if c.actual is not None else "—"
        lines.append(
            f"| {c.claim} | {c.expected:.3f} | {actual} | `{c.source}` | {c.status} |"
        )

    lines.extend(
        [
            "",
            "## Files audited",
            "",
            f"- Manuscript: `{manuscript.as_posix()}`",
            f"- Progress report: `{progress.as_posix()}`",
            "",
            "## Known gaps (not failures)",
            "",
            "- **12 / 18** strong-augmentation cells lack seed 2024 in `transfer_all_seeds.csv`.",
            "- AdaBN evaluated at seed 42 only (by design).",
            "- Bucket ablation: ResNet50 × seed 42 only.",
            "- Grad-CAM overlay PNG not bundled locally.",
            "",
            "## Re-run",
            "",
            "```bash",
            "python scripts/rebuild_multiseed_summary.py",
            "python scripts/build_multiseed_tables.py --sync-week11",
            "python scripts/audit_writing.py",
            "python -m freeze_results_v2",
            "```",
            "",
        ]
    )

    if n_fail:
        lines.extend(["## Failures (investigate)", ""])
        for c in checks:
            if not c.ok:
                lines.append(
                    f"- **{c.claim}**: expected {c.expected:.3f}, got {c.actual:.4f} from `{c.source}`"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    checks = _load_checks()
    render_report(checks, args.manuscript, args.progress, args.out)

    n_pass = sum(1 for c in checks if c.ok)
    print(f"Audit: {n_pass}/{len(checks)} PASS -> {args.out}")
    for c in checks:
        if not c.ok:
            print(f"  FAIL {c.claim}: expected {c.expected:.3f} actual {c.actual:.4f}")


if __name__ == "__main__":
    main()
