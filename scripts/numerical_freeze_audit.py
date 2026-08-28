"""Numerical freeze audit — checks that *can fail*.

Unlike ``freeze_results_v2`` (which verifies file-copy / SHA-256 integrity of
v1→v2 core copies), this script recomputes statistics from frozen CSVs and
multi-seed tables and compares them to published aggregates.

Exit code 0 = all checks PASS; 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes" / "numerical_freeze_audit.md"
TOL = 1e-9
DISPLAY_TOL = 5e-4


@dataclass
class Check:
    name: str
    expected: float | str
    actual: float | str
    ok: bool
    detail: str = ""


def _allclose(a: float, b: float, tol: float = TOL) -> bool:
    if isinstance(a, (float, int, np.floating)) and isinstance(b, (float, int, np.floating)):
        if np.isnan(a) and np.isnan(b):
            return True
        return abs(float(a) - float(b)) <= tol
    return a == b


def run_checks() -> list[Check]:
    checks: list[Check] = []
    frozen = ROOT / "frozen_results"
    multiseed = ROOT / "week11_results" / "multiseed"
    if not (multiseed / "transfer_all_seeds.csv").exists():
        multiseed = ROOT / "results" / "multiseed"
    stats = ROOT / "results" / "stats"

    # 1. Gap arithmetic on frozen baseline / aug
    for gap_name in ["generalization_gap.csv", "generalization_gap_aug.csv"]:
        gap = pd.read_csv(frozen / gap_name)
        recomputed = gap["in_dataset_macro_f1"] - gap["cross_macro_f1"]
        max_err = float((recomputed - gap["generalization_gap"]).abs().max())
        checks.append(
            Check(
                f"{gap_name} gap arithmetic",
                0.0,
                max_err,
                max_err <= TOL,
                f"max |recomputed − stored| = {max_err:.3e}",
            )
        )

    # 2. Seed-42 rows in transfer_all_seeds must equal frozen matrices
    transfer = pd.read_csv(multiseed / "transfer_all_seeds.csv")
    for aug, matrix_name, col in [
        ("default", "crossdataset_matrix.csv", "macro_f1"),
        ("strong", "crossdataset_matrix_aug.csv", "macro_f1"),
    ]:
        frozen_m = pd.read_csv(frozen / matrix_name)
        s42 = transfer[(transfer["seed"] == 42) & (transfer["augmentation"] == aug)]
        merged = s42.merge(
            frozen_m,
            on=["train_dataset", "test_dataset", "model", "seed"],
            suffixes=("_ms", "_fr"),
        )
        if len(merged) != 18:
            checks.append(
                Check(
                    f"seed-42 {aug} coverage vs frozen",
                    18,
                    len(merged),
                    False,
                    "row count after merge",
                )
            )
            continue
        max_err = float((merged["cross_macro_f1"] - merged[col]).abs().max())
        checks.append(
            Check(
                f"seed-42 {aug} cross_macro_f1 == frozen",
                0.0,
                max_err,
                max_err <= TOL,
                f"max abs diff = {max_err:.3e} (n={len(merged)})",
            )
        )

    # 3. Recompute cell means and compare to transfer_cell_mean_std.csv
    cell = pd.read_csv(multiseed / "transfer_cell_mean_std.csv")
    recomputed = (
        transfer.groupby(
            ["augmentation", "train_dataset", "test_dataset", "model"], as_index=False
        )
        .agg(
            cross_mean=("cross_macro_f1", "mean"),
            cross_std=("cross_macro_f1", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    # Drop bucket-* from comparison (ablation rows may appear in transfer)
    recomputed = recomputed[
        recomputed["augmentation"].isin(["default", "strong"])
    ]
    cell_cmp = cell[cell["augmentation"].isin(["default", "strong"])]
    merged_c = cell_cmp.merge(
        recomputed,
        on=["augmentation", "train_dataset", "test_dataset", "model"],
        suffixes=("_stored", "_re"),
    )
    max_mean = float(
        (merged_c["cross_mean_stored"] - merged_c["cross_mean_re"]).abs().max()
    )
    checks.append(
        Check(
            "transfer_cell_mean_std cross_mean recomputes",
            0.0,
            max_mean,
            max_mean <= TOL,
            f"max abs diff = {max_mean:.3e}",
        )
    )

    # 4. Wilcoxon on cell-mean paired deltas must match stats_tests.csv
    base = transfer[transfer["augmentation"] == "default"]
    strong = transfer[transfer["augmentation"] == "strong"]
    keys = ["train_dataset", "test_dataset", "model", "seed"]
    paired = base[keys + ["cross_macro_f1"]].rename(
        columns={"cross_macro_f1": "baseline"}
    ).merge(
        strong[keys + ["cross_macro_f1"]].rename(
            columns={"cross_macro_f1": "strong"}
        ),
        on=keys,
        how="inner",
    )
    paired["delta"] = paired["strong"] - paired["baseline"]
    cell_delta = paired.groupby(
        ["train_dataset", "test_dataset", "model"], as_index=False
    )["delta"].mean()
    result = wilcoxon(
        cell_delta["delta"].to_numpy(), zero_method="wilcox", alternative="two-sided"
    )
    mean_delta = float(cell_delta["delta"].mean())
    n_pos = int((cell_delta["delta"] > 0).sum())

    stats_path = stats / "stats_tests.csv"
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        row = stats_df[
            stats_df["claim"] == "augmentation_gt_baseline_mean_over_seeds_18_cells"
        ].iloc[0]
        checks.append(
            Check(
                "Wilcoxon mean_delta matches stats_tests",
                float(row["mean_delta"]),
                mean_delta,
                _allclose(float(row["mean_delta"]), mean_delta, DISPLAY_TOL),
                f"stored={row['mean_delta']:.6f} recomputed={mean_delta:.6f}",
            )
        )
        checks.append(
            Check(
                "Wilcoxon pvalue matches stats_tests",
                float(row["pvalue"]),
                float(result.pvalue),
                _allclose(float(row["pvalue"]), float(result.pvalue), 1e-6),
                f"stored={row['pvalue']:.6g} recomputed={result.pvalue:.6g}",
            )
        )
        checks.append(
            Check(
                "Wilcoxon n_positive matches stats_tests",
                int(row["n_positive"]),
                n_pos,
                int(row["n_positive"]) == n_pos,
                f"stored={int(row['n_positive'])} recomputed={n_pos}",
            )
        )
    else:
        checks.append(
            Check("stats_tests.csv present", "exists", "missing", False, str(stats_path))
        )

    # 5. Headline 3-seed means (Option A publication pair)
    base_mean = float(
        cell[cell["augmentation"] == "default"]["cross_mean"].mean()
    )
    strong_mean = float(
        cell[cell["augmentation"] == "strong"]["cross_mean"].mean()
    )
    checks.append(
        Check(
            "Option A baseline cross mean ≈ 0.445",
            0.445,
            base_mean,
            abs(base_mean - 0.445) < 0.001,
            f"actual={base_mean:.6f}",
        )
    )
    checks.append(
        Check(
            "Option A strong cross mean ≈ 0.502",
            0.502,
            strong_mean,
            abs(strong_mean - 0.502) < 0.001,
            f"actual={strong_mean:.6f}",
        )
    )

    # 6. Mitigation pairwise re-derives from frozen matrices
    pairwise = pd.read_csv(frozen / "mitigation_pairwise_aug.csv")
    base_m = pd.read_csv(frozen / "crossdataset_matrix.csv")
    aug_m = pd.read_csv(frozen / "crossdataset_matrix_aug.csv")
    gap_b = pd.read_csv(frozen / "generalization_gap.csv")
    gap_a = pd.read_csv(frozen / "generalization_gap_aug.csv")
    rebuild = base_m.merge(
        aug_m,
        on=["train_dataset", "test_dataset", "model", "seed"],
        suffixes=("_base", "_aug"),
    )
    rebuild = rebuild.merge(
        gap_b[
            ["train_dataset", "test_dataset", "model", "seed", "generalization_gap"]
        ].rename(columns={"generalization_gap": "baseline_gap"}),
        on=["train_dataset", "test_dataset", "model", "seed"],
    )
    rebuild = rebuild.merge(
        gap_a[
            ["train_dataset", "test_dataset", "model", "seed", "generalization_gap"]
        ].rename(columns={"generalization_gap": "aug_gap"}),
        on=["train_dataset", "test_dataset", "model", "seed"],
    )
    rebuild["cross_f1_improvement"] = rebuild["macro_f1_aug"] - rebuild["macro_f1_base"]
    joined = pairwise.merge(
        rebuild[
            [
                "train_dataset",
                "test_dataset",
                "model",
                "seed",
                "macro_f1_base",
                "macro_f1_aug",
                "cross_f1_improvement",
            ]
        ],
        on=["train_dataset", "test_dataset", "model", "seed"],
    )
    err = float(
        (
            joined["baseline_cross_macro_f1"] - joined["macro_f1_base"]
        ).abs().max()
    )
    checks.append(
        Check(
            "mitigation_pairwise baseline matches matrix",
            0.0,
            err,
            err <= TOL,
            f"max abs diff = {err:.3e}",
        )
    )
    err2 = float(
        (joined["aug_cross_macro_f1"] - joined["macro_f1_aug"]).abs().max()
    )
    checks.append(
        Check(
            "mitigation_pairwise aug matches matrix",
            0.0,
            err2,
            err2 <= TOL,
            f"max abs diff = {err2:.3e}",
        )
    )

    return checks


def render(checks: list[Check], out_path: Path) -> int:
    n_pass = sum(1 for c in checks if c.ok)
    n_fail = len(checks) - n_pass
    lines = [
        "# Numerical freeze audit",
        "",
        "These checks **recompute** statistics from frozen CSVs / multi-seed tables",
        "and compare to stored aggregates. Unlike `freeze_results_v2` (copy/hash",
        "integrity), a wrong number here will FAIL.",
        "",
        f"**Result: {n_pass}/{len(checks)} PASS**"
        + (f", **{n_fail} FAIL**" if n_fail else ""),
        "",
        "| Check | Expected | Actual | Status | Detail |",
        "|-------|----------|--------|--------|--------|",
    ]
    for c in checks:
        status = "PASS" if c.ok else "FAIL"
        exp = f"{c.expected:.6g}" if isinstance(c.expected, float) else str(c.expected)
        act = f"{c.actual:.6g}" if isinstance(c.actual, float) else str(c.actual)
        lines.append(
            f"| {c.name} | {exp} | {act} | {status} | {c.detail} |"
        )
    lines.extend(
        [
            "",
            "## Re-run",
            "",
            "```bash",
            "python scripts/rebuild_multiseed_summary.py",
            "python scripts/build_multiseed_tables.py --sync-week11",
            "python -m run_stats",
            "python scripts/numerical_freeze_audit.py",
            "```",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Numerical audit: {n_pass}/{len(checks)} PASS -> {out_path}")
    for c in checks:
        if not c.ok:
            print(f"  FAIL {c.name}: expected={c.expected} actual={c.actual} ({c.detail})")
    return 0 if n_fail == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    code = render(run_checks(), args.out)
    sys.exit(code)


if __name__ == "__main__":
    main()
