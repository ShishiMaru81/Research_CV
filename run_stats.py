"""Phase 4 statistics layer: Wilcoxon, seed variance, bootstrap CIs.

Reads reconstructed multi-seed transfer tables (and optional per-sample
prediction CSVs) and writes:

  results/stats/stats_tests.csv
  results/stats/seed_variance.csv
  results/stats/bootstrap_ci.csv
  results/stats/STATS_SUMMARY.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
DEFAULT_TRANSFER = ROOT / "week11_results" / "multiseed" / "transfer_all_seeds.csv"
DEFAULT_LODO = ROOT / "week11_results" / "multiseed" / "lodo_all_seeds.csv"
DEFAULT_ADABN = ROOT / "adabn_results.csv"
DEFAULT_PRED_ROOTS = [
    ROOT / "results" / "predictions",
    ROOT / "week11_light" / "predictions",
    ROOT / "week12_results" / "predictions",
]
OUT_DIR = ROOT / "results" / "stats"


def _wilcoxon_paired(deltas: np.ndarray, claim: str) -> dict[str, object]:
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    n = int(deltas.size)
    n_pos = int((deltas > 0).sum())
    mean_delta = float(np.mean(deltas)) if n else float("nan")
    if n < 2 or np.allclose(deltas, 0.0):
        return {
            "claim": claim,
            "test": "wilcoxon_signed_rank",
            "n": n,
            "n_positive": n_pos,
            "mean_delta": mean_delta,
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "note": "insufficient non-zero pairs",
        }
    # zero_method='wilcox' drops exact zeros (standard for signed-rank).
    result = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
    return {
        "claim": claim,
        "test": "wilcoxon_signed_rank",
        "n": n,
        "n_positive": n_pos,
        "mean_delta": mean_delta,
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "note": "",
    }


def seed_variance_table(transfer: pd.DataFrame) -> pd.DataFrame:
    keys = ["train_dataset", "test_dataset", "model", "augmentation"]
    grouped = (
        transfer.groupby(keys, as_index=False)
        .agg(
            n_seeds=("cross_macro_f1", "count"),
            mean_cross_f1=("cross_macro_f1", "mean"),
            std_cross_f1=("cross_macro_f1", "std"),
            mean_gap=("gap", "mean"),
            std_gap=("gap", "std"),
        )
    )
    return grouped.sort_values(keys)


def paired_aug_deltas(transfer: pd.DataFrame) -> pd.DataFrame:
    keys = ["train_dataset", "test_dataset", "model", "seed"]
    base = transfer[transfer["augmentation"] == "default"][keys + ["cross_macro_f1"]]
    strong = transfer[transfer["augmentation"] == "strong"][keys + ["cross_macro_f1"]]
    base = base.rename(columns={"cross_macro_f1": "baseline"})
    strong = strong.rename(columns={"cross_macro_f1": "strong"})
    paired = base.merge(strong, on=keys, how="inner")
    paired["delta"] = paired["strong"] - paired["baseline"]
    return paired


def bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    point = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = f1_score(
            y_true[idx], y_pred[idx], average="macro", zero_division=0
        )
    low, high = np.quantile(stats, [0.025, 0.975])
    return {"point": point, "ci_low": float(low), "ci_high": float(high)}


def discover_prediction_csvs(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.exists():
            found.extend(sorted(root.glob("*.csv")))
    return found


def bootstrap_from_predictions(
    pred_paths: list[Path],
    *,
    n_boot: int = 1000,
    limit: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paths = pred_paths[:limit] if limit is not None else pred_paths
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 — skip unreadable artifacts
            rows.append(
                {
                    "predictions_path": str(path),
                    "n_samples": 0,
                    "point_macro_f1": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "note": f"read_error:{exc}",
                }
            )
            continue
        if "true_index" not in df.columns or "pred_index" not in df.columns:
            continue
        y_true = df["true_index"].to_numpy()
        y_pred = df["pred_index"].to_numpy()
        ci = bootstrap_macro_f1(y_true, y_pred, n_boot=n_boot)
        rows.append(
            {
                "predictions_path": str(path).replace("\\", "/"),
                "n_samples": int(len(df)),
                "point_macro_f1": ci["point"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def run_stats(
    transfer_path: Path = DEFAULT_TRANSFER,
    lodo_path: Path = DEFAULT_LODO,
    adabn_path: Path = DEFAULT_ADABN,
    pred_roots: list[Path] | None = None,
    output_dir: Path = OUT_DIR,
    n_boot: int = 1000,
    bootstrap_limit: int | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    transfer = pd.read_csv(transfer_path)

    variance = seed_variance_table(transfer)
    variance_path = output_dir / "seed_variance.csv"
    variance.to_csv(variance_path, index=False)

    paired = paired_aug_deltas(transfer)
    # Prefer seed-mean deltas per cell when multiple seeds exist; also test
    # all available paired seed rows and seed-42-only for transparency.
    cell_mean = (
        paired.groupby(["train_dataset", "test_dataset", "model"], as_index=False)[
            "delta"
        ]
        .mean()
    )
    tests = [
        _wilcoxon_paired(
            cell_mean["delta"].to_numpy(),
            "augmentation_gt_baseline_mean_over_seeds_18_cells",
        ),
        _wilcoxon_paired(
            paired["delta"].to_numpy(),
            "augmentation_gt_baseline_all_seed_pair_rows",
        ),
        _wilcoxon_paired(
            paired.loc[paired["seed"] == 42, "delta"].to_numpy(),
            "augmentation_gt_baseline_seed42_18_pairs",
        ),
    ]

    if lodo_path.exists():
        lodo = pd.read_csv(lodo_path)
        # LODO vs single-source is strategy-level; if a matched delta column
        # exists use it, else skip formal test and only summarize.
        if {"lodo_macro_f1", "single_source_macro_f1"}.issubset(lodo.columns):
            deltas = (
                lodo["lodo_macro_f1"] - lodo["single_source_macro_f1"]
            ).to_numpy()
            tests.append(
                _wilcoxon_paired(deltas, "lodo_gt_single_source_matched_rows")
            )

    if adabn_path.exists():
        adabn = pd.read_csv(adabn_path)
        if "delta_macro_f1" in adabn.columns:
            tests.append(
                _wilcoxon_paired(
                    adabn["delta_macro_f1"].to_numpy(),
                    "adabn_gt_baseline_18_pairs_seed42",
                )
            )

    tests_df = pd.DataFrame(tests)
    tests_path = output_dir / "stats_tests.csv"
    tests_df.to_csv(tests_path, index=False)

    pred_roots = pred_roots or DEFAULT_PRED_ROOTS
    pred_paths = discover_prediction_csvs(pred_roots)
    bootstrap_df = bootstrap_from_predictions(
        pred_paths, n_boot=n_boot, limit=bootstrap_limit
    )
    bootstrap_path = output_dir / "bootstrap_ci.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)

    # Headline noise-floor comparison (workflow §4.1).
    base_std = variance[variance["augmentation"] == "default"]["std_cross_f1"]
    mean_seed_std = float(base_std.mean()) if len(base_std) else float("nan")
    mean_aug_delta = float(cell_mean["delta"].mean()) if len(cell_mean) else float("nan")
    ratio = (
        mean_aug_delta / mean_seed_std
        if mean_seed_std and np.isfinite(mean_seed_std) and mean_seed_std > 0
        else float("nan")
    )
    aug_test = tests_df[
        tests_df["claim"] == "augmentation_gt_baseline_mean_over_seeds_18_cells"
    ].iloc[0]

    summary = f"""# Phase 4 statistics summary

Generated from `{transfer_path.as_posix()}`.

## Headline noise floor

| Quantity | Value |
|----------|------:|
| Mean paired aug delta (cell means over seeds) | {mean_aug_delta:+.3f} |
| Mean across-seed std of baseline cross macro-F1 | +/-{mean_seed_std:.3f} |
| Ratio (aug delta / seed std) | {ratio:.2f} |

Interpretation guide: if |aug delta| is **comparable to or smaller than** the
across-seed std, soften single-seed augmentation anecdotes in the paper.

## Wilcoxon signed-rank (augmentation)

- Claim: `{aug_test['claim']}`
- n = {int(aug_test['n'])}, positive = {int(aug_test['n_positive'])}
- mean delta = {float(aug_test['mean_delta']):+.4f}
- W = {aug_test['statistic']}, p = {aug_test['pvalue']}
- Note: {aug_test['note'] or 'ok'}

## Artifacts

- `{variance_path.as_posix()}`
- `{tests_path.as_posix()}`
- `{bootstrap_path.as_posix()}` ({len(bootstrap_df)} prediction files bootstrapped)

Seed-2024 strong-aug coverage may still be incomplete in the reconstructed
transfer table; re-run after filling missing cells and refresh this summary.
"""
    summary_path = output_dir / "STATS_SUMMARY.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, default=DEFAULT_TRANSFER)
    parser.add_argument("--lodo", type=Path, default=DEFAULT_LODO)
    parser.add_argument("--adabn", type=Path, default=DEFAULT_ADABN)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument(
        "--bootstrap_limit",
        type=int,
        default=None,
        help="Optional cap on prediction CSVs for a quick smoke run.",
    )
    parser.add_argument(
        "--pred_roots",
        nargs="*",
        default=None,
        help="Directories containing per-sample prediction CSVs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    roots = [Path(p) for p in args.pred_roots] if args.pred_roots else None
    run_stats(
        transfer_path=args.transfer,
        lodo_path=args.lodo,
        adabn_path=args.adabn,
        pred_roots=roots,
        output_dir=args.output,
        n_boot=args.n_boot,
        bootstrap_limit=args.bootstrap_limit,
    )
