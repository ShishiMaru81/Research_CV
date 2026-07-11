from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import load_config

PAIR_KEYS = ["train_dataset", "test_dataset", "model", "classes", "seed"]


def _load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    print(f"WARNING: missing {path}; related columns will be blank.")
    return pd.DataFrame()


def build_pairwise_augmentation(
    baseline_matrix: pd.DataFrame,
    baseline_gap: pd.DataFrame,
    aug_matrix: pd.DataFrame,
    aug_gap: pd.DataFrame,
) -> pd.DataFrame:
    """Per transfer pair: baseline vs strong-augmentation, same keys."""
    if baseline_matrix.empty or aug_matrix.empty:
        return pd.DataFrame()

    base = baseline_matrix[PAIR_KEYS + ["macro_f1"]].rename(
        columns={"macro_f1": "baseline_cross_macro_f1"}
    )
    aug = aug_matrix[PAIR_KEYS + ["macro_f1"]].rename(
        columns={"macro_f1": "aug_cross_macro_f1"}
    )
    merged = base.merge(aug, on=PAIR_KEYS, how="outer")

    if not baseline_gap.empty:
        merged = merged.merge(
            baseline_gap[PAIR_KEYS + ["generalization_gap"]].rename(
                columns={"generalization_gap": "baseline_gap"}
            ),
            on=PAIR_KEYS,
            how="left",
        )
    if not aug_gap.empty:
        merged = merged.merge(
            aug_gap[PAIR_KEYS + ["generalization_gap"]].rename(
                columns={"generalization_gap": "aug_gap"}
            ),
            on=PAIR_KEYS,
            how="left",
        )

    merged["cross_f1_improvement"] = (
        merged["aug_cross_macro_f1"] - merged["baseline_cross_macro_f1"]
    )
    if "baseline_gap" in merged and "aug_gap" in merged:
        merged["gap_reduction"] = merged["baseline_gap"] - merged["aug_gap"]
    return merged.sort_values(PAIR_KEYS).reset_index(drop=True)


def build_comparison_by_target(
    baseline_matrix: pd.DataFrame,
    aug_matrix: pd.DataFrame,
    lodo: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side per (test dataset, model): single-source vs mitigations.

    Single-source columns average all Week 5 transfer pairs that target a given
    test dataset. LODO trains on the other two datasets combined and evaluates
    on that same held-out target. Class sets differ between strategies, so this
    is a strategy-level comparison, not a matched-class comparison.
    """
    frames: list[pd.DataFrame] = []

    if not baseline_matrix.empty:
        base = (
            baseline_matrix.groupby(["test_dataset", "model"], as_index=False)
            .agg(baseline_single_source_macro_f1=("macro_f1", "mean"))
            .rename(columns={"test_dataset": "target_dataset"})
        )
        frames.append(base)

    if not aug_matrix.empty:
        aug = (
            aug_matrix.groupby(["test_dataset", "model"], as_index=False)
            .agg(aug_single_source_macro_f1=("macro_f1", "mean"))
            .rename(columns={"test_dataset": "target_dataset"})
        )
        frames.append(aug)

    if not lodo.empty:
        lodo_summary = lodo.rename(
            columns={"held_out_dataset": "target_dataset", "macro_f1": "lodo_macro_f1"}
        )[["target_dataset", "model", "lodo_macro_f1"]]
        frames.append(lodo_summary)

    if not frames:
        return pd.DataFrame()

    comparison = frames[0]
    for frame in frames[1:]:
        comparison = comparison.merge(
            frame, on=["target_dataset", "model"], how="outer"
        )

    if {"baseline_single_source_macro_f1", "lodo_macro_f1"}.issubset(
        comparison.columns
    ):
        comparison["lodo_vs_single_source"] = (
            comparison["lodo_macro_f1"]
            - comparison["baseline_single_source_macro_f1"]
        )
    if {
        "baseline_single_source_macro_f1",
        "aug_single_source_macro_f1",
    }.issubset(comparison.columns):
        comparison["aug_vs_single_source"] = (
            comparison["aug_single_source_macro_f1"]
            - comparison["baseline_single_source_macro_f1"]
        )
    return comparison.sort_values(["target_dataset", "model"]).reset_index(
        drop=True
    )


def run_mitigation(config_path: str = "config.yaml") -> tuple[Path, Path]:
    config = load_config(config_path)
    results_root = Path(config["results_root"])

    baseline_matrix = _load_csv(results_root / "crossdataset_matrix.csv")
    baseline_gap = _load_csv(results_root / "generalization_gap.csv")
    aug_matrix = _load_csv(results_root / "crossdataset_matrix_aug.csv")
    aug_gap = _load_csv(results_root / "generalization_gap_aug.csv")
    lodo = _load_csv(results_root / "lodo_results.csv")

    pairwise = build_pairwise_augmentation(
        baseline_matrix, baseline_gap, aug_matrix, aug_gap
    )
    comparison = build_comparison_by_target(baseline_matrix, aug_matrix, lodo)

    pairwise_path = results_root / "mitigation_pairwise_aug.csv"
    comparison_path = results_root / "mitigation_comparison.csv"

    if not pairwise.empty:
        pairwise.to_csv(pairwise_path, index=False)
        print("\n=== Per-pair augmentation effect ===")
        print(pairwise.to_string(index=False))
        print(f"\nWrote: {pairwise_path}")
        mean_improvement = pairwise["cross_f1_improvement"].mean()
        print(f"Mean cross macro-F1 improvement (aug): {mean_improvement:.4f}")
        if "gap_reduction" in pairwise:
            print(
                f"Mean generalization-gap reduction (aug): "
                f"{pairwise['gap_reduction'].mean():.4f}"
            )

    if not comparison.empty:
        comparison.to_csv(comparison_path, index=False)
        print("\n=== Mitigation comparison by target dataset ===")
        print(comparison.to_string(index=False))
        print(f"\nWrote: {comparison_path}")

    return pairwise_path, comparison_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 7 mitigation comparison aggregator."
    )
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_mitigation(config_path=args.config)
