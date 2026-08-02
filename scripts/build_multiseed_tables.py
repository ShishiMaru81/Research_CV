"""Build camera-ready multi-seed paper tables (mean ± std) from multiseed aggregates."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MULTISEED = ROOT / "results" / "multiseed"
DEFAULT_STATS = ROOT / "results" / "stats"
DEFAULT_FROZEN = ROOT / "frozen_results"
OUT_TABLES = ROOT / "paper" / "tables"
OUT_NOTES = ROOT / "notes"

DATASET_LABELS = {
    "riceleafbd": "RiceLeafBD",
    "dhan_shomadhan": "Dhan-Shomadhan",
    "brri_rice_disease_pest": "BRRI",
}

MODEL_LABELS = {
    "mobilenetv2_100": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet50",
}


def _write_table(df: pd.DataFrame, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    tex_path = stem.with_suffix(".tex")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    tex_path.write_text(
        df.to_latex(index=False, float_format=lambda value: f"{value:.3f}"),
        encoding="utf-8",
    )
    return [csv_path, tex_path]


def _label_transfer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["train"] = out["train_dataset"].map(DATASET_LABELS)
    out["test"] = out["test_dataset"].map(DATASET_LABELS)
    out["model"] = out["model"].map(MODEL_LABELS)
    return out


def build_transfer_multiseed(cell_path: Path) -> pd.DataFrame:
    cell = pd.read_csv(cell_path)
    cell = _label_transfer(cell)
    cols = [
        "augmentation",
        "train",
        "test",
        "model",
        "n_seeds",
        "cross_mean",
        "cross_std",
        "gap_mean",
        "gap_std",
    ]
    return cell[cols].sort_values(["augmentation", "train", "test", "model"])


def build_lodo_multiseed(cell_path: Path) -> pd.DataFrame:
    cell = pd.read_csv(cell_path)
    cell["held_out"] = cell["held_out_dataset"].map(DATASET_LABELS)
    cell["model"] = cell["model"].map(MODEL_LABELS)
    return cell[
        ["held_out", "model", "n_seeds", "f1_mean", "f1_std"]
    ].sort_values(["held_out", "model"])


def build_summary_multiseed(
    cell_path: Path,
    stats_path: Path,
    frozen_pairwise: Path,
) -> pd.DataFrame:
    cell = pd.read_csv(cell_path)
    stats = pd.read_csv(stats_path)
    pairwise = pd.read_csv(frozen_pairwise)

    base = cell[cell["augmentation"] == "default"]
    strong = cell[cell["augmentation"] == "strong"]

    aug_row = stats[stats["claim"] == "augmentation_gt_baseline_mean_over_seeds_18_cells"]
    wilcoxon_p = float(aug_row["pvalue"].iloc[0]) if len(aug_row) else float("nan")
    mean_delta = float(aug_row["mean_delta"].iloc[0]) if len(aug_row) else float("nan")
    n_pos = int(aug_row["n_positive"].iloc[0]) if len(aug_row) else 0

    merged = base.merge(
        strong,
        on=["train_dataset", "test_dataset", "model"],
        suffixes=("_base", "_aug"),
    )
    merged["delta_mean"] = merged["cross_mean_aug"] - merged["cross_mean_base"]
    cells_positive = int((merged["delta_mean"] > 0).sum())

    rows = [
        ("baseline_cross_mean_3seed", base["cross_mean"].mean()),
        ("baseline_cross_mean_std_avg", base["cross_std"].mean()),
        ("strong_cross_mean_available_seeds", strong["cross_mean"].mean()),
        ("strong_cross_std_avg", strong["cross_std"].mean()),
        ("mean_paired_aug_delta_cell_means", mean_delta),
        ("cells_positive_aug_delta", float(cells_positive)),
        ("wilcoxon_p_18_cells", wilcoxon_p),
        ("seed42_baseline_cross_mean", pairwise["baseline_cross_macro_f1"].mean()),
        ("seed42_aug_cross_mean", pairwise["aug_cross_macro_f1"].mean()),
        ("seed42_mean_cross_improvement", pairwise["cross_f1_improvement"].mean()),
        ("strong_cells_missing_seed2024", float(18 - strong["n_seeds"].eq(3).sum())),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value"])


def sync_to_week11(multiseed_dir: Path) -> None:
    """Keep ``week11_results/multiseed/`` aligned with ``results/multiseed/``."""
    dest = ROOT / "week11_results" / "multiseed"
    dest.mkdir(parents=True, exist_ok=True)
    for path in multiseed_dir.glob("*.csv"):
        shutil.copy2(path, dest / path.name)
    summary = multiseed_dir / "PHASE1_MULTISEED_SUMMARY.md"
    if summary.exists():
        shutil.copy2(summary, dest / summary.name)


def build_tables(
    multiseed_dir: Path,
    stats_dir: Path,
    frozen_dir: Path,
    tables_dir: Path,
    notes_dir: Path,
) -> dict[str, Path]:
    cell_path = multiseed_dir / "transfer_cell_mean_std.csv"
    lodo_path = multiseed_dir / "lodo_cell_mean_std.csv"
    stats_path = stats_dir / "stats_tests.csv"
    pairwise_path = frozen_dir / "mitigation_pairwise_aug.csv"

    for path in (cell_path, lodo_path, stats_path, pairwise_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    transfer = build_transfer_multiseed(cell_path)
    lodo = build_lodo_multiseed(lodo_path)
    summary = build_summary_multiseed(cell_path, stats_path, pairwise_path)

    paths: dict[str, Path] = {}
    for name, frame in [
        ("table_transfer_multiseed", transfer),
        ("table_lodo_multiseed", lodo),
        ("table_summary_stats_multiseed", summary),
    ]:
        written = _write_table(frame, tables_dir / name)
        paths[name] = written[0]

    # Coverage note for missing seed-2024 strong cells
    cell = pd.read_csv(cell_path)
    strong = cell[cell["augmentation"] == "strong"]
    missing = strong[strong["n_seeds"] < 3][
        ["train_dataset", "test_dataset", "model", "n_seeds"]
    ]
    note_lines = [
        "# Multi-seed table build",
        "",
        f"Source: `{cell_path.as_posix()}`",
        "",
        "## Headline (from `table_summary_stats_multiseed.csv`)",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Strong-aug cells with fewer than 3 seeds",
        "",
    ]
    if len(missing):
        note_lines.append(
            "| train | test | model | n_seeds |\n"
            "|-------|------|-------|--------|\n"
            + "\n".join(
                f"| {r.train_dataset} | {r.test_dataset} | {r.model} | {int(r.n_seeds)} |"
                for r in missing.itertuples()
            )
        )
        note_lines.append("")
        note_lines.append(
            f"**{len(missing)} / 18** strong-aug cells lack seed 2024 locally."
        )
    else:
        note_lines.append("All 18 strong-aug cells have 3 seeds.")
    notes_dir.mkdir(parents=True, exist_ok=True)
    note_path = notes_dir / "multiseed_tables_build.md"
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    paths["multiseed_tables_build"] = note_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multiseed-dir",
        type=Path,
        default=DEFAULT_MULTISEED,
    )
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--tables-dir", type=Path, default=OUT_TABLES)
    parser.add_argument("--notes-dir", type=Path, default=OUT_NOTES)
    parser.add_argument(
        "--sync-week11",
        action="store_true",
        help="Copy multiseed CSVs to week11_results/multiseed/",
    )
    args = parser.parse_args()

    if args.sync_week11:
        sync_to_week11(args.multiseed_dir)

    paths = build_tables(
        args.multiseed_dir,
        args.stats_dir,
        args.frozen_dir,
        args.tables_dir,
        args.notes_dir,
    )
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
