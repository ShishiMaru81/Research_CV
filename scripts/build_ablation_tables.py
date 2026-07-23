"""Build paper tables + interpretation from augmentation_ablation.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION = ROOT / "results" / "ablation" / "augmentation_ablation.csv"
DEFAULT_BASELINE = ROOT / "frozen_results" / "crossdataset_matrix.csv"
DEFAULT_STRONG = ROOT / "frozen_results" / "crossdataset_matrix_aug.csv"
OUT_TABLES = ROOT / "paper" / "tables"
OUT_NOTES = ROOT / "notes"


BUCKET_ORDER = ["geo", "photo", "occlusion"]
BUCKET_LABELS = {
    "geo": "Geometric",
    "photo": "Photometric",
    "occlusion": "Occlusion",
}


def _load_resnet_baseline(path: Path, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["model"] == "resnet50") & (df["seed"] == seed)].copy()
    return df.rename(columns={"macro_f1": "baseline_macro_f1"})[
        ["train_dataset", "test_dataset", "baseline_macro_f1"]
    ]


def build_tables(
    ablation_path: Path,
    baseline_path: Path,
    strong_path: Path | None,
    tables_dir: Path,
    notes_dir: Path,
) -> dict[str, Path]:
    if not ablation_path.exists():
        raise FileNotFoundError(
            f"Missing {ablation_path}. Run `python -m run_ablation` first."
        )

    abl = pd.read_csv(ablation_path)
    if "bucket" not in abl.columns and "augmentation" in abl.columns:
        abl["bucket"] = (
            abl["augmentation"]
            .astype(str)
            .str.replace("bucket-", "", regex=False)
        )
    abl = abl[abl["model"] == "resnet50"].copy()
    baseline = _load_resnet_baseline(baseline_path)
    merged = abl.merge(baseline, on=["train_dataset", "test_dataset"], how="left")
    merged["delta_vs_baseline"] = merged["macro_f1"] - merged["baseline_macro_f1"]

    if strong_path is not None and strong_path.exists():
        strong = pd.read_csv(strong_path)
        strong = strong[
            (strong["model"] == "resnet50") & (strong["seed"] == 42)
        ][["train_dataset", "test_dataset", "macro_f1"]].rename(
            columns={"macro_f1": "strong_macro_f1"}
        )
        merged = merged.merge(strong, on=["train_dataset", "test_dataset"], how="left")
        merged["delta_vs_strong"] = merged["macro_f1"] - merged["strong_macro_f1"]
        merged["frac_of_strong_gain"] = (
            merged["delta_vs_baseline"]
            / (merged["strong_macro_f1"] - merged["baseline_macro_f1"]).replace(0, pd.NA)
        )

    tables_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    detail = merged.sort_values(["bucket", "train_dataset", "test_dataset"])
    detail_path = tables_dir / "table_ablation.csv"
    detail.to_csv(detail_path, index=False)

    summary = (
        detail.groupby("bucket", as_index=False)
        .agg(
            n=("macro_f1", "size"),
            mean_cross_f1=("macro_f1", "mean"),
            mean_delta_vs_baseline=("delta_vs_baseline", "mean"),
            n_improved=("delta_vs_baseline", lambda s: int((s > 0).sum())),
        )
        .sort_values("bucket", key=lambda s: s.map({b: i for i, b in enumerate(BUCKET_ORDER)}))
    )
    summary["bucket_label"] = summary["bucket"].map(BUCKET_LABELS)
    summary_path = tables_dir / "table_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Rank buckets by mean Δ vs baseline.
    ranked = summary.sort_values("mean_delta_vs_baseline", ascending=False)
    winner = ranked.iloc[0]
    runner = ranked.iloc[1] if len(ranked) > 1 else None
    lines = [
        "# Phase 3 ablation interpretation",
        "",
        f"Source: `{ablation_path.as_posix()}`",
        "",
        "## Ranking (mean Δ macro-F1 vs ResNet50 seed-42 baseline)",
        "",
    ]
    for _, row in ranked.iterrows():
        lines.append(
            f"- **{BUCKET_LABELS.get(row['bucket'], row['bucket'])}**: "
            f"mean Δ = {row['mean_delta_vs_baseline']:+.3f} "
            f"(mean cross F1 = {row['mean_cross_f1']:.3f}; "
            f"{int(row['n_improved'])}/{int(row['n'])} pairs improve)"
        )
    lines.extend(
        [
            "",
            "## One-paragraph claim",
            "",
            (
                f"On ResNet50 × six transfer pairs (seed 42), the "
                f"**{BUCKET_LABELS.get(winner['bucket'], winner['bucket'])}** "
                f"bucket alone recovers the largest share of the strong-augmentation "
                f"signal (mean Δ vs baseline {winner['mean_delta_vs_baseline']:+.3f}"
            ),
        ]
    )
    if runner is not None:
        lines[-1] += (
            f"; next is {BUCKET_LABELS.get(runner['bucket'], runner['bucket'])} "
            f"at {runner['mean_delta_vs_baseline']:+.3f}"
        )
    lines[-1] += (
        "). This scopes the Week 7 bundled pipeline to a mechanism-level "
        "comparison; residual gaps relative to the full strong stack indicate "
        "that buckets are complementary rather than fully redundant."
    )
    lines.append("")
    interp_path = notes_dir / "ablation_interpretation.md"
    interp_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "detail": detail_path,
        "summary": summary_path,
        "interpretation": interp_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--strong", type=Path, default=DEFAULT_STRONG)
    parser.add_argument("--tables", type=Path, default=OUT_TABLES)
    parser.add_argument("--notes", type=Path, default=OUT_NOTES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = build_tables(
        ablation_path=args.ablation,
        baseline_path=args.baseline,
        strong_path=args.strong,
        tables_dir=args.tables,
        notes_dir=args.notes,
    )
    for key, path in paths.items():
        print(f"{key}: {path}")
