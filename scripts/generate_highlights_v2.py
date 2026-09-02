"""Build highlights_v2.md from frozen authoritative CSVs (Week 10, CPU)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "paper" / "highlights_v2.md"

INDATASET = ROOT / "frozen_results_v2" / "indataset_results.csv"
GAP = ROOT / "frozen_results_v2" / "generalization_gap.csv"
CONFOUND = ROOT / "frozen_results_v2" / "background_confound.csv"
STATS = ROOT / "frozen_results_v2" / "stats_tests.csv"
MITIGATION = ROOT / "frozen_results_v2" / "mitigation_comparison.csv"
ABLATION = ROOT / "frozen_results_v2" / "augmentation_ablation.csv"
TRANSFER_MS = ROOT / "paper" / "tables" / "table_transfer_multiseed.csv"


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    indataset = _load(INDATASET)
    gap = _load(GAP)
    confound = _load(CONFOUND)
    stats = _load(STATS)
    mitigation = _load(MITIGATION)
    transfer_ms = _load(TRANSFER_MS)

    in_mean = indataset["macro_f1"].mean()
    cross_mean_3seed = transfer_ms.loc[
        transfer_ms["augmentation"] == "default", "cross_mean"
    ].mean()
    gap_mean = gap["generalization_gap"].mean()

    aug_row = stats.loc[
        stats["claim"] == "augmentation_gt_baseline_mean_over_seeds_18_cells"
    ].iloc[0]
    aug_delta = float(aug_row["mean_delta"])
    aug_p = float(aug_row["pvalue"])
    aug_pos = int(aug_row["n_positive"])

    adabn_row = stats.loc[
        stats["claim"] == "adabn_gt_baseline_18_pairs_seed42"
    ].iloc[0]
    adabn_delta = float(adabn_row["mean_delta"])

    white = confound.loc[confound["condition"] == "dhan_white", "macro_f1"].iloc[0]
    field = confound.loc[confound["condition"] == "dhan_field", "macro_f1"].iloc[0]
    cross_c = confound.loc[confound["condition"] == "riceleafbd_field", "macro_f1"].iloc[0]

    lodo_wins = int((mitigation["lodo_vs_single_source"] > 0).sum())
    lodo_total = len(mitigation)

    ablation_summary = pd.read_csv(ROOT / "paper" / "tables" / "table_ablation_summary.csv")
    geo_mean = float(
        ablation_summary.loc[ablation_summary["bucket"] == "geo", "mean_cross_f1"].iloc[0]
    )
    geo_delta = float(
        ablation_summary.loc[
            ablation_summary["bucket"] == "geo", "mean_delta_vs_baseline"
        ].iloc[0]
    )
    baseline_resnet_six = geo_mean - geo_delta

    base_mat = _load(ROOT / "frozen_results_v2" / "crossdataset_matrix.csv")
    aug_mat = _load(ROOT / "frozen_results_v2" / "crossdataset_matrix_aug.csv")
    resnet_base = base_mat.loc[base_mat["model"] == "resnet50", "macro_f1"]
    resnet_strong = aug_mat.loc[aug_mat["model"] == "resnet50", "macro_f1"]
    strong_mean = float(resnet_strong.mean())
    strong_delta = strong_mean - float(resnet_base.mean())
    geo_share = (geo_mean - baseline_resnet_six) / strong_delta

    bullets = [
        (
            f"Cross-dataset macro-F1 {cross_mean_3seed:.2f} versus in-dataset "
            f"{in_mean:.2f} on Bangladeshi rice collections."
        ),
        (
            f"White-background F1 {white:.2f} exceeds field {field:.2f} and "
            f"cross-field {cross_c:.2f} on fixed Dhan model."
        ),
        (
            f"Strong augmentation adds {aug_delta:+.2f} cross-F1 ({aug_pos}/18; "
            f"Wilcoxon p~{aug_p:.3g}); LODO wins {lodo_wins}/{lodo_total}."
        ),
        (
            f"AdaBN mean delta {adabn_delta:+.2f}; ResNet50 never improves "
            f"under BatchNorm recalibration alone."
        ),
        (
            f"Geometric transforms recover {geo_share:.0%} of ResNet50 "
            f"strong-aug cross-F1 gain on six pairs."
        ),
    ]

    for i, b in enumerate(bullets, start=1):
        n_words = len(b.replace("−", " ").split())
        if n_words > 20:
            print(f"WARNING bullet {i} has {n_words} words (target ≤15): {b}")

    content = "# Highlights (v2)\n\n" + "\n".join(f"- {b}" for b in bullets) + "\n"
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(content, encoding="utf-8")

    print(f"Wrote {OUT_MD}")
    print(content)
    print("\nSource means:")
    print(f"  in_mean={in_mean:.4f} cross_3seed={cross_mean_3seed:.4f} gap={gap_mean:.4f}")
    print(f"  confound white/field/cross={white:.3f}/{field:.3f}/{cross_c:.3f}")
    print(f"  aug_delta={aug_delta:.4f} p={aug_p:.6f} geo_share={geo_share:.3f}")


if __name__ == "__main__":
    main()
