"""Grad-CAM negative-result statistics (Week 10, CPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
GRADCAM_CSV = ROOT / "frozen_results" / "gradcam_records.csv"
OUT_CSV = ROOT / "frozen_results_v2" / "gradcam_negative_summary.csv"

ENRICHMENT_COL = "border_attention_enrichment"


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def main() -> None:
    if not GRADCAM_CSV.is_file():
        raise FileNotFoundError(f"Missing Grad-CAM records: {GRADCAM_CSV}")

    df = pd.read_csv(GRADCAM_CSV)
    print(f"Loaded {GRADCAM_CSV}: shape={df.shape}")
    print(f"Columns: {list(df.columns)}")

    if ENRICHMENT_COL not in df.columns:
        raise KeyError(f"Expected column {ENRICHMENT_COL!r} in {GRADCAM_CSV}")

    enrichment = df[ENRICHMENT_COL].astype(float)
    correct_mask = df["correct"].astype(bool)
    incorrect_mask = ~correct_mask

    correct_vals = enrichment[correct_mask]
    incorrect_vals = enrichment[incorrect_mask]

    print("\n=== 1. correct vs incorrect ===")
    print(f"n_correct={len(correct_vals)} mean={correct_vals.mean():.6f} std={correct_vals.std(ddof=0):.6f}")
    print(
        f"n_incorrect={len(incorrect_vals)} mean={incorrect_vals.mean():.6f} "
        f"std={incorrect_vals.std(ddof=0):.6f}"
    )
    print("Note: n=6 per group — underpowered for Mann-Whitney.")

    u_stat, p_value = mannwhitneyu(
        correct_vals, incorrect_vals, alternative="two-sided"
    )
    print(f"Mann-Whitney U statistic: {u_stat}")
    print(f"Mann-Whitney p-value: {p_value}")

    print("\n=== 2. by true_label ===")
    for label, group in df.groupby("true_label"):
        vals = group[ENRICHMENT_COL].astype(float)
        print(
            f"  {label}: n={len(vals)} mean={vals.mean():.6f} std={vals.std(ddof=0):.6f}"
        )

    below_1 = enrichment < 1.0
    n_below = int(below_1.sum())
    frac_below = n_below / len(df)
    print("\n=== 3. enrichment < 1.0 ===")
    print(f"n={n_below} fraction={frac_below:.6f} percentage={frac_below * 100:.2f}%")

    print("\n=== 4. correct x (enrichment < 1.0) ===")
    ctab = pd.crosstab(
        correct_mask, below_1, rownames=["correct"], colnames=["enrichment_lt_1"]
    )
    print(ctab)

    rows: list[dict[str, str]] = [
        {"metric": "n_total", "value": str(len(df))},
        {"metric": "n_correct", "value": str(len(correct_vals))},
        {"metric": "mean_enrichment_correct", "value": _fmt(float(correct_vals.mean()))},
        {"metric": "std_enrichment_correct", "value": _fmt(float(correct_vals.std(ddof=0)))},
        {"metric": "n_incorrect", "value": str(len(incorrect_vals))},
        {"metric": "mean_enrichment_incorrect", "value": _fmt(float(incorrect_vals.mean()))},
        {"metric": "std_enrichment_incorrect", "value": _fmt(float(incorrect_vals.std(ddof=0)))},
        {"metric": "mannwhitney_u_statistic", "value": _fmt(float(u_stat))},
        {"metric": "mannwhitney_p_value", "value": _fmt(float(p_value))},
        {"metric": "n_enrichment_below_1p0", "value": str(n_below)},
        {"metric": "frac_enrichment_below_1p0", "value": _fmt(frac_below)},
    ]

    for label, group in df.groupby("true_label"):
        vals = group[ENRICHMENT_COL].astype(float)
        rows.append({"metric": f"n_label_{label}", "value": str(len(vals))})
        rows.append(
            {
                "metric": f"mean_enrichment_label_{label}",
                "value": _fmt(float(vals.mean())),
            }
        )

    for (is_correct, is_below), count in ctab.stack().items():
        rows.append(
            {
                "metric": f"crosstab_correct_{is_correct}_below1_{is_below}",
                "value": str(int(count)),
            }
        )

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"\nWrote {OUT_CSV}")
    print(f"Shape: {out.shape}")
    print(out.head(12))


if __name__ == "__main__":
    main()
