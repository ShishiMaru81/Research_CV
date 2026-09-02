"""Generate deployment table for manuscript (Week 10, CPU)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROFILE_CSV = ROOT / "frozen_results_v2" / "deployment_profile.csv"
OUT_DIR = ROOT / "paper" / "tables_v2"
OUT_CSV = OUT_DIR / "table_deployment.csv"
OUT_TEX = OUT_DIR / "table_deployment.tex"

DISPLAY_NAMES = {
    "mobilenetv2_100": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet50",
}


def _format_latency(mean: float, std: float) -> str:
    return f"{mean:.1f} $\\pm$ {std:.1f}"


def main() -> None:
    if not PROFILE_CSV.is_file():
        raise FileNotFoundError(
            f"Missing {PROFILE_CSV}. Run scripts/profile_deployment.py first."
        )

    df = pd.read_csv(PROFILE_CSV)
    print(f"Loaded {PROFILE_CSV}: shape={df.shape}")
    print(df.head())

    table = pd.DataFrame(
        {
            "Model": df["model"].map(DISPLAY_NAMES),
            "Parameters (M)": (df["n_params"] / 1e6).round(1),
            "GFLOPs": df["gflops"].round(1),
            "CPU Latency (ms)": [
                f"{m:.1f} +/- {s:.1f}"
                for m, s in zip(
                    df["cpu_latency_ms_mean"], df["cpu_latency_ms_std"], strict=True
                )
            ],
            "Size (MB)": df["model_size_mb"].round(1),
        }
    )
    table = table.sort_values("GFLOPs", ascending=True).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)

    tex_rows: list[str] = []
    for _, row in table.iterrows():
        lat_tex = _format_latency(
            float(row["CPU Latency (ms)"].split()[0]),
            float(row["CPU Latency (ms)"].split()[2]),
        )
        tex_rows.append(
            f"{row['Model']} & {row['Parameters (M)']:.1f} & {row['GFLOPs']:.1f} "
            f"& {lat_tex} & {row['Size (MB)']:.1f} \\\\"
        )

    tex_body = "\n".join(tex_rows)
    tex = f"""\\begin{{table}}[h]
\\centering
\\caption{{Deployment characteristics of CNN backbones (224x224, batch 1).}}
\\label{{tab:deployment}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Model & Parameters (M) & GFLOPs & CPU Latency (ms) & Size (MB) \\\\
\\midrule
{tex_body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    OUT_TEX.write_text(tex, encoding="utf-8")

    print(f"\nWrote {OUT_CSV} ({len(table)} rows)")
    print(f"Wrote {OUT_TEX}")
    print("\n--- table_deployment.csv ---")
    print(table.to_string(index=False))
    print("\n--- table_deployment.tex ---")
    print(tex)

    if len(table) != len(df):
        raise RuntimeError(
            f"Row count mismatch: table={len(table)} profile={len(df)}"
        )


if __name__ == "__main__":
    main()
