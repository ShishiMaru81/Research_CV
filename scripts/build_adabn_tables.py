"""Build AdaBN publication tables from adabn_results.csv."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "adabn_results.csv")

pub = df[
    [
        "train_dataset",
        "test_dataset",
        "model",
        "classes",
        "seed",
        "baseline_macro_f1",
        "adabn_macro_f1",
        "delta_macro_f1",
        "n_samples",
        "n_adapt_images",
    ]
].copy()
for c in ["baseline_macro_f1", "adabn_macro_f1", "delta_macro_f1"]:
    pub[c] = pub[c].round(3)
pub = pub.sort_values(["train_dataset", "test_dataset", "model"])

tables = ROOT / "paper" / "tables"
tables.mkdir(parents=True, exist_ok=True)
pub.to_csv(tables / "table_adabn.csv", index=False)

lines = [
    r"\begin{tabular}{lllrrr}",
    r"\toprule",
    r"train & test & model & baseline F1 & AdaBN F1 & $\Delta$ F1 \\",
    r"\midrule",
]
for _, r in pub.iterrows():
    tr = str(r["train_dataset"]).replace("_", r"\_")
    te = str(r["test_dataset"]).replace("_", r"\_")
    mo = str(r["model"]).replace("_", r"\_")
    lines.append(
        f"{tr} & {te} & {mo} & {r['baseline_macro_f1']:.3f} & "
        f"{r['adabn_macro_f1']:.3f} & {r['delta_macro_f1']:+.3f} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
(tables / "table_adabn.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

sm = (
    df.groupby("model")
    .agg(
        mean_baseline=("baseline_macro_f1", "mean"),
        mean_adabn=("adabn_macro_f1", "mean"),
        mean_delta=("delta_macro_f1", "mean"),
        std_delta=("delta_macro_f1", "std"),
        n_improve=("delta_macro_f1", lambda s: int((s > 0).sum())),
        n=("delta_macro_f1", "size"),
    )
    .round(3)
    .reset_index()
)
sm.to_csv(tables / "table_adabn_by_model.csv", index=False)

w = ROOT / "week12_results" / "adabn"
w.mkdir(parents=True, exist_ok=True)
df.to_csv(w / "adabn_results.csv", index=False)
pub.to_csv(w / "table_adabn.csv", index=False)
sm.to_csv(w / "table_adabn_by_model.csv", index=False)

print(sm.to_string(index=False))
print("overall mean delta", round(float(df["delta_macro_f1"].mean()), 3))
print("improve", int((df["delta_macro_f1"] > 0).sum()), "/", len(df))
