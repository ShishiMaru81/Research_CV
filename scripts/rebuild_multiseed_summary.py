"""Rebuild multi-seed tables from transcript logs + frozen seed-42, then summarize."""
from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT = Path(
    r"C:\Users\paula\.cursor\projects\d-Research-Own\agent-transcripts"
    r"\27d0d383-1806-41f5-bd8a-dfe51c688375\27d0d383-1806-41f5-bd8a-dfe51c688375.jsonl"
)
OUT = ROOT / "results" / "multiseed"


def user_texts() -> list[str]:
    texts: list[str] = []
    with TRANSCRIPT.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("role") != "user":
                continue
            content = obj.get("message", {}).get("content", [])
            text = "\n".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
            if "RECORDED:" in text or "Cross-dataset results" in text:
                texts.append(text)
    return texts


def parse_recorded_with_context(text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    transfer_pat = re.compile(
        r"RECORDED:\s+(\S+)\s+\|\s+([^|]+)\s+\|\s+"
        r"same=([\d.]+)\s+\|\s+cross=([\d.]+)\s+\|\s+gap=([\d.]+)"
    )
    lodo_pat = re.compile(
        r"RECORDED:\s+(\S+)\s+\|\s+holdout=(\S+)\s+\|\s+macro_f1=([\d.]+)\s+\|\s+n=(\d+)"
    )
    train_xfer = re.compile(r"=== TRAIN (\S+): (\S+) -> (\S+) \(seed=(\d+)\) ===")
    train_lodo = re.compile(
        r"=== TRAIN (\S+): .+ \(holdout (\S+), seed=(\d+)\) ==="
    )
    ckpt_aug = re.compile(r"Best checkpoint:.*?__(aug-strong|seed\d+)")

    rows_x, rows_l = [], []
    current_aug = "default"
    pending = None
    last_ckpt_aug = None

    for line in text.splitlines():
        if "--- transfer_baseline ---" in line:
            current_aug = "default"
        elif "--- transfer_aug" in line:
            current_aug = "strong"
        if "aug-strong" in line and "Best checkpoint:" in line:
            last_ckpt_aug = "strong"
        elif "Best checkpoint:" in line and "aug-strong" not in line:
            # LODO / baseline checkpoints omit aug-strong
            if "lodo-holdout" not in line:
                last_ckpt_aug = "default"

        m = train_xfer.search(line)
        if m:
            pending = (
                "xfer",
                m.group(1),
                m.group(2),
                m.group(3),
                int(m.group(4)),
                current_aug,
            )
            continue
        m = train_lodo.search(line)
        if m:
            pending = ("lodo", m.group(1), m.group(2), int(m.group(3)))
            continue

        m = transfer_pat.search(line)
        if m:
            model = m.group(1)
            pair = m.group(2).strip()
            same, cross, gap = map(float, m.group(3, 4, 5))
            tr, ev = pair.split(":", 1) if ":" in pair else (None, None)
            seed, aug = None, current_aug
            if pending and pending[0] == "xfer":
                model, tr, ev, seed, aug = pending[1], pending[2], pending[3], pending[4], pending[5]
            # Prefer checkpoint cue when section header was wrong/missing
            if last_ckpt_aug is not None and pending and pending[0] == "xfer":
                # only override if TRAIN just happened for this run
                if pending[5] == current_aug:
                    aug = last_ckpt_aug if last_ckpt_aug == "strong" else aug
                    # If current section is strong, trust section
                    if current_aug == "strong":
                        aug = "strong"
            rows_x.append(
                dict(
                    model=model,
                    train_dataset=tr,
                    test_dataset=ev,
                    seed=seed,
                    augmentation=aug,
                    in_macro_f1=same,
                    cross_macro_f1=cross,
                    gap=gap,
                )
            )
            continue

        m = lodo_pat.search(line)
        if m:
            model, holdout = m.group(1), m.group(2)
            f1, n = float(m.group(3)), int(m.group(4))
            seed = None
            if pending and pending[0] == "lodo":
                model, holdout, seed = pending[1], pending[2], pending[3]
            rows_l.append(
                dict(
                    model=model,
                    held_out_dataset=holdout,
                    seed=seed,
                    macro_f1=f1,
                    n_samples=n,
                )
            )

    xf = pd.DataFrame(rows_x)
    ld = pd.DataFrame(rows_l)
    return xf, ld


def parse_gap_tables(text: str) -> pd.DataFrame:
    """Parse '=== Generalization gaps ===' whitespace tables."""
    blocks = re.split(r"=== Generalization gaps ===\s*", text)
    frames = []
    for block in blocks[1:]:
        # take until next === or Mean
        end = re.search(r"\n=== |\n=== Mean", block)
        chunk = block[: end.start()] if end else block[:4000]
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        # header is first line
        try:
            df = pd.read_csv(StringIO("\n".join(lines)), sep=r"\s+")
        except Exception:
            continue
        needed = {
            "train_dataset",
            "test_dataset",
            "model",
            "seed",
            "in_dataset_macro_f1",
            "cross_macro_f1",
            "generalization_gap",
        }
        if not needed.issubset(df.columns):
            continue
        if "augmentation" not in df.columns:
            # infer from checkpoint path if present
            if "checkpoint_path" in df.columns:
                df["augmentation"] = df["checkpoint_path"].map(
                    lambda p: "strong" if isinstance(p, str) and "aug-strong" in p else "default"
                )
            else:
                df["augmentation"] = "default"
        frames.append(
            df[
                [
                    "train_dataset",
                    "test_dataset",
                    "model",
                    "seed",
                    "augmentation",
                    "in_dataset_macro_f1",
                    "cross_macro_f1",
                    "generalization_gap",
                ]
            ].rename(
                columns={
                    "in_dataset_macro_f1": "in_macro_f1",
                    "generalization_gap": "gap",
                }
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_lodo_from_recorded(text: str) -> pd.DataFrame:
    """LODO whitespace tables are unusable (JSON per_class_f1). Use RECORDED only."""
    train_lodo = re.compile(
        r"=== TRAIN (?P<model>\S+): .+ \(holdout (?P<hold>\S+), seed=(?P<seed>\d+)\) ==="
    )
    rec = re.compile(
        r"RECORDED:\s+\S+\s+\|\s+holdout=\S+\s+\|\s+macro_f1=(?P<f1>[\d.]+)\s+\|\s+n=(?P<n>\d+)"
    )
    rows = []
    pending = None
    for line in text.splitlines():
        m = train_lodo.search(line)
        if m:
            pending = (m.group("model"), m.group("hold"), int(m.group("seed")))
            continue
        m = rec.search(line)
        if m and pending:
            rows.append(
                {
                    "model": pending[0],
                    "held_out_dataset": pending[1],
                    "seed": pending[2],
                    "macro_f1": float(m.group("f1")),
                    "n_samples": int(m.group("n")),
                }
            )
    return pd.DataFrame(rows)


def load_frozen_transfer() -> pd.DataFrame:
    base = pd.read_csv(ROOT / "frozen_results" / "crossdataset_matrix.csv")
    gap = pd.read_csv(ROOT / "frozen_results" / "generalization_gap.csv")
    # merge gap for in-dataset f1
    gcols = [
        "train_dataset",
        "test_dataset",
        "model",
        "seed",
        "in_dataset_macro_f1",
        "generalization_gap",
    ]
    if "augmentation" not in gap.columns:
        gap = gap.copy()
        gap["augmentation"] = "default"
    merged = base.merge(
        gap[
            [
                "train_dataset",
                "test_dataset",
                "model",
                "seed",
                "in_dataset_macro_f1",
                "generalization_gap",
            ]
        ],
        on=["train_dataset", "test_dataset", "model", "seed"],
        how="left",
    )
    merged["augmentation"] = "default"
    merged = merged.rename(
        columns={
            "macro_f1": "cross_macro_f1",
            "in_dataset_macro_f1": "in_macro_f1",
            "generalization_gap": "gap",
        }
    )
    return merged[
        [
            "train_dataset",
            "test_dataset",
            "model",
            "seed",
            "augmentation",
            "in_macro_f1",
            "cross_macro_f1",
            "gap",
        ]
    ]


def load_frozen_aug() -> pd.DataFrame:
    base = pd.read_csv(ROOT / "frozen_results" / "crossdataset_matrix_aug.csv")
    gap_path = ROOT / "frozen_results" / "generalization_gap_aug.csv"
    if not gap_path.exists():
        # week7 fallback
        gap_path = ROOT / "week7_results" / "generalization_gap_aug.csv"
    gap = pd.read_csv(gap_path)
    merged = base.merge(
        gap[
            [
                "train_dataset",
                "test_dataset",
                "model",
                "seed",
                "in_dataset_macro_f1",
                "generalization_gap",
            ]
        ],
        on=["train_dataset", "test_dataset", "model", "seed"],
        how="left",
        suffixes=("", "_g"),
    )
    if "augmentation" not in merged.columns:
        merged["augmentation"] = "strong"
    else:
        merged["augmentation"] = merged["augmentation"].replace({"strong": "strong"})
        merged.loc[merged["augmentation"].isna(), "augmentation"] = "strong"
    merged = merged.rename(
        columns={
            "macro_f1": "cross_macro_f1",
            "in_dataset_macro_f1": "in_macro_f1",
            "generalization_gap": "gap",
        }
    )
    return merged[
        [
            "train_dataset",
            "test_dataset",
            "model",
            "seed",
            "augmentation",
            "in_macro_f1",
            "cross_macro_f1",
            "gap",
        ]
    ]


def load_frozen_lodo() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "frozen_results" / "lodo_results.csv")
    return df[["held_out_dataset", "model", "seed", "macro_f1", "n_samples"]]


def fix_aug_from_checkpoint_in_week11() -> pd.DataFrame:
    """Use downloaded week11_light gap/matrix for seed 2024 strong rows."""
    frames = []
    for name in ["generalization_gap_aug.csv", "crossdataset_matrix_aug.csv"]:
        p = ROOT / "week11_light" / "multiseed" / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "cross_macro_f1" in df.columns:
            # gap file
            out = df.rename(
                columns={
                    "in_dataset_macro_f1": "in_macro_f1",
                    "generalization_gap": "gap",
                }
            )
            out["augmentation"] = "strong"
            frames.append(
                out[
                    [
                        "train_dataset",
                        "test_dataset",
                        "model",
                        "seed",
                        "augmentation",
                        "in_macro_f1",
                        "cross_macro_f1",
                        "gap",
                    ]
                ]
            )
        elif "macro_f1" in df.columns:
            out = df.rename(columns={"macro_f1": "cross_macro_f1"})
            out["augmentation"] = "strong"
            out["in_macro_f1"] = np.nan
            out["gap"] = np.nan
            frames.append(
                out[
                    [
                        "train_dataset",
                        "test_dataset",
                        "model",
                        "seed",
                        "augmentation",
                        "in_macro_f1",
                        "cross_macro_f1",
                        "gap",
                    ]
                ]
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    texts = user_texts()
    print(f"Parsing {len(texts)} user messages")

    xfer_parts, lodo_parts, gap_parts = [], [], []
    for text in texts:
        xf, _ld_unused = parse_recorded_with_context(text)
        if len(xf):
            xfer_parts.append(xf)
        ld = parse_lodo_from_recorded(text)
        if len(ld):
            lodo_parts.append(ld)
        g = parse_gap_tables(text)
        if len(g):
            gap_parts.append(g)

    xf = pd.concat(xfer_parts, ignore_index=True) if xfer_parts else pd.DataFrame()
    if gap_parts:
        gaps = pd.concat(gap_parts, ignore_index=True)
        gaps = gaps.drop_duplicates(
            subset=["model", "train_dataset", "test_dataset", "seed", "augmentation"],
            keep="last",
        )
        print("gap-table rows:", len(gaps), gaps.groupby(["seed", "augmentation"]).size().to_dict())
    else:
        gaps = pd.DataFrame()

    ckpt_cued = ROOT / "results" / "multiseed" / "_ckpt_cued_transfer.csv"
    ckpt_df = pd.read_csv(ckpt_cued) if ckpt_cued.exists() else pd.DataFrame()
    if len(ckpt_df):
        ckpt_df = ckpt_df.drop(columns=["checkpoint_path"], errors="ignore")
        print("ckpt-cued rows:", len(ckpt_df), ckpt_df.groupby(["seed", "augmentation"]).size().to_dict())

    if len(xf):
        xf = xf.drop_duplicates(
            subset=["model", "train_dataset", "test_dataset", "seed", "augmentation"],
            keep="last",
        )
        print("recorded xfer:", len(xf), xf.groupby(["seed", "augmentation"]).size().to_dict())

    frames = [
        f
        for f in [xf, gaps, ckpt_df, fix_aug_from_checkpoint_in_week11()]
        if len(f)
    ]
    transfer = pd.concat(frames, ignore_index=True)
    # When both in_macro and cross present, prefer those rows
    transfer = transfer.sort_values("in_macro_f1", na_position="first")
    transfer = transfer.drop_duplicates(
        subset=["model", "train_dataset", "test_dataset", "seed", "augmentation"],
        keep="last",
    )

    # Attach frozen seed 42
    frozen = pd.concat([load_frozen_transfer(), load_frozen_aug()], ignore_index=True)
    transfer = pd.concat([frozen, transfer], ignore_index=True)
    transfer = transfer.drop_duplicates(
        subset=["model", "train_dataset", "test_dataset", "seed", "augmentation"],
        keep="last",
    )
    transfer["seed"] = transfer["seed"].astype(int)

    print("\nFINAL transfer coverage:")
    print(transfer.groupby(["seed", "augmentation"]).size())

    # LODO
    ld = pd.concat(lodo_parts, ignore_index=True) if lodo_parts else pd.DataFrame()
    w11 = ROOT / "week11_light" / "multiseed" / "lodo_results.csv"
    if w11.exists():
        w = pd.read_csv(w11)[["held_out_dataset", "model", "seed", "macro_f1", "n_samples"]]
        ld = pd.concat([ld, w], ignore_index=True)
    if len(ld):
        ld = ld.dropna(subset=["seed"])
        ld["seed"] = pd.to_numeric(ld["seed"], errors="coerce")
        ld = ld.dropna(subset=["seed"])
        ld = ld.drop_duplicates(subset=["model", "held_out_dataset", "seed"], keep="last")
    ld = pd.concat([load_frozen_lodo(), ld], ignore_index=True)
    ld = ld.drop_duplicates(subset=["model", "held_out_dataset", "seed"], keep="last")
    ld["seed"] = ld["seed"].astype(int)
    print("\nFINAL lodo coverage:")
    print(ld.groupby("seed").size())

    transfer.to_csv(OUT / "transfer_all_seeds.csv", index=False)
    ld.to_csv(OUT / "lodo_all_seeds.csv", index=False)

    # --- Summaries ---
    # Per-cell mean±std over seeds
    cell = (
        transfer.groupby(
            ["augmentation", "train_dataset", "test_dataset", "model"], as_index=False
        )
        .agg(
            n_seeds=("seed", "nunique"),
            cross_mean=("cross_macro_f1", "mean"),
            cross_std=("cross_macro_f1", "std"),
            gap_mean=("gap", "mean"),
            gap_std=("gap", "std"),
        )
    )
    cell.to_csv(OUT / "transfer_cell_mean_std.csv", index=False)

    # Headline: mean across-seed std of cross macro-F1
    noise = cell.groupby("augmentation")["cross_std"].mean()
    print("\nMean across-seed std of cross macro-F1:")
    print(noise)

    # Augmentation delta per cell (strong - default), then mean over cells, with seed-level pairing
    base = transfer[transfer.augmentation == "default"].set_index(
        ["train_dataset", "test_dataset", "model", "seed"]
    )["cross_macro_f1"]
    strong = transfer[transfer.augmentation == "strong"].set_index(
        ["train_dataset", "test_dataset", "model", "seed"]
    )["cross_macro_f1"]
    paired = pd.DataFrame({"baseline": base, "strong": strong}).dropna()
    paired["delta"] = paired["strong"] - paired["baseline"]
    print(f"\nPaired aug deltas (n={len(paired)} seed-cells):")
    print(
        f"  mean delta={paired['delta'].mean():.4f}  "
        f"std={paired['delta'].std():.4f}  "
        f"median={paired['delta'].median():.4f}"
    )
    # mean delta per cell across seeds, then overall
    cell_delta = paired.groupby(level=["train_dataset", "test_dataset", "model"])["delta"].agg(
        ["mean", "std", "count"]
    )
    print(
        f"  mean of per-cell mean-deltas={cell_delta['mean'].mean():.4f}  "
        f"(cells with 3 seeds: {(cell_delta['count']==3).sum()}/{len(cell_delta)})"
    )
    cell_delta.reset_index().to_csv(OUT / "aug_delta_per_cell.csv", index=False)
    paired.reset_index().to_csv(OUT / "aug_delta_paired.csv", index=False)

    # By model
    by_model = (
        transfer.groupby(["augmentation", "model"])
        .agg(cross_mean=("cross_macro_f1", "mean"), cross_std=("cross_macro_f1", "std"), n=("cross_macro_f1", "size"))
        .reset_index()
    )
    by_model.to_csv(OUT / "transfer_by_model.csv", index=False)
    print("\nBy model:")
    print(by_model.to_string(index=False))

    # LODO mean±std
    lodo_cell = (
        ld.groupby(["held_out_dataset", "model"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            f1_mean=("macro_f1", "mean"),
            f1_std=("macro_f1", "std"),
        )
    )
    lodo_hold = (
        ld.groupby("held_out_dataset", as_index=False)
        .agg(f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"), n=("macro_f1", "size"))
    )
    lodo_cell.to_csv(OUT / "lodo_cell_mean_std.csv", index=False)
    lodo_hold.to_csv(OUT / "lodo_by_holdout_mean_std.csv", index=False)
    print("\nLODO by holdout:")
    print(lodo_hold.to_string(index=False))

    # Seed-42 headline check: BRRI→RiceLeafBD MobileNetV2
    key = transfer[
        (transfer.train_dataset == "brri_rice_disease_pest")
        & (transfer.test_dataset == "riceleafbd")
        & (transfer.model == "mobilenetv2_100")
        & (transfer.augmentation == "default")
    ][["seed", "cross_macro_f1", "gap"]].sort_values("seed")
    print("\nHeadline check BRRI->RiceLeafBD MobileNetV2 baseline:")
    print(key.to_string(index=False))

    # Coverage gap report
    expected_seeds = {42, 7, 2024}
    missing = []
    for aug in ["default", "strong"]:
        for (tr, te, model), g in transfer[transfer.augmentation == aug].groupby(
            ["train_dataset", "test_dataset", "model"]
        ):
            have = set(g["seed"].tolist())
            for s in expected_seeds - have:
                missing.append((aug, tr, te, model, s))
    print(f"\nMissing transfer cells: {len(missing)}")
    for row in missing[:30]:
        print(" ", row)
    if len(missing) > 30:
        print(f"  ... +{len(missing)-30} more")

    # Write markdown report
    report = OUT / "PHASE1_MULTISEED_SUMMARY.md"
    mean_noise_base = float(noise.get("default", np.nan))
    mean_noise_aug = float(noise.get("strong", np.nan))
    mean_delta = float(paired["delta"].mean()) if len(paired) else float("nan")
    report.write_text(
        f"""# Phase 1 multi-seed summary

Reconstructed from `frozen_results/` (seed 42) + campaign logs / `week11_light/` (seeds 7, 2024).

## Coverage

Transfer cells by seed × augmentation:

```
{transfer.groupby(['seed','augmentation']).size().to_string()}
```

LODO by seed:

```
{ld.groupby('seed').size().to_string()}
```

Missing transfer seed-cells: **{len(missing)}** (see script output).  
If non-zero, prefer re-downloading `results/multiseed/*.csv` from Kaggle.

## Headline numbers (paper-facing)

| Quantity | Value |
|---|---|
| Mean across-seed std (baseline cross macro-F1) | {mean_noise_base:.4f} |
| Mean across-seed std (strong-aug cross macro-F1) | {mean_noise_aug:.4f} |
| Mean paired aug delta (strong − baseline) | {mean_delta:.4f} |
| Paired seed-cells used for delta | {len(paired)} |

Interpretation cue from the workflow: compare mean aug gain ({mean_delta:.3f}) to baseline seed noise ({mean_noise_base:.3f}).

## LODO (models pooled)

```
{lodo_hold.to_string(index=False)}
```

## Outputs

- `transfer_all_seeds.csv`
- `lodo_all_seeds.csv`
- `transfer_cell_mean_std.csv`
- `aug_delta_per_cell.csv` / `aug_delta_paired.csv`
- `lodo_cell_mean_std.csv` / `lodo_by_holdout_mean_std.csv`
""",
        encoding="utf-8",
    )
    print("\nWrote", report)


if __name__ == "__main__":
    main()
