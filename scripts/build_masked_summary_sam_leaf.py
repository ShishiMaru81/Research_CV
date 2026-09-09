"""Build the two-seed SAM-masked transfer summary from frozen CSV inputs.

No metric or finding is embedded in this script. All values are computed from
the seed-matched raw and SAM-masked transfer tables at runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "frozen_results_v2"
RAW_PATH = V2 / "transfer_all_seeds.csv"
MASKED_PATHS = (
    V2 / "crossdataset_matrix_masked_sam_leaf.csv",
    V2 / "crossdataset_matrix_masked_sam_leaf_seed2024.csv",
)
OUTPUT_PATH = V2 / "masked_summary_sam_leaf.csv"
KEY_COLUMNS = ["train_dataset", "test_dataset", "model"]
RAW_METRIC = "cross_macro_f1"
MASKED_METRIC = "macro_f1"
CONDITION = "sam_leaf"
EXPECTED_CELLS_PER_SEED = 18


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file missing: {path}")


def load_raw() -> pd.DataFrame:
    require_file(RAW_PATH)
    raw = pd.read_csv(RAW_PATH)
    required = set(KEY_COLUMNS) | {"seed", "augmentation", RAW_METRIC}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{RAW_PATH} missing columns: {missing}")
    raw = raw[raw["augmentation"] == "default"].copy()
    if raw.empty:
        raise ValueError(f"No default-augmentation rows in {RAW_PATH}")
    return raw


def load_masked(path: Path) -> pd.DataFrame:
    require_file(path)
    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS) | {
        "seed",
        "condition",
        MASKED_METRIC,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if len(frame) != EXPECTED_CELLS_PER_SEED:
        raise ValueError(
            f"Expected {EXPECTED_CELLS_PER_SEED} rows in {path}, got {len(frame)}"
        )
    if frame[KEY_COLUMNS].duplicated().any():
        raise ValueError(f"Duplicate transfer keys in {path}")
    conditions = set(frame["condition"].astype(str))
    if conditions != {CONDITION}:
        raise ValueError(
            f"{path}.condition must be {CONDITION!r}, got {sorted(conditions)}"
        )
    return frame


def build_summary() -> pd.DataFrame:
    raw = load_raw()
    paired_frames: list[pd.DataFrame] = []
    seeds: list[int] = []

    for path in MASKED_PATHS:
        masked = load_masked(path)
        seed_values = pd.to_numeric(masked["seed"], errors="raise").unique()
        if len(seed_values) != 1:
            raise ValueError(f"{path} must contain exactly one seed")
        seed = int(seed_values[0])
        if seed in seeds:
            raise ValueError(f"Duplicate masked seed across inputs: {seed}")
        seeds.append(seed)

        raw_seed = raw[raw["seed"] == seed].copy()
        if len(raw_seed) != EXPECTED_CELLS_PER_SEED:
            raise ValueError(
                f"Expected {EXPECTED_CELLS_PER_SEED} raw default rows for seed "
                f"{seed}, got {len(raw_seed)} in {RAW_PATH}"
            )
        if raw_seed[KEY_COLUMNS].duplicated().any():
            raise ValueError(f"Duplicate raw transfer keys for seed {seed}")

        paired = raw_seed[KEY_COLUMNS + [RAW_METRIC]].merge(
            masked[KEY_COLUMNS + [MASKED_METRIC]],
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != EXPECTED_CELLS_PER_SEED:
            raise ValueError(
                f"Expected {EXPECTED_CELLS_PER_SEED} matched cells for seed "
                f"{seed}, got {len(paired)}"
            )
        paired["seed"] = seed
        paired["delta"] = paired[MASKED_METRIC] - paired[RAW_METRIC]
        paired_frames.append(paired)

    pool = pd.concat(paired_frames, ignore_index=True)
    if not np.isfinite(
        pool[[RAW_METRIC, MASKED_METRIC, "delta"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Paired metrics contain NaN or infinite values")

    statistic, p_value = wilcoxon(pool["delta"])
    summary = pd.DataFrame(
        [
            {
                "condition": CONDITION,
                "n_seeds": len(seeds),
                "n_pairs": len(pool),
                "mean_cross_macro_f1": pool[MASKED_METRIC].mean(),
                "std_cross_macro_f1": pool[MASKED_METRIC].std(),
                "mean_paired_delta": pool["delta"].mean(),
                "median_paired_delta": pool["delta"].median(),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p": p_value,
                "n_positive": int((pool["delta"] > 0).sum()),
            }
        ]
    )
    return summary


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing frozen v2 output: {OUTPUT_PATH}"
        )
    summary = build_summary()
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Shape: {summary.shape}")
    print(summary.head())
    print(summary.describe(include="all"))
    print(f"File written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
