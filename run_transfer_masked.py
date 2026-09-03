"""Week 13 — retrain the 18-pair transfer matrix on masked images.

Reuses Week 5 protocol via ``src.train.train`` / ``src.eval.evaluate`` and the
same ``TRANSFER_PAIRS`` as ``run_crossdataset.py``. Images are read from
``data/masked/{condition}/`` by remapping ``data/raw`` → masked root.

Writes:
  frozen_results_v2/crossdataset_matrix_masked_{condition}.csv
  results_masked/{condition}/checkpoints/*.pth  (working artifacts)

Never writes to frozen_results/. Never hardcodes metrics.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import pandas as pd
import torch

from run_crossdataset import (
    DEFAULT_MODELS,
    TransferPair,
    _select_pairs,
    _upsert_csv,
    _validate_pair_classes,
)
from src.eval import evaluate
from src.train import train, training_stem
from src.utils import load_config, set_seed

ROOT = Path(__file__).resolve().parent
FROZEN_MANIFEST = ROOT / "frozen_results" / "manifest.csv"
BASELINE_MATRIX = ROOT / "frozen_results" / "crossdataset_matrix.csv"
OUT_V2 = ROOT / "frozen_results_v2"
AUDIT_DECISION = ROOT / "notes" / "mask_audit" / "audit_decision.md"
VALID_CONDITIONS = ("sam_leaf", "hsv_leaf")


def _print_seeds(seed: int) -> None:
    print(f"python random.seed({seed})")
    print(f"numpy.random.seed({seed})")
    print(f"torch.manual_seed({seed})")
    if torch.cuda.is_available():
        print(f"torch.cuda.manual_seed_all({seed})")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


def _masked_root(condition: str) -> Path:
    return ROOT / "data" / "masked" / condition


def _path_remap(condition: str) -> tuple[str, str]:
    return ("data/raw", f"data/masked/{condition}")


def _relative_under_raw(image_path: str) -> Path:
    posix = str(image_path).replace("\\", "/")
    marker = "data/raw/"
    if marker not in posix:
        raise ValueError(f"Expected image_path under data/raw/, got: {image_path!r}")
    return Path(posix.split(marker, 1)[1])


def _load_frozen_manifest() -> pd.DataFrame:
    if not FROZEN_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Missing immutable manifest: {FROZEN_MANIFEST}. "
            "Do not regenerate it."
        )
    df = pd.read_csv(FROZEN_MANIFEST)
    nondup = df[df["is_duplicate"] == False].reset_index(drop=True)
    if len(nondup) != 5419:
        raise ValueError(
            f"Expected 5419 non-duplicate rows, got {len(nondup)} from {FROZEN_MANIFEST}"
        )
    return nondup


def _assert_masked_images_exist(manifest: pd.DataFrame, condition: str) -> None:
    root = _masked_root(condition)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Masked image directory missing: {root}. "
            "Run Week 12 mask builders first."
        )
    missing: list[str] = []
    for image_path in manifest["image_path"].astype(str):
        dst = root / _relative_under_raw(image_path)
        if not dst.is_file():
            missing.append(str(dst))
    if missing:
        preview = "\n  ".join(missing[:10])
        more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} masked images missing under {root}.\n  {preview}{more}"
        )
    print(f"Verified {len(manifest)} masked images under {root}")


def _prepare_working_results(condition: str) -> Path:
    """Copy frozen manifest into a writable working results_root for train/eval."""
    work = ROOT / "results_masked" / condition
    work.mkdir(parents=True, exist_ok=True)
    (work / "checkpoints").mkdir(parents=True, exist_ok=True)
    (work / "predictions").mkdir(parents=True, exist_ok=True)
    dest_manifest = work / "manifest.csv"
    shutil.copy2(FROZEN_MANIFEST, dest_manifest)
    return work


def _assert_condition_cleared(condition: str, allow_without_audit: bool) -> None:
    if allow_without_audit:
        print(
            "WARNING: --allow-without-audit set; skipping Week 12 decision gate."
        )
        return
    if not AUDIT_DECISION.is_file():
        raise FileNotFoundError(
            f"Missing audit decision: {AUDIT_DECISION}. "
            "Fill notes/mask_audit/audit_sheet.csv, run "
            "scripts/parse_audit_verdicts.py, or pass --allow-without-audit "
            "only for smoke tests."
        )
    text = AUDIT_DECISION.read_text(encoding="utf-8")
    marker = f"**Variants cleared for Week 13:**"
    if marker not in text:
        # Also accept FAIL-both block
        if "Week 13 blocked" in text:
            raise RuntimeError(
                "Week 12 audit blocked Week 13. Do not train masked transfer."
            )
        raise RuntimeError(
            f"Could not parse cleared variants from {AUDIT_DECISION}"
        )
    line = [ln for ln in text.splitlines() if "Variants cleared for Week 13" in ln][0]
    if condition not in line:
        raise RuntimeError(
            f"Condition '{condition}' was not cleared by Week 12 audit.\n"
            f"Decision file says: {line}\n"
            "Do not train this variant."
        )
    print(f"Audit gate OK for condition={condition}")


def _matrix_out_path(condition: str) -> Path:
    return OUT_V2 / f"crossdataset_matrix_masked_{condition}.csv"


def _compare_to_baseline(masked: pd.DataFrame, seed: int) -> None:
    if not BASELINE_MATRIX.is_file():
        raise FileNotFoundError(f"Missing baseline matrix: {BASELINE_MATRIX}")
    base = pd.read_csv(BASELINE_MATRIX)
    base42 = base[base["seed"] == seed].copy()
    if len(base42) == 0:
        raise ValueError(f"No seed={seed} rows in {BASELINE_MATRIX}")

    keys = ["train_dataset", "test_dataset", "model", "classes"]
    m = masked.copy()
    for col in keys + ["macro_f1"]:
        if col not in m.columns:
            raise KeyError(f"Masked matrix missing column: {col}")

    merged = base42[keys + ["macro_f1"]].rename(columns={"macro_f1": "baseline_f1"}).merge(
        m[keys + ["macro_f1"]].rename(columns={"macro_f1": "masked_f1"}),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) == 0:
        raise RuntimeError("No overlapping keys between baseline and masked matrices.")

    delta = merged["masked_f1"] - merged["baseline_f1"]
    print(f"\nBaseline (raw, seed {seed}):")
    print(f"  mean macro-F1: {base42['macro_f1'].mean():.3f}")
    print(f"  std macro-F1: {base42['macro_f1'].std(ddof=0):.3f}")
    print(f"\nMasked (seed {seed}):")
    print(f"  mean macro-F1: {merged['masked_f1'].mean():.3f}")
    print(f"  std macro-F1: {merged['masked_f1'].std(ddof=0):.3f}")
    print("\nDelta (masked - raw):")
    print(f"  mean: {delta.mean():+.3f}")
    print(f"  range: [{delta.min():+.3f}, {delta.max():+.3f}]")
    print(f"  positive: {int((delta > 0).sum())}/{len(delta)} pairs")


def run_transfer_masked(
    condition: str,
    seed: int = 42,
    models: list[str] | None = None,
    pair_keys: list[str] | None = None,
    skip_existing: bool = True,
    allow_without_audit: bool = False,
    dry_run: bool = False,
    config_path: str = "config.yaml",
) -> Path:
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"Invalid --condition {condition!r}. Choose from {VALID_CONDITIONS}"
        )

    random.seed(seed)
    set_seed(seed)
    _print_seeds(seed)

    _assert_condition_cleared(condition, allow_without_audit=allow_without_audit)

    manifest = _load_frozen_manifest()
    print(f"Loaded manifest: {len(manifest)} non-duplicate rows from {FROZEN_MANIFEST}")
    print(f"Condition: {condition}")
    print(f"Seed: {seed}")

    _assert_masked_images_exist(manifest, condition)

    models = models or list(DEFAULT_MODELS)
    pairs = _select_pairs(pair_keys)
    expected_runs = len(pairs) * len(models)
    print(
        f"\nStarting transfer training "
        f"({len(pairs)} pairs × {len(models)} models = {expected_runs} runs)..."
    )

    if dry_run:
        for pair in pairs:
            _validate_pair_classes(manifest, pair)
            for model_name in models:
                print(f"  PENDING: {model_name} | {pair.key} | seed={seed}")
        print("Dry run complete — no training started.")
        return _matrix_out_path(condition)

    work = _prepare_working_results(condition)
    # evaluate() reloads config.yaml; RESULTS_ROOT must point at the working dir.
    os.environ["RESULTS_ROOT"] = str(work)
    config = load_config(config_path)
    config["results_root"] = str(work)
    config["seed"] = seed
    config["split_seed"] = 42

    path_remap = _path_remap(condition)
    print(f"path_remap: {path_remap[0]!r} -> {path_remap[1]!r}")
    print(f"Working results_root: {work}")

    OUT_V2.mkdir(parents=True, exist_ok=True)
    matrix_path = _matrix_out_path(condition)

    for pair in pairs:
        _validate_pair_classes(manifest, pair)
        classes = list(pair.classes)
        for model_name in models:
            key_row: dict[str, object] = {
                "train_dataset": pair.train_dataset,
                "test_dataset": pair.test_dataset,
                "model": model_name,
                "classes": pair.class_string,
                "seed": seed,
                "condition": condition,
            }
            matrix_df = (
                pd.read_csv(matrix_path) if matrix_path.exists() else pd.DataFrame()
            )
            if skip_existing and not matrix_df.empty:
                mask = pd.Series(True, index=matrix_df.index)
                for column, value in key_row.items():
                    if column not in matrix_df.columns:
                        mask[:] = False
                        break
                    mask &= matrix_df[column].astype(str) == str(value)
                if bool(mask.any()):
                    print(
                        "SKIP completed: "
                        f"{model_name} | {pair.key} | seed={seed} | {condition}"
                    )
                    continue

            run_tag = f"{pair.run_tag}__mask-{condition}"
            artifact_stem = training_stem(
                model_name, [pair.train_dataset], seed, run_tag
            )
            checkpoint_path = work / "checkpoints" / f"{artifact_stem}.pth"

            if checkpoint_path.exists():
                print(f"\nREUSE checkpoint: {checkpoint_path}")
            else:
                print(
                    f"\n=== TRAIN {model_name}: {pair.train_dataset} "
                    f"-> {pair.test_dataset} (seed={seed}, {condition}) ==="
                )
                checkpoint_path, _ = train(
                    model_name=model_name,
                    train_datasets=[pair.train_dataset],
                    classes=classes,
                    config=config,
                    train_seed=seed,
                    split_seed=42,
                    eval_dataset=pair.train_dataset,
                    path_remap=path_remap,
                    run_tag=run_tag,
                    augmentation="default",
                )

            cross = evaluate(
                checkpoint_path=checkpoint_path,
                eval_dataset=pair.test_dataset,
                classes=classes,
                split="test",
                train_seed=seed,
                split_seed=42,
                config_path=config_path,
                sample_n=10,
                path_remap=path_remap,
            )

            matrix_row = {
                **key_row,
                "accuracy": cross["accuracy"],
                "macro_f1": cross["macro_f1"],
                "n_samples": cross["n_samples"],
                "checkpoint_path": str(checkpoint_path),
            }
            _upsert_csv(matrix_path, matrix_row)
            print(
                "RECORDED: "
                f"{model_name} | {pair.key} | condition={condition} | "
                f"macro_f1={cross['macro_f1']:.4f} | n={cross['n_samples']}"
            )

    if not matrix_path.is_file():
        raise FileNotFoundError(f"Expected output missing: {matrix_path}")

    out = pd.read_csv(matrix_path)
    if len(out) != expected_runs:
        raise AssertionError(
            f"Expected {expected_runs} (pair, model) rows for seed={seed} "
            f"condition={condition}, got {len(out)}"
        )

    # Spot-check one checkpoint exists.
    sample_ckpt = Path(str(out.iloc[0]["checkpoint_path"]))
    if not sample_ckpt.is_file():
        raise FileNotFoundError(f"Spot-check checkpoint missing: {sample_ckpt}")

    print(f"\nSaved: {matrix_path} ({len(out)} rows)")
    print(f"Checkpoints under: {work / 'checkpoints'}")
    _compare_to_baseline(out, seed=seed)

    print(f"\nShape: {out.shape}")
    print(out.head())
    print(out[["macro_f1", "accuracy", "n_samples"]].describe())
    print("Do NOT interpret these numbers. Report to researcher for manuscript writing.")
    return matrix_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        required=True,
        choices=list(VALID_CONDITIONS),
        help="Masking variant cleared by Week 12 audit.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Optional subset as train:test keys.",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-run combinations already present in the output CSV.",
    )
    parser.add_argument(
        "--allow-without-audit",
        action="store_true",
        help="Smoke-test only: skip Week 12 audit gate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and list pending jobs; do not train.",
    )
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_transfer_masked(
        condition=args.condition,
        seed=args.seed,
        models=args.models,
        pair_keys=args.pairs,
        skip_existing=not args.no_skip_existing,
        allow_without_audit=args.allow_without_audit,
        dry_run=args.dry_run,
        config_path=args.config,
    )
