from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.eval import evaluate
from src.train import train, training_stem
from src.run_registry import RunRegistry, default_registry_path, make_run_id
from src.utils import load_config

DEFAULT_MODELS = ["mobilenetv2_100", "efficientnet_b0", "resnet50"]
ALL_DATASETS = ["riceleafbd", "dhan_shomadhan", "brri_rice_disease_pest"]

KEY_COLUMNS = ["held_out_dataset", "model", "seed"]


@dataclass(frozen=True)
class LodoSplit:
    held_out: str
    train_datasets: tuple[str, ...]
    classes: tuple[str, ...]

    @property
    def class_string(self) -> str:
        return "|".join(self.classes)

    @property
    def run_tag(self) -> str:
        return f"lodo-holdout-{self.held_out}"


def _parse_image_roots(items: list[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    roots: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --image_roots item '{item}'. Use dataset=/abs/path"
            )
        key, value = item.split("=", 1)
        roots[key.strip()] = value.strip()
    return roots


def build_lodo_splits(manifest: pd.DataFrame) -> list[LodoSplit]:
    """Largest non-degenerate shared label space per held-out dataset.

    The label space is every class in the held-out dataset that also appears
    in at least one training dataset. Classes unique to the held-out set cannot
    be learned and are excluded; this is documented per run.
    """
    classes_by_dataset = {
        dataset: set(
            manifest.loc[manifest["dataset"] == dataset, "mapped_class"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        for dataset in ALL_DATASETS
    }

    splits: list[LodoSplit] = []
    for held_out in ALL_DATASETS:
        train_datasets = tuple(d for d in ALL_DATASETS if d != held_out)
        train_union: set[str] = set()
        for dataset in train_datasets:
            train_union |= classes_by_dataset[dataset]
        shared = sorted(classes_by_dataset[held_out] & train_union)
        if len(shared) < 2:
            raise ValueError(
                f"LODO for held-out '{held_out}' has <2 usable classes: {shared}"
            )
        splits.append(
            LodoSplit(held_out, train_datasets, tuple(shared))
        )
    return splits


def _has_key(df: pd.DataFrame, row: dict[str, object]) -> bool:
    if df.empty or not set(KEY_COLUMNS).issubset(df.columns):
        return False
    mask = pd.Series(True, index=df.index)
    for column in KEY_COLUMNS:
        mask &= df[column].astype(str) == str(row[column])
    return bool(mask.any())


def _upsert_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_csv(path)
        if set(KEY_COLUMNS).issubset(df.columns):
            mask = pd.Series(True, index=df.index)
            for column in KEY_COLUMNS:
                mask &= df[column].astype(str) == str(row[column])
            df = df.loc[~mask].copy()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _print_summary(results_path: Path) -> None:
    if not results_path.exists():
        print("LODO results are not complete yet.")
        return
    results = pd.read_csv(results_path)
    print("\n=== Leave-one-dataset-out results ===")
    print(
        results.sort_values(["held_out_dataset", "model"]).to_string(index=False)
    )
    if not results.empty:
        print("\n=== Mean macro-F1 by held-out dataset (models pooled) ===")
        print(
            results.groupby("held_out_dataset")["macro_f1"]
            .mean()
            .to_string(float_format=lambda value: f"{value:.4f}")
        )


def run_lodo(
    config_path: str = "config.yaml",
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    image_roots: dict[str, str] | None = None,
    skip_existing: bool = True,
    augmentation: str = "default",
    output_dir: str | Path | None = None,
    results_filename: str = "lodo_results.csv",
) -> Path:
    config = load_config(config_path)
    results_root = Path(config["results_root"])
    manifest_path = results_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Prepare Kaggle paths first "
            "(see notebooks/kaggle_week7.md)."
        )

    manifest = pd.read_csv(manifest_path)
    models = models or DEFAULT_MODELS
    seeds = seeds or [
        int(seed) for seed in config.get("seeds", [config.get("seed", 42)])
    ]
    splits = build_lodo_splits(manifest)
    output_root = Path(output_dir) if output_dir is not None else results_root
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / results_filename
    registry = RunRegistry(default_registry_path(results_root))
    split_seed = int(config.get("split_seed", 42))

    for split in splits:
        classes = list(split.classes)
        print(
            f"\n### LODO hold out {split.held_out} | "
            f"train {'+'.join(split.train_datasets)} | classes {classes}"
        )
        for model_name in models:
            for seed in seeds:
                key_row: dict[str, object] = {
                    "held_out_dataset": split.held_out,
                    "model": model_name,
                    "seed": seed,
                }
                run_id = make_run_id(
                    model=model_name,
                    train_datasets=list(split.train_datasets),
                    train_seed=seed,
                    eval_dataset=split.held_out,
                    run_tag=split.run_tag,
                    augmentation=augmentation,
                )
                existing = (
                    pd.read_csv(results_path)
                    if results_path.exists()
                    else pd.DataFrame()
                )
                if skip_existing and registry.is_complete(run_id):
                    print(
                        f"SKIP registry-complete: {model_name} | "
                        f"holdout={split.held_out} | seed={seed}"
                    )
                    continue
                if skip_existing and _has_key(existing, key_row):
                    print(
                        f"SKIP completed: {model_name} | "
                        f"holdout={split.held_out} | seed={seed}"
                    )
                    continue

                artifact_stem = training_stem(
                    model_name,
                    list(split.train_datasets),
                    seed,
                    split.run_tag,
                )
                checkpoint_path = (
                    results_root / "checkpoints" / f"{artifact_stem}.pth"
                )

                registry.register_run(
                    run_id=run_id,
                    experiment_type="lodo",
                    model=model_name,
                    train_datasets=list(split.train_datasets),
                    eval_dataset=split.held_out,
                    classes=classes,
                    split_seed=split_seed,
                    train_seed=seed,
                    augmentation=augmentation,
                    checkpoint_path=checkpoint_path if checkpoint_path.exists() else None,
                )

                if checkpoint_path.exists():
                    print(f"\nREUSE checkpoint after interruption: {checkpoint_path}")
                else:
                    print(
                        f"\n=== TRAIN {model_name}: "
                        f"{'+'.join(split.train_datasets)} "
                        f"(holdout {split.held_out}, seed={seed}) ==="
                    )
                    checkpoint_path, _ = train(
                        model_name=model_name,
                        train_datasets=list(split.train_datasets),
                        classes=classes,
                        config=config,
                        train_seed=seed,
                        split_seed=split_seed,
                        eval_dataset=split.held_out,
                        image_roots=image_roots,
                        run_tag=split.run_tag,
                        augmentation=augmentation,
                    )

                print(f"=== LODO EVAL on held-out {split.held_out} ===")
                result = evaluate(
                    checkpoint_path=checkpoint_path,
                    eval_dataset=split.held_out,
                    classes=classes,
                    split="test",
                    train_seed=seed,
                    split_seed=split_seed,
                    config_path=config_path,
                    sample_n=10,
                    image_roots=image_roots,
                )

                row = {
                    **key_row,
                    "train_datasets": "+".join(split.train_datasets),
                    "classes": split.class_string,
                    "augmentation": augmentation,
                    "accuracy": result["accuracy"],
                    "macro_f1": result["macro_f1"],
                    "n_samples": result["n_samples"],
                    "per_class_f1": json.dumps(result["per_class_f1"]),
                    "checkpoint_path": str(checkpoint_path),
                }
                _upsert_csv(results_path, row)
                registry.mark_complete(
                    run_id,
                    checkpoint_path=checkpoint_path,
                    predictions_path=result.get("predictions_path"),
                )
                print(
                    f"RECORDED: {model_name} | holdout={split.held_out} | "
                    f"macro_f1={result['macro_f1']:.4f} | "
                    f"n={result['n_samples']}"
                )

    _print_summary(results_path)
    print(f"\nWrote: {results_path}")
    return results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 7 leave-one-dataset-out mitigation runner."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--image_roots",
        nargs="*",
        default=None,
        help="dataset=/abs/path pairs for Kaggle mounts.",
    )
    parser.add_argument(
        "--augmentation",
        default="default",
        choices=["default", "strong"],
        help="Augmentation profile (default keeps LODO as a single variable).",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-run combinations already present in the output CSV.",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only print the existing LODO summary.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["results_root"])
    if args.summary_only:
        _print_summary(root / "lodo_results.csv")
    else:
        run_lodo(
            config_path=args.config,
            models=args.models,
            seeds=args.seeds,
            image_roots=_parse_image_roots(args.image_roots),
            skip_existing=not args.no_skip_existing,
            augmentation=args.augmentation,
        )
