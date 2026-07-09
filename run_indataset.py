from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval import evaluate
from src.train import discover_classes, train
from src.utils import load_config


DEFAULT_MODELS = ["mobilenetv2_100", "efficientnet_b0", "resnet50"]
DEFAULT_DATASETS = ["riceleafbd", "dhan_shomadhan", "brri_rice_disease_pest"]


def _parse_image_roots(items: list[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    roots: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --image_roots item '{item}'. Use dataset=/abs/path")
        key, value = item.split("=", 1)
        roots[key.strip()] = value.strip()
    return roots


def _append_result(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Replace an existing identical run if present.
        mask = (
            (df["model"] == row["model"])
            & (df["dataset"] == row["dataset"])
            & (df["seed"] == row["seed"])
        )
        df = df.loc[~mask].copy()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)


def _print_summary(csv_path: Path) -> None:
    if not csv_path.exists():
        print("No indataset_results.csv yet.")
        return
    df = pd.read_csv(csv_path)
    print("\n=== In-dataset results (all rows) ===")
    print(df.to_string(index=False))

    summary = (
        df.groupby(["model", "dataset"], as_index=False)
        .agg(accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
             accuracy_std=("accuracy", "std"), macro_f1_std=("macro_f1", "std"), n=("macro_f1", "count"))
        .sort_values(["model", "dataset"])
    )
    print("\n=== Summary mean +/- std ===")
    for _, row in summary.iterrows():
        f1_std = 0.0 if pd.isna(row["macro_f1_std"]) else row["macro_f1_std"]
        acc_std = 0.0 if pd.isna(row["accuracy_std"]) else row["accuracy_std"]
        print(
            f"{row['model']:16s} | {row['dataset']:24s} | "
            f"acc={row['accuracy_mean']:.4f}+/-{acc_std:.4f} | "
            f"macro_f1={row['macro_f1_mean']:.4f}+/-{f1_std:.4f} | n={int(row['n'])}"
        )

    low = df[df["macro_f1"] < 0.80]
    if len(low) > 0:
        print("\nWARNING: some runs are below macro-F1 0.80. Investigate before Week 5:")
        print(low[["model", "dataset", "seed", "macro_f1"]].to_string(index=False))
    else:
        print("\nAll recorded runs are >= 0.80 macro-F1.")


def run_indataset(
    config_path: str = "config.yaml",
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    image_roots: dict[str, str] | None = None,
    skip_existing: bool = True,
) -> Path:
    config = load_config(config_path)
    results_root = Path(config["results_root"])
    manifest_path = results_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Prepare Kaggle paths first "
            "(see notebooks/kaggle_week4.md)."
        )

    models = models or DEFAULT_MODELS
    datasets = datasets or DEFAULT_DATASETS
    seeds = seeds or [int(s) for s in config.get("seeds", [config.get("seed", 42)])]

    out_csv = results_root / "indataset_results.csv"
    existing = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()

    for model_name in models:
        for dataset in datasets:
            classes = discover_classes(manifest_path, [dataset])
            for seed in seeds:
                if skip_existing and not existing.empty:
                    hit = existing[
                        (existing["model"] == model_name)
                        & (existing["dataset"] == dataset)
                        & (existing["seed"] == seed)
                    ]
                    if len(hit) > 0:
                        print(f"SKIP existing: {model_name} | {dataset} | seed={seed}")
                        continue

                print(f"\n=== TRAIN {model_name} on {dataset} (seed={seed}) ===")
                print(f"classes ({len(classes)}): {classes}")
                checkpoint_path, _ = train(
                    model_name=model_name,
                    train_datasets=[dataset],
                    classes=classes,
                    config=config,
                    seed=seed,
                    eval_dataset=dataset,
                    image_roots=image_roots,
                )

                print(f"=== EVAL {model_name} on {dataset} (seed={seed}) ===")
                metrics = evaluate(
                    checkpoint_path=checkpoint_path,
                    eval_dataset=dataset,
                    classes=classes,
                    split="test",
                    seed=seed,
                    config_path=config_path,
                    sample_n=5,
                    image_roots=image_roots,
                )

                row = {
                    "model": model_name,
                    "dataset": dataset,
                    "seed": seed,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "n_samples": metrics["n_samples"],
                    "checkpoint_path": str(checkpoint_path),
                }
                _append_result(out_csv, row)
                print(
                    f"RECORDED: {model_name} | {dataset} | seed={seed} | "
                    f"acc={row['accuracy']:.4f} | macro_f1={row['macro_f1']:.4f}"
                )

    _print_summary(out_csv)
    print(f"\nWrote: {out_csv}")
    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 4 in-dataset baseline matrix.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--image_roots",
        nargs="*",
        default=None,
        help="dataset=/abs/path pairs for Kaggle mounts.",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-run even if a model/dataset/seed row already exists.",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only print summary from existing indataset_results.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.summary_only:
        cfg = load_config(args.config)
        _print_summary(Path(cfg["results_root"]) / "indataset_results.csv")
    else:
        run_indataset(
            config_path=args.config,
            models=args.models,
            datasets=args.datasets,
            seeds=args.seeds,
            image_roots=_parse_image_roots(args.image_roots),
            skip_existing=not args.no_skip_existing,
        )
