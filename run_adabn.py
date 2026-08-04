"""Run Adaptive BatchNorm (AdaBN) on cross-dataset baseline checkpoints.

For each transfer pair × model × seed: load the *source-trained* checkpoint,
recalibrate BN stats on the *target train* split (unlabeled for adaptation),
evaluate on the *target test* split, and compare to the unrecalibrated baseline.

Usage (Kaggle, after manifest + checkpoints are present)::

    python -m run_adabn --seeds 42
    python -m run_adabn --seeds 42 7 2024 --dry_run
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from run_crossdataset import TRANSFER_PAIRS, TransferPair
from src.adabn import adapt_batch_norm, count_batch_norm_layers
from src.data_loader import default_eval_transform, make_loaders
from src.eval import collect_predictions, load_checkpoint, save_predictions_csv
from src.train import build_model
from src.utils import get_device, load_config, set_seed
from sklearn.metrics import accuracy_score, f1_score


DEFAULT_MODELS = ["mobilenetv2_100", "efficientnet_b0", "resnet50"]
DEFAULT_SEEDS = [42]


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


def _checkpoint_for_pair(
    results_root: Path,
    model_name: str,
    pair: TransferPair,
    seed: int,
    augmentation: str = "default",
) -> Path:
    class_tag = "+".join(pair.classes)
    stem = (
        f"{model_name}__train-{pair.train_dataset}"
        f"__run-to-{pair.test_dataset}__classes-{class_tag}"
    )
    if augmentation == "strong":
        stem += "__aug-strong"
    stem += f"__seed{seed}"
    path = results_root / "checkpoints" / f"{stem}.pth"
    return path


def _metrics_from_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    classes: list[str],
) -> dict[str, Any]:
    paths, labels, preds, probs = collect_predictions(model, loader, device)
    accuracy = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    per_class = f1_score(
        labels, preds, average=None, labels=list(range(len(classes))), zero_division=0
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": {classes[i]: float(per_class[i]) for i in range(len(classes))},
        "n_samples": len(labels),
        "paths": paths,
        "labels": labels,
        "preds": preds,
        "probs": probs,
    }


def run_adabn_one(
    checkpoint_path: Path,
    pair: TransferPair,
    *,
    config_path: str = "config.yaml",
    image_roots: dict[str, str] | None = None,
    save_predictions: bool = True,
) -> dict[str, Any]:
    device = get_device()
    config = load_config(config_path)
    results_root = Path(config["results_root"])
    manifest_path = results_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    ckpt = load_checkpoint(checkpoint_path, device)
    model_name = str(ckpt["model_name"])
    classes = list(pair.classes)
    ckpt_classes = list(ckpt["classes"])
    if classes != ckpt_classes:
        raise ValueError(
            f"Class mismatch: pair={classes} checkpoint={ckpt_classes} path={checkpoint_path}"
        )
    class_to_index = dict(ckpt["class_to_index"])
    index_to_class = {idx: name for name, idx in class_to_index.items()}
    train_seed = int(ckpt.get("train_seed", ckpt.get("seed", config.get("seed", 42))))
    split_seed = int(ckpt.get("split_seed", config.get("split_seed", 42)))
    set_seed(train_seed)

    # Target-domain loaders: train split for AdaBN, test split for evaluation.
    # Eval transform on the adapt loader (no augmentation during BN calibration).
    adapt_loader, _val, eval_loader, meta = make_loaders(
        train_datasets=[pair.test_dataset],
        eval_dataset=pair.test_dataset,
        classes=classes,
        image_size=int(config.get("image_size", 224)),
        batch_size=int(config.get("batch_size", 32)),
        aug_pipeline=default_eval_transform(int(config.get("image_size", 224))),
        return_class_weights=False,
        manifest_path=str(manifest_path),
        image_roots=image_roots,
        train_seed=train_seed,
    )
    if meta["class_to_index"] != class_to_index:
        raise AssertionError(
            f"class_to_index mismatch checkpoint={class_to_index} loader={meta['class_to_index']}"
        )

    model = build_model(model_name, num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    n_bn = count_batch_norm_layers(model)

    baseline = _metrics_from_loader(model, eval_loader, device, classes)

    model_adapted = copy.deepcopy(model)
    adapt_info = adapt_batch_norm(model_adapted, adapt_loader, device)
    adapted = _metrics_from_loader(model_adapted, eval_loader, device, classes)

    pred_path = None
    if save_predictions:
        pred_dir = results_root / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"{model_name}__train-{pair.train_dataset}"
            f"__run-to-{pair.test_dataset}__classes-{'+'.join(classes)}"
            f"__adabn__eval-{pair.test_dataset}__seed{train_seed}"
        )
        pred_path = pred_dir / f"{stem}.csv"
        save_predictions_csv(
            pred_path,
            adapted["paths"],
            adapted["labels"],
            adapted["preds"],
            adapted["probs"],
            index_to_class,
            dataset=pair.test_dataset,
        )

    row = {
        "train_dataset": pair.train_dataset,
        "test_dataset": pair.test_dataset,
        "model": model_name,
        "classes": pair.class_string,
        "seed": train_seed,
        "split_seed": split_seed,
        "augmentation": "default",
        "method": "adabn",
        "n_bn_layers": n_bn,
        "n_adapt_images": adapt_info["n_adapt_images"],
        "baseline_accuracy": baseline["accuracy"],
        "baseline_macro_f1": baseline["macro_f1"],
        "adabn_accuracy": adapted["accuracy"],
        "adabn_macro_f1": adapted["macro_f1"],
        "delta_macro_f1": adapted["macro_f1"] - baseline["macro_f1"],
        "n_samples": adapted["n_samples"],
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(pred_path) if pred_path else "",
    }
    return row


def run_adabn(
    *,
    config_path: str = "config.yaml",
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    image_roots: dict[str, str] | None = None,
    dry_run: bool = False,
    skip_missing: bool = True,
) -> pd.DataFrame:
    config = load_config(config_path)
    results_root = Path(config["results_root"])
    out_dir = results_root / "adabn"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "adabn_results.csv"

    models = models or list(DEFAULT_MODELS)
    seeds = seeds or list(DEFAULT_SEEDS)
    rows: list[dict[str, Any]] = []
    planned = 0
    missing = 0

    for seed in seeds:
        for pair in TRANSFER_PAIRS:
            for model_name in models:
                planned += 1
                ckpt = _checkpoint_for_pair(results_root, model_name, pair, seed)
                if not ckpt.exists():
                    missing += 1
                    msg = f"MISSING checkpoint: {ckpt}"
                    if skip_missing:
                        print(msg)
                        continue
                    raise FileNotFoundError(msg)
                if dry_run:
                    print(f"[dry_run] {model_name} | {pair.key} | seed={seed} | {ckpt}")
                    continue
                print(f"\n=== AdaBN {model_name} | {pair.key} | seed={seed} ===")
                row = run_adabn_one(
                    ckpt,
                    pair,
                    config_path=config_path,
                    image_roots=image_roots,
                )
                print(
                    f"baseline_f1={row['baseline_macro_f1']:.4f}  "
                    f"adabn_f1={row['adabn_macro_f1']:.4f}  "
                    f"delta={row['delta_macro_f1']:+.4f}  "
                    f"adapt_n={row['n_adapt_images']}"
                )
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    if dry_run:
        print(
            f"Dry run: {planned} jobs planned, {missing} checkpoints missing, "
            f"{planned - missing} would execute."
        )
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(out_csv, index=False)
        print(f"\nWrote {out_csv} ({len(df)} rows)")
        print(df.groupby("model")["delta_macro_f1"].agg(["mean", "std", "count"]))
    else:
        print("No AdaBN rows written (no checkpoints found).")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="AdaBN cross-dataset evaluation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--image_roots",
        nargs="*",
        default=None,
        help="dataset=/abs/path remaps (usually unused if Cell 2 rewrote manifest)",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--require_checkpoints",
        action="store_true",
        help="Fail if any planned checkpoint is missing",
    )
    args = parser.parse_args()
    run_adabn(
        config_path=args.config,
        models=args.models,
        seeds=args.seeds,
        image_roots=_parse_image_roots(args.image_roots),
        dry_run=args.dry_run,
        skip_missing=not args.require_checkpoints,
    )


if __name__ == "__main__":
    main()
