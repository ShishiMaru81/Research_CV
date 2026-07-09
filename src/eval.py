from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

from src.data_loader import make_loaders
from src.train import build_model
from src.utils import get_device, load_config, set_seed


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    required = {"model_name", "classes", "model_state_dict", "class_to_index"}
    missing = required - set(checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")
    return checkpoint


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[str], list[int], list[int]]:
    model.eval()
    paths: list[str] = []
    labels: list[int] = []
    preds: list[int] = []

    for images, batch_labels, batch_paths in tqdm(loader, leave=False):
        images = images.to(device)
        logits = model(images)
        batch_preds = logits.argmax(dim=1).cpu().tolist()
        paths.extend(list(batch_paths))
        labels.extend(batch_labels.tolist())
        preds.extend(batch_preds)

    return paths, labels, preds


def print_sample_predictions(
    paths: list[str],
    labels: list[int],
    preds: list[int],
    index_to_class: dict[int, str],
    n: int = 10,
) -> None:
    print(f"\nSample predictions (n={min(n, len(paths))}):")
    for i in range(min(n, len(paths))):
        true_label = index_to_class[labels[i]]
        pred_label = index_to_class[preds[i]]
        print(f"  {paths[i]} | true={true_label} | pred={pred_label}")


def save_confusion_matrix_png(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate(
    checkpoint_path: str | Path,
    eval_dataset: str,
    classes: list[str] | None = None,
    split: str = "test",
    seed: int = 42,
    config_path: str = "config.yaml",
    sample_n: int = 10,
    verify_images: bool = False,
    path_remap: tuple[str, str] | None = None,
    image_root: str | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    device = get_device()
    config = load_config(config_path)
    results_root = Path(config["results_root"])
    manifest_path = results_root / "manifest.csv"

    ckpt = load_checkpoint(Path(checkpoint_path), device)
    model_name = str(ckpt["model_name"])
    train_datasets = list(ckpt.get("train_datasets", ["unknown"]))
    ckpt_classes = list(ckpt["classes"])
    class_to_index = dict(ckpt["class_to_index"])
    index_to_class = {idx: name for name, idx in class_to_index.items()}

    if classes is None:
        classes = ckpt_classes
    if classes != ckpt_classes:
        raise ValueError(
            "Eval class list must match the checkpoint class list to avoid index swaps. "
            f"checkpoint={ckpt_classes}, requested={classes}"
        )

    # make_loaders always uses split=='test' for eval_loader; for val we filter manually if needed.
    train_loader, val_loader, eval_loader, meta = make_loaders(
        train_datasets=train_datasets if train_datasets != ["unknown"] else [eval_dataset],
        eval_dataset=eval_dataset,
        classes=classes,
        image_size=int(config.get("image_size", 224)),
        batch_size=int(config.get("batch_size", 32)),
        return_class_weights=False,
        manifest_path=str(manifest_path),
        verify_images=verify_images,
        path_remap=path_remap,
        image_root=image_root,
    )

    if meta["class_to_index"] != class_to_index:
        raise AssertionError(
            "Class->index mapping mismatch between checkpoint and loader. "
            f"checkpoint={class_to_index}, loader={meta['class_to_index']}"
        )

    loader = eval_loader if split == "test" else val_loader
    model = build_model(model_name, num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    paths, labels, preds = collect_predictions(model, loader, device)
    accuracy = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    per_class_f1_vals = f1_score(labels, preds, average=None, labels=list(range(len(classes))), zero_division=0)
    per_class_f1 = {classes[i]: float(per_class_f1_vals[i]) for i in range(len(classes))}
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))

    train_tag = "+".join(train_datasets)
    stem = f"{model_name}__train-{train_tag}__eval-{eval_dataset}__seed{seed}"
    metrics_path = results_root / f"{stem}__metrics.json"
    cm_path = results_root / f"{stem}__confusion_matrix.png"

    metrics: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "model_name": model_name,
        "train_datasets": train_datasets,
        "eval_dataset": eval_dataset,
        "split": split,
        "seed": seed,
        "classes": classes,
        "class_to_index": class_to_index,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(labels),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix_png(cm, classes, cm_path, title=f"{model_name} | {eval_dataset}")

    print(f"accuracy={accuracy:.4f} | macro_f1={macro_f1:.4f} | n={len(labels)}")
    print(f"per-class F1: {per_class_f1}")
    print(f"metrics: {metrics_path}")
    print(f"confusion matrix: {cm_path}")
    print_sample_predictions(paths, labels, preds, index_to_class, n=sample_n)

    # Append experiment log if present/creatable.
    log_csv = results_root / "experiment_log.csv"
    row = {
        "date": pd.Timestamp.utcnow().isoformat(),
        "run_id": stem,
        "model": model_name,
        "train_dataset": train_tag,
        "eval_dataset": eval_dataset,
        "seed": seed,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "notes": f"split={split}",
    }
    if log_csv.exists():
        log_df = pd.read_csv(log_csv)
        log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    else:
        log_df = pd.DataFrame([row])
    log_df.to_csv(log_csv, index=False)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained rice leaf disease checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--eval_dataset", required=True, help="Dataset key to evaluate on.")
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample_n", type=int, default=10)
    parser.add_argument("--verify_images", action="store_true")
    parser.add_argument(
        "--path_remap",
        nargs=2,
        metavar=("OLD", "NEW"),
        default=None,
        help="Rewrite manifest image_path prefix for Kaggle mounts.",
    )
    parser.add_argument(
        "--image_root",
        default=None,
        help="Rebuild paths as {image_root}/{original_class}/{filename}.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path_remap = tuple(args.path_remap) if args.path_remap else None
    evaluate(
        checkpoint_path=args.checkpoint,
        eval_dataset=args.eval_dataset,
        classes=args.classes,
        split=args.split,
        seed=args.seed,
        config_path=args.config,
        sample_n=args.sample_n,
        verify_images=args.verify_images,
        path_remap=path_remap,
        image_root=args.image_root,
    )
