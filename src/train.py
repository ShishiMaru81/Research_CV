from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.data_loader import make_loaders
from src.utils import get_device, load_config, set_seed

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast


def discover_classes(manifest_path: Path, datasets: list[str]) -> list[str]:
    manifest = pd.read_csv(manifest_path)
    classes = (
        manifest.loc[manifest["dataset"].isin(datasets), "mapped_class"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return sorted(classes)


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if any(key in name for key in ("classifier", "fc", "head")):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


@torch.no_grad()
def evaluate_loader(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = float((torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item())
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    avg_loss = total_loss / max(len(all_labels), 1)
    return {"loss": avg_loss, "accuracy": accuracy, "macro_f1": macro_f1}


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler: GradScaler | None,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0

    for images, labels, _ in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            try:
                amp_ctx = autocast("cuda")
            except TypeError:  # older torch
                amp_ctx = autocast()
            with amp_ctx:
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        n_samples += images.size(0)

    return running_loss / max(n_samples, 1)


def train(
    model_name: str,
    train_datasets: list[str],
    classes: list[str] | None,
    config: dict[str, Any],
    seed: int,
    eval_dataset: str | None = None,
    use_amp: bool = True,
    verify_images: bool = False,
    path_remap: tuple[str, str] | None = None,
    image_root: str | None = None,
    image_roots: dict[str, str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    set_seed(seed)
    device = get_device()

    results_root = Path(config["results_root"])
    manifest_path = results_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Copy artifacts/manifest.csv to results/ or run python -m src.build_manifest."
        )

    if classes is None:
        classes = discover_classes(manifest_path, train_datasets)
    if not classes:
        raise ValueError(f"No classes found for datasets={train_datasets}")

    eval_ds = eval_dataset or train_datasets[0]
    image_size = int(config.get("image_size", 224))
    batch_size = int(config.get("batch_size", 32))
    num_epochs = int(config.get("num_epochs", 30))
    learning_rate = float(config.get("learning_rate", 1e-3))
    patience = int(config.get("early_stopping_patience", 7))
    head_epochs = int(config.get("head_epochs", 3))

    train_loader, val_loader, _, meta = make_loaders(
        train_datasets=train_datasets,
        eval_dataset=eval_ds,
        classes=classes,
        image_size=image_size,
        batch_size=batch_size,
        return_class_weights=True,
        manifest_path=str(manifest_path),
        verify_images=verify_images,
        path_remap=path_remap,
        image_root=image_root,
        image_roots=image_roots,
    )

    model = build_model(model_name, num_classes=len(classes), pretrained=True).to(device)
    class_weights = meta["class_weights"]
    if class_weights is None:
        raise ValueError("Class weights were not returned by make_loaders.")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    use_amp = bool(use_amp and device.type == "cuda")
    try:
        scaler = GradScaler("cuda", enabled=use_amp)
    except TypeError:  # older torch
        scaler = GradScaler(enabled=use_amp)

    history: list[dict[str, float]] = []
    best_macro_f1 = -1.0
    epochs_without_improve = 0

    train_tag = "+".join(train_datasets)
    ckpt_dir = results_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"{model_name}__train-{train_tag}__seed{seed}.pth"
    log_path = results_root / f"{model_name}__train-{train_tag}__seed{seed}.json"

    for epoch in range(1, num_epochs + 1):
        if epoch == 1:
            set_backbone_trainable(model, trainable=False)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )
        elif epoch == head_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate * 0.1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_amp, scaler
        )
        val_metrics = evaluate_loader(model, val_loader, device)
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            epochs_without_improve = 0
            torch.save(
                {
                    "model_name": model_name,
                    "train_datasets": train_datasets,
                    "eval_dataset": eval_ds,
                    "classes": classes,
                    "class_to_index": meta["class_to_index"],
                    "seed": seed,
                    "best_val_macro_f1": best_macro_f1,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    log_dict: dict[str, Any] = {
        "model_name": model_name,
        "train_datasets": train_datasets,
        "eval_dataset": eval_ds,
        "classes": classes,
        "class_to_index": meta["class_to_index"],
        "seed": seed,
        "best_val_macro_f1": best_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }
    log_path.write_text(json.dumps(log_dict, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training log: {log_path}")
    return checkpoint_path, log_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a timm model on rice leaf datasets.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None, help="timm model name, e.g. mobilenetv2_100")
    parser.add_argument(
        "--train_datasets",
        nargs="+",
        default=None,
        help="One or more dataset keys from config.",
    )
    parser.add_argument(
        "--eval_dataset",
        default=None,
        help="Eval dataset key (defaults to first train dataset).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional class subset. Default: all classes.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument(
        "--verify_images",
        action="store_true",
        help="Scan all images at startup (slow).",
    )
    parser.add_argument(
        "--path_remap",
        nargs=2,
        metavar=("OLD", "NEW"),
        default=None,
        help="Rewrite manifest image_path prefix, e.g. 'data/raw' '/kaggle/input'.",
    )
    parser.add_argument(
        "--image_root",
        default=None,
        help=(
            "Rebuild paths as {image_root}/{original_class}/{filename}. "
            "Preferred on Kaggle, e.g. /kaggle/input/riceleafbd"
        ),
    )
    parser.add_argument(
        "--image_roots",
        nargs="*",
        default=None,
        help="Per-dataset roots as dataset=/abs/path (can pass multiple).",
    )
    return parser.parse_args()


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


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model or str(cfg.get("model_name", "mobilenetv2_100"))
    train_datasets = args.train_datasets or ["riceleafbd"]
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    path_remap = tuple(args.path_remap) if args.path_remap else None

    train(
        model_name=model_name,
        train_datasets=train_datasets,
        classes=args.classes,
        config=cfg,
        seed=seed,
        eval_dataset=args.eval_dataset,
        use_amp=not args.no_amp,
        verify_images=args.verify_images,
        path_remap=path_remap,
        image_root=args.image_root,
        image_roots=_parse_image_roots(args.image_roots),
    )
