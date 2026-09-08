"""Run the Week 13 SAM-masked transfer matrix for one training seed.

This file is intentionally self-contained so it can be copied to a Kaggle
working directory and run without importing the Research_CV package.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


# Edit these two constants after inspecting `ls /kaggle/input/`.
MANIFEST_PATH = Path("/kaggle/input/<dataset>/frozen_results/manifest.csv")
MASKED_ROOT = Path("/kaggle/input/<dataset>/data/masked/sam_leaf")

CONDITION = "sam_leaf"
SCHEMA_PATH = Path("frozen_results_v2/crossdataset_matrix_masked_sam_leaf.csv")
OUTPUT_DIR = Path("/kaggle/working")

EXPECTED_MANIFEST_ROWS = 5419
IMAGE_SIZE = 224
BATCH_SIZE = 32
HEAD_EPOCHS = 3
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3
SCHEDULER_PATIENCE = 5
EARLY_STOPPING_PATIENCE = 7

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODELS = ["resnet50", "efficientnet_b0", "mobilenetv2_100"]


# Source of truth: TRANSFER_PAIRS in run_crossdataset.py. Use it when the
# repository module is importable; otherwise keep this exact ordered copy so
# the Kaggle cell remains self-contained.
if importlib.util.find_spec("run_crossdataset") is not None:
    from run_crossdataset import TRANSFER_PAIRS as _SOURCE_TRANSFER_PAIRS

    TRANSFER_PAIRS = [
        (pair.train_dataset, pair.test_dataset, tuple(pair.classes))
        for pair in _SOURCE_TRANSFER_PAIRS
    ]
else:
    TRANSFER_PAIRS = [
        ("riceleafbd", "dhan_shomadhan", ("brown_spot", "tungro")),
        ("dhan_shomadhan", "riceleafbd", ("brown_spot", "tungro")),
        ("riceleafbd", "brri_rice_disease_pest", ("healthy", "tungro")),
        ("brri_rice_disease_pest", "riceleafbd", ("healthy", "tungro")),
        (
            "dhan_shomadhan",
            "brri_rice_disease_pest",
            ("rice_blast", "scald", "tungro"),
        ),
        (
            "brri_rice_disease_pest",
            "dhan_shomadhan",
            ("rice_blast", "scald", "tungro"),
        ),
    ]


class ManifestImageDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(
        self,
        rows: pd.DataFrame,
        class_to_index: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.class_to_index = class_to_index
        self.transform = transform
        self.image_paths = self.rows["image_path"].astype(str).tolist()
        self.labels = [
            self.class_to_index[label]
            for label in self.rows["mapped_class"].astype(str).tolist()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
        image_tensor = self.transform(rgb_image)
        return image_tensor, self.labels[index], image_path


def set_seed(seed: int, print_values: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if print_values:
        print(f"python random.seed({seed})")
        print(f"numpy.random.seed({seed})")
        print(f"torch.manual_seed({seed})")
        print(f"torch.cuda.manual_seed_all({seed})")


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def compute_class_weights(
    train_rows: pd.DataFrame, class_to_index: dict[str, int]
) -> torch.Tensor:
    counts = train_rows["mapped_class"].value_counts()
    total = float(len(train_rows))
    weights = np.zeros(len(class_to_index), dtype=np.float32)
    for class_name, class_index in class_to_index.items():
        class_count = float(counts.get(class_name, 0.0))
        if class_count <= 0:
            raise ValueError(
                f"Class {class_name!r} has zero source-train samples; "
                "cannot compute class weight."
            )
        weights[class_index] = total / (len(class_to_index) * class_count)
    return torch.tensor(weights, dtype=torch.float32)


def make_loaders(
    manifest: pd.DataFrame,
    source_dataset: str,
    target_dataset: str,
    classes: list[str],
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}

    train_rows = manifest[
        (manifest["dataset"] == source_dataset)
        & (manifest["split"] == "train")
        & (manifest["mapped_class"].isin(classes))
    ].copy()
    val_rows = manifest[
        (manifest["dataset"] == source_dataset)
        & (manifest["split"] == "val")
        & (manifest["mapped_class"].isin(classes))
    ].copy()
    target_test_rows = manifest[
        (manifest["dataset"] == target_dataset)
        & (manifest["split"] == "test")
        & (manifest["mapped_class"].isin(classes))
    ].copy()

    if train_rows.empty or val_rows.empty or target_test_rows.empty:
        raise ValueError(
            "One or more loaders have no rows after filtering: "
            f"train={len(train_rows)}, val={len(val_rows)}, "
            f"target_test={len(target_test_rows)}"
        )

    train_paths = set(train_rows["image_path"].tolist())
    target_paths = set(target_test_rows["image_path"].tolist())
    overlap = train_paths.intersection(target_paths)
    if overlap:
        raise AssertionError(
            f"Source-train/target-test leakage detected: {next(iter(overlap))}"
        )

    train_classes = sorted(train_rows["mapped_class"].astype(str).unique().tolist())
    target_classes = sorted(
        target_test_rows["mapped_class"].astype(str).unique().tolist()
    )
    if train_classes != target_classes:
        raise AssertionError(
            "Class set mismatch between source train and target test: "
            f"train={train_classes}, target={target_classes}"
        )

    train_loader = DataLoader(
        ManifestImageDataset(train_rows, class_to_index, train_transform()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
    val_loader = DataLoader(
        ManifestImageDataset(val_rows, class_to_index, eval_transform()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
    )
    target_test_loader = DataLoader(
        ManifestImageDataset(target_test_rows, class_to_index, eval_transform()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader, target_test_loader, {
        "class_to_index": class_to_index,
        "class_weights": compute_class_weights(train_rows, class_to_index),
    }


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if any(key in name for key in ("classifier", "fc", "head")):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = trainable


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * images.size(0)
        all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = float(
        (torch.tensor(all_predictions) == torch.tensor(all_labels))
        .float()
        .mean()
        .item()
    )
    macro_f1 = float(
        f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    )
    return {
        "loss": total_loss / max(len(all_labels), 1),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0
    use_amp = scaler is not None

    for images, labels, _ in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", enabled=True):
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


def safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", value).strip("-")


def training_stem(
    model_name: str,
    source_dataset: str,
    seed: int,
    run_tag: str,
) -> str:
    return (
        f"{model_name}__train-{source_dataset}__run-{safe_tag(run_tag)}"
        f"__seed{seed}"
    )


def train_model(
    model_name: str,
    source_dataset: str,
    target_dataset: str,
    classes: list[str],
    manifest: pd.DataFrame,
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
) -> Path:
    set_seed(seed)
    train_loader, val_loader, _, meta = make_loaders(
        manifest,
        source_dataset,
        target_dataset,
        classes,
        seed,
    )

    model = build_model(model_name, len(classes), pretrained=True).to(device)
    class_weights = meta["class_weights"]
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    best_macro_f1 = -1.0
    epochs_without_improve = 0
    run_tag = (
        f"to-{target_dataset}__classes-{'+'.join(classes)}"
        f"__mask-{CONDITION}"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == 1:
            set_backbone_trainable(model, trainable=False)
            optimizer = torch.optim.Adam(
                filter(lambda parameter: parameter.requires_grad, model.parameters()),
                lr=LEARNING_RATE,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=SCHEDULER_PATIENCE,
            )
        elif epoch == HEAD_EPOCHS + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=SCHEDULER_PATIENCE,
            )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
        )
        val_metrics = evaluate_loader(model, val_loader, device)
        scheduler.step(val_metrics["macro_f1"])
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            epochs_without_improve = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_name": model_name,
                    "train_datasets": [source_dataset],
                    "eval_dataset": source_dataset,
                    "classes": classes,
                    "class_to_index": meta["class_to_index"],
                    "seed": seed,
                    "train_seed": seed,
                    "split_seed": 42,
                    "run_tag": safe_tag(run_tag),
                    "augmentation": "default",
                    "best_val_macro_f1": best_macro_f1,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={EARLY_STOPPING_PATIENCE})."
                )
                break

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Expected best checkpoint missing: {checkpoint_path}")
    return checkpoint_path


def evaluate_checkpoint(
    checkpoint_path: Path,
    target_loader: DataLoader,
    model_name: str,
    class_count: int,
    device: torch.device,
) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    required = {"model_name", "classes", "model_state_dict", "class_to_index"}
    missing = required - set(checkpoint.keys())
    if missing:
        raise ValueError(
            f"Checkpoint {checkpoint_path} missing keys: {sorted(missing)}"
        )
    if checkpoint["model_name"] != model_name:
        raise ValueError(
            f"Checkpoint model mismatch: expected {model_name}, "
            f"got {checkpoint['model_name']}"
        )

    model = build_model(model_name, class_count, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for images, batch_labels, _ in tqdm(target_loader, leave=False):
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.tolist())

    return {
        "accuracy": float(
            (torch.tensor(predictions) == torch.tensor(labels)).float().mean().item()
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "n_samples": float(len(labels)),
    }


def load_and_resolve_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Expected manifest path: {MANIFEST_PATH}")
    if not MASKED_ROOT.is_dir():
        raise FileNotFoundError(f"Expected masked root path: {MASKED_ROOT}")

    manifest = pd.read_csv(MANIFEST_PATH)
    required = {"image_path", "dataset", "mapped_class", "split", "is_duplicate"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Manifest {MANIFEST_PATH} missing columns: {sorted(missing)}"
        )
    manifest = manifest[manifest["is_duplicate"] == False].reset_index(drop=True)
    assert len(manifest) == EXPECTED_MANIFEST_ROWS, (
        f"Expected {EXPECTED_MANIFEST_ROWS} non-duplicate rows, "
        f"got {len(manifest)} from {MANIFEST_PATH}"
    )

    resolved_root = MASKED_ROOT.resolve()
    resolved_paths: list[str] = []
    missing_paths: list[str] = []
    for image_path in manifest["image_path"].astype(str):
        normalized = image_path.replace("\\", "/")
        prefix = "data/raw/"
        if prefix not in normalized:
            raise ValueError(
                f"Manifest image_path is not under data/raw/: {image_path!r}"
            )
        relative_suffix = normalized.split(prefix, 1)[1]
        resolved_path = (MASKED_ROOT / Path(*relative_suffix.split("/"))).resolve()
        if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
            raise ValueError(
                f"Resolved image escapes masked root: {image_path!r} -> "
                f"{resolved_path}"
            )
        resolved_paths.append(str(resolved_path))
        if not resolved_path.is_file():
            missing_paths.append(str(resolved_path))

    resolvable_count = len(resolved_paths) - len(missing_paths)
    if resolvable_count != EXPECTED_MANIFEST_ROWS:
        difference = resolvable_count - EXPECTED_MANIFEST_ROWS
        print(
            f"Resolvable masked paths: {resolvable_count}; "
            f"difference from expected: {difference}"
        )
        print("Missing paths (up to 10):")
        for missing_path in missing_paths[:10]:
            print(f"  {missing_path}")
        raise FileNotFoundError(
            f"Masked path pre-flight failed: {len(missing_paths)} missing "
            f"paths under {MASKED_ROOT}"
        )

    resolved_manifest = manifest.copy()
    resolved_manifest["image_path"] = resolved_paths
    print(f"Loaded manifest: {len(resolved_manifest)} non-duplicate rows")
    print(f"Resolved masked images: {resolvable_count} under {MASKED_ROOT}")
    return resolved_manifest


def validate_pair_classes(manifest: pd.DataFrame) -> None:
    for source_dataset, target_dataset, classes in TRANSFER_PAIRS:
        expected = set(classes)
        for dataset in (source_dataset, target_dataset):
            available = set(
                manifest.loc[
                    manifest["dataset"] == dataset, "mapped_class"
                ].astype(str)
            )
            missing = expected - available
            if missing:
                raise ValueError(
                    f"{source_dataset}:{target_dataset} requires classes missing "
                    f"from {dataset}: {sorted(missing)}"
                )


def read_schema_columns() -> list[str]:
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Expected schema CSV path: {SCHEMA_PATH}")
    columns = pd.read_csv(SCHEMA_PATH, nrows=0).columns.tolist()
    if not columns:
        raise ValueError(f"Schema CSV has no header columns: {SCHEMA_PATH}")
    print(f"Output schema read from: {SCHEMA_PATH}")
    print(f"Output columns: {columns}")
    return columns


def combination_key(row: pd.Series | dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["train_dataset"]),
        str(row["test_dataset"]),
        str(row["model"]),
    )


def load_partial(
    partial_path: Path,
    schema_columns: list[str],
    expected_keys: set[tuple[str, str, str]],
    seed: int,
) -> pd.DataFrame:
    if not partial_path.is_file():
        print("Completed combinations being skipped: 0")
        return pd.DataFrame(columns=schema_columns)

    partial = pd.read_csv(partial_path)
    if partial.columns.tolist() != schema_columns:
        raise ValueError(
            f"Partial CSV columns do not match schema: {partial_path}"
        )
    if partial["seed"].astype(int).ne(seed).any():
        raise ValueError(f"Partial CSV contains a different seed: {partial_path}")
    if partial["condition"].astype(str).ne(CONDITION).any():
        raise ValueError(f"Partial CSV contains a different condition: {partial_path}")

    keys = [combination_key(row) for _, row in partial.iterrows()]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Partial CSV contains duplicate combinations: {partial_path}")
    unknown = set(keys) - expected_keys
    if unknown:
        raise ValueError(
            f"Partial CSV contains unexpected combinations: {sorted(unknown)}"
        )
    print(f"Completed combinations being skipped: {len(keys)}")
    return partial


def append_partial(
    partial_path: Path,
    row: dict[str, Any],
    schema_columns: list[str],
) -> None:
    row_columns = list(row.keys())
    if row_columns != schema_columns:
        raise ValueError(
            f"Result row columns do not match frozen schema: "
            f"row={row_columns}, schema={schema_columns}"
        )
    if partial_path.is_file():
        existing = pd.read_csv(partial_path)
        if existing.columns.tolist() != schema_columns:
            raise ValueError(
                f"Partial CSV columns changed before append: {partial_path}"
            )
    pd.DataFrame([row], columns=schema_columns).to_csv(
        partial_path,
        mode="a",
        header=not partial_path.is_file(),
        index=False,
    )


def ensure_partial_exists(
    partial_path: Path,
    schema_columns: list[str],
) -> None:
    if partial_path.is_file():
        return
    pd.DataFrame(columns=schema_columns).to_csv(partial_path, index=False)


def main(seed: int) -> None:
    set_seed(seed, print_values=True)
    print(f"torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU name: unavailable (CUDA is not available)")
    print(f"Manifest root: {MANIFEST_PATH.resolve()}")
    print(f"Masked root: {MASKED_ROOT.resolve()}")

    manifest = load_and_resolve_manifest()
    validate_pair_classes(manifest)
    schema_columns = read_schema_columns()

    expected_keys = {
        (source_dataset, target_dataset, model_name)
        for source_dataset, target_dataset, _ in TRANSFER_PAIRS
        for model_name in MODELS
    }
    partial_path = OUTPUT_DIR / f"partial_seed{seed}.csv"
    output_path = OUTPUT_DIR / (
        f"crossdataset_matrix_masked_{CONDITION}_seed{seed}.csv"
    )
    partial = load_partial(partial_path, schema_columns, expected_keys, seed)
    completed_keys = {
        combination_key(row) for _, row in partial.iterrows()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = OUTPUT_DIR / "results_masked" / CONDITION / "checkpoints"

    for source_dataset, target_dataset, class_tuple in TRANSFER_PAIRS:
        classes = list(class_tuple)
        for model_name in MODELS:
            key = (source_dataset, target_dataset, model_name)
            if key in completed_keys:
                print(
                    f"SKIP completed: {model_name} | "
                    f"{source_dataset}->{target_dataset} | seed={seed}"
                )
                continue

            run_tag = (
                f"to-{target_dataset}__classes-{'+'.join(classes)}"
                f"__mask-{CONDITION}"
            )
            stem = training_stem(model_name, source_dataset, seed, run_tag)
            checkpoint_path = checkpoint_dir / f"{stem}.pth"
            print(
                f"\n=== TRAIN {model_name}: {source_dataset} -> "
                f"{target_dataset} (seed={seed}, {CONDITION}) ==="
            )
            if checkpoint_path.is_file():
                print(f"REUSE checkpoint: {checkpoint_path}")
            else:
                train_model(
                    model_name,
                    source_dataset,
                    target_dataset,
                    classes,
                    manifest,
                    seed,
                    device,
                    checkpoint_path,
                )

            set_seed(seed)
            _, _, target_loader, _ = make_loaders(
                manifest,
                source_dataset,
                target_dataset,
                classes,
                seed,
            )
            cross = evaluate_checkpoint(
                checkpoint_path,
                target_loader,
                model_name,
                len(classes),
                device,
            )
            row: dict[str, Any] = {
                "train_dataset": source_dataset,
                "test_dataset": target_dataset,
                "model": model_name,
                "classes": "|".join(classes),
                "seed": seed,
                "condition": CONDITION,
                "accuracy": cross["accuracy"],
                "macro_f1": cross["macro_f1"],
                "n_samples": int(cross["n_samples"]),
                "checkpoint_path": str(checkpoint_path),
            }
            if list(row.keys()) != schema_columns:
                raise ValueError(
                    f"Frozen schema mismatch: row={list(row.keys())}, "
                    f"schema={schema_columns}"
                )
            append_partial(partial_path, row, schema_columns)
            completed_keys.add(key)
            print(
                f"RECORDED: {model_name} | {source_dataset}->{target_dataset} | "
                f"macro_f1={cross['macro_f1']:.4f} | n={int(cross['n_samples'])}"
            )

    ensure_partial_exists(partial_path, schema_columns)
    final = pd.read_csv(partial_path)
    if len(final) != len(expected_keys):
        final_keys = {combination_key(row) for _, row in final.iterrows()}
        missing = sorted(expected_keys - final_keys)
        print(f"Missing combinations ({len(missing)}):")
        for key in missing:
            print(f"  {key[0]} -> {key[1]} | {key[2]}")
        raise AssertionError(
            f"Expected {len(expected_keys)} rows before writing output, "
            f"got {len(final)}"
        )
    if final.columns.tolist() != schema_columns:
        raise ValueError("Partial result columns changed before final write")
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing separate output: {output_path}"
        )

    final.to_csv(output_path, index=False)
    print(f"Output shape: {final.shape}")
    print(final.head())
    print(f"Output file written: {output_path}")
    print(f"output_path: {output_path}")
    print(f"row_count: {len(final)}")
    print(f"mean_macro_f1: {final['macro_f1'].mean()}")
    print(f"median_macro_f1: {final['macro_f1'].median()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments.seed)
