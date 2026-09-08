"""Week 13 SAM-masked transfer matrix at train seed 2024 (Kaggle cell).

Self-contained. Paste into a Kaggle notebook after cloning Research_CV and
setting MANIFEST_PATH / MASKED_ROOT to the actual Input mounts.

Seed is fixed at 2024 so results pair against the existing raw baseline in
frozen_results_v2/transfer_all_seeds.csv (seeds 7, 42, 2024 only). Protocol
mirrors run_transfer_masked.py / Week 13; the only intentional change is the
seed.

Reads notes/data_rules.md: originals only, frozen splits, no silent fallbacks.
Never writes to frozen_results/ or frozen_results_v2/.
"""

from __future__ import annotations

import importlib.util
import json
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

SEED = 2024
CONDITION = "sam_leaf"
OUTPUT_DIR = Path("/kaggle/working")
PARTIAL_PATH = OUTPUT_DIR / "partial_seed2024.csv"
OUTPUT_PATH = OUTPUT_DIR / "crossdataset_matrix_masked_sam_leaf_seed2024.csv"
ENV_PATH = OUTPUT_DIR / "env_seed2024.json"

# Seed-42 schema source (read header at runtime; never hardcode names).
SCHEMA_CANDIDATES = [
    Path("frozen_results_v2/crossdataset_matrix_masked_sam_leaf.csv"),
    Path("/kaggle/working/Research_CV/frozen_results_v2/crossdataset_matrix_masked_sam_leaf.csv"),
]

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

# Source of truth: DEFAULT_MODELS in run_crossdataset.py (Week 5 / Week 13).
MODELS = ["mobilenetv2_100", "efficientnet_b0", "resnet50"]


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


def resolve_schema_path() -> Path:
    for candidate in SCHEMA_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Expected seed-42 schema CSV at one of: "
        + ", ".join(str(path) for path in SCHEMA_CANDIDATES)
    )


def read_schema_columns() -> list[str]:
    schema_path = resolve_schema_path()
    columns = pd.read_csv(schema_path, nrows=0).columns.tolist()
    if not columns:
        raise ValueError(f"Schema CSV has no header columns: {schema_path}")
    print(f"Output schema read from: {schema_path}")
    print(f"Output columns: {columns}")
    return columns


def write_env_json(device: torch.device) -> None:
    env = {
        "seed": SEED,
        "condition": CONDITION,
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "device": str(device),
        "manifest_path": str(MANIFEST_PATH),
        "masked_root": str(MASKED_ROOT),
    }
    ENV_PATH.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote environment capture: {ENV_PATH}")


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


def build_result_row(
    schema_columns: list[str],
    source_dataset: str,
    target_dataset: str,
    model_name: str,
    classes: list[str],
    cross: dict[str, float],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Build a result row keyed exactly to the frozen schema header."""
    values: dict[str, Any] = {
        "train_dataset": source_dataset,
        "test_dataset": target_dataset,
        "model": model_name,
        "classes": "|".join(classes),
        "seed": SEED,
        "condition": CONDITION,
        "accuracy": cross["accuracy"],
        "macro_f1": cross["macro_f1"],
        "n_samples": int(cross["n_samples"]),
        "checkpoint_path": str(checkpoint_path),
    }
    missing = [column for column in schema_columns if column not in values]
    if missing:
        raise KeyError(
            f"Schema requires columns not produced by this runner: {missing}"
        )
    extra = [column for column in values if column not in schema_columns]
    if extra:
        raise KeyError(f"Runner produced columns absent from schema: {extra}")
    return {column: values[column] for column in schema_columns}


def main() -> None:
    if SEED != 2024:
        raise RuntimeError(f"SEED must remain 2024; got {SEED}")

    set_seed(SEED, print_values=True)
    print(f"SEED: {SEED}")
    print(f"CONDITION: {CONDITION}")
    print(f"torch version: {torch.__version__}")
    print(f"timm version: {timm.__version__}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("GPU name: unavailable (CUDA is not available)")
        print("CUDA version: None")
    print(f"Manifest path: {MANIFEST_PATH}")
    print(f"Masked root: {MASKED_ROOT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_env_json(device)

    manifest = load_and_resolve_manifest()
    validate_pair_classes(manifest)
    schema_columns = read_schema_columns()

    expected_keys = {
        (source_dataset, target_dataset, model_name)
        for source_dataset, target_dataset, _ in TRANSFER_PAIRS
        for model_name in MODELS
    }
    if len(expected_keys) != 18:
        raise AssertionError(
            f"Expected 18 (pair, model) combinations, got {len(expected_keys)}"
        )

    partial = load_partial(PARTIAL_PATH, schema_columns, expected_keys, SEED)
    completed_keys = {combination_key(row) for _, row in partial.iterrows()}
    checkpoint_dir = OUTPUT_DIR / "results_masked" / CONDITION / "checkpoints"

    for source_dataset, target_dataset, class_tuple in TRANSFER_PAIRS:
        classes = list(class_tuple)
        for model_name in MODELS:
            key = (source_dataset, target_dataset, model_name)
            if key in completed_keys:
                print(
                    f"SKIP completed: {model_name} | "
                    f"{source_dataset}->{target_dataset} | seed={SEED}"
                )
                continue

            run_tag = (
                f"to-{target_dataset}__classes-{'+'.join(classes)}"
                f"__mask-{CONDITION}"
            )
            stem = training_stem(model_name, source_dataset, SEED, run_tag)
            checkpoint_path = checkpoint_dir / f"{stem}.pth"
            print(
                f"\n=== TRAIN {model_name}: {source_dataset} -> "
                f"{target_dataset} (seed={SEED}, {CONDITION}) ==="
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
                    SEED,
                    device,
                    checkpoint_path,
                )

            set_seed(SEED)
            _, _, target_loader, _ = make_loaders(
                manifest,
                source_dataset,
                target_dataset,
                classes,
                SEED,
            )
            cross = evaluate_checkpoint(
                checkpoint_path,
                target_loader,
                model_name,
                len(classes),
                device,
            )
            row = build_result_row(
                schema_columns,
                source_dataset,
                target_dataset,
                model_name,
                classes,
                cross,
                checkpoint_path,
            )
            if list(row.keys()) != schema_columns:
                raise ValueError(
                    f"Frozen schema mismatch: row={list(row.keys())}, "
                    f"schema={schema_columns}"
                )
            append_partial(PARTIAL_PATH, row, schema_columns)
            completed_keys.add(key)
            metric_name = (
                "macro_f1" if "macro_f1" in row else schema_columns[-3]
            )
            print(
                f"RECORDED: {model_name} | {source_dataset}->{target_dataset} | "
                f"{metric_name}={row[metric_name]:.4f} | n={row['n_samples']}"
            )

    ensure_partial_exists(PARTIAL_PATH, schema_columns)
    final = pd.read_csv(PARTIAL_PATH)
    if len(final) != 18:
        final_keys = {combination_key(row) for _, row in final.iterrows()}
        missing = sorted(expected_keys - final_keys)
        print(f"Missing combinations ({len(missing)}):")
        for key in missing:
            print(f"  {key[0]} -> {key[1]} | {key[2]}")
        raise AssertionError(
            f"Expected 18 rows before writing output, got {len(final)}. "
            f"Partial remains at {PARTIAL_PATH}"
        )
    if final.columns.tolist() != schema_columns:
        raise ValueError(
            "Partial result columns do not match schema before final write: "
            f"got={final.columns.tolist()}, expected={schema_columns}"
        )
    if int(final["seed"].nunique()) != 1 or int(final["seed"].iloc[0]) != SEED:
        raise AssertionError(f"Output seed column must be entirely {SEED}")
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT_PATH}"
        )

    final.to_csv(OUTPUT_PATH, index=False)

    # Metric column name comes from schema (seed-42 file uses macro_f1).
    if "macro_f1" in final.columns:
        metric_col = "macro_f1"
    elif "cross_macro_f1" in final.columns:
        metric_col = "cross_macro_f1"
    else:
        raise KeyError(
            f"Schema has neither macro_f1 nor cross_macro_f1: {schema_columns}"
        )

    print(f"Output shape: {final.shape}")
    print(final.head())
    print(f"Output file written: {OUTPUT_PATH}")
    print(OUTPUT_PATH)
    print(len(final))
    print(float(final[metric_col].mean()))
    print(float(final[metric_col].median()))


if __name__ == "__main__":
    main()
