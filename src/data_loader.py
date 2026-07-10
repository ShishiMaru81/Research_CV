from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src.utils import load_config


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class LoaderMeta:
    class_to_index: dict[str, int]
    index_to_class: dict[int, str]
    train_size: int
    val_size: int
    eval_size: int
    class_weights: torch.Tensor | None = None


def default_train_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def default_eval_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class ManifestImageDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(
        self,
        rows: pd.DataFrame,
        class_to_index: dict[str, int],
        transform: A.Compose,
        verify_images: bool = True,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.class_to_index = class_to_index
        self.transform = transform
        self.image_paths = self.rows["image_path"].tolist()
        self.labels = [self.class_to_index[label] for label in self.rows["mapped_class"].tolist()]
        if verify_images:
            self._verify_images()

    def _verify_images(self) -> None:
        broken: list[str] = []
        for image_path in self.image_paths:
            image = cv2.imread(image_path)
            if image is None:
                broken.append(image_path)
        if broken:
            preview = ", ".join(broken[:5])
            raise ValueError(f"Found broken images ({len(broken)}). Examples: {preview}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        image_tensor: torch.Tensor = transformed["image"]
        return image_tensor, self.labels[idx], image_path


def _compute_class_weights(train_rows: pd.DataFrame, class_to_index: dict[str, int]) -> torch.Tensor:
    counts = train_rows["mapped_class"].value_counts()
    total = float(len(train_rows))
    weights = np.zeros(len(class_to_index), dtype=np.float32)
    for cls, idx in class_to_index.items():
        cls_count = float(counts.get(cls, 0.0))
        if cls_count <= 0:
            raise ValueError(f"Class '{cls}' has zero train samples; cannot compute class weight.")
        weights[idx] = total / (len(class_to_index) * cls_count)
    return torch.tensor(weights, dtype=torch.float32)


def _assert_no_leakage(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> None:
    train_paths = set(train_rows["image_path"].tolist())
    eval_paths = set(eval_rows["image_path"].tolist())
    overlap = train_paths.intersection(eval_paths)
    if overlap:
        example = next(iter(overlap))
        raise AssertionError(f"Train/eval leakage detected: {example}")


def _assert_identical_mapping(train_classes: list[str], eval_classes: list[str]) -> None:
    if train_classes != eval_classes:
        raise AssertionError(
            f"Class ordering mismatch between train/eval. train={train_classes}, eval={eval_classes}"
        )


def _posix_path(path: str) -> str:
    return str(path).replace("\\", "/")


def _filename_from_manifest_path(path: str) -> str:
    return _posix_path(path).split("/")[-1]


def _relative_under_dataset(path: str, dataset: str, original_class: str) -> str:
    """Return path relative to the dataset root, preserving nested folders."""
    posix = _posix_path(path)
    markers = [f"data/raw/{dataset}/", f"/{dataset}/", f"{dataset}/"]
    for marker in markers:
        if marker in posix:
            return posix.split(marker, 1)[1]
    return f"{original_class}/{_filename_from_manifest_path(posix)}"


def _apply_path_remap(manifest: pd.DataFrame, path_remap: tuple[str, str] | None) -> pd.DataFrame:
    if path_remap is None:
        return manifest
    old, new = path_remap
    remapped = manifest.copy()
    remapped["image_path"] = (
        remapped["image_path"]
        .astype(str)
        .map(_posix_path)
        .str.replace(old.replace("\\", "/"), new.replace("\\", "/"), regex=False)
    )
    return remapped


def _rebuild_paths_from_image_root(manifest: pd.DataFrame, image_root: str | Path) -> pd.DataFrame:
    """Rebuild image_path as {image_root}/{original_class}/{filename}."""
    root = Path(image_root)
    rebuilt = manifest.copy()
    rebuilt["image_path"] = [
        str(root / str(original_class) / _filename_from_manifest_path(path))
        for path, original_class in zip(rebuilt["image_path"], rebuilt["original_class"])
    ]
    return rebuilt


def _rebuild_paths_from_dataset_roots(
    manifest: pd.DataFrame, image_roots: dict[str, str]
) -> pd.DataFrame:
    """Rebuild paths using per-dataset roots, preserving nested relative structure.

    Example:
      data\\raw\\dhan_shomadhan\\White Background\\Brown Spot\\x.jpg
      + root[/dhan] -> {root}/White Background/Brown Spot/x.jpg
    """
    rebuilt = manifest.copy()
    new_paths: list[str] = []
    for _, row in rebuilt.iterrows():
        dataset = str(row["dataset"])
        if dataset not in image_roots:
            new_paths.append(str(row["image_path"]))
            continue
        rel = _relative_under_dataset(
            str(row["image_path"]), dataset, str(row["original_class"])
        )
        new_paths.append(str(Path(image_roots[dataset]) / rel))
    rebuilt["image_path"] = new_paths
    return rebuilt


def resolve_manifest_paths(
    manifest: pd.DataFrame,
    path_remap: tuple[str, str] | None = None,
    image_root: str | None = None,
    image_roots: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Resolve manifest paths for local or Kaggle dataset layouts."""
    if image_roots:
        return _rebuild_paths_from_dataset_roots(manifest, image_roots)
    if image_root:
        return _rebuild_paths_from_image_root(manifest, image_root)
    return _apply_path_remap(manifest, path_remap)


def _assert_sample_paths_exist(manifest: pd.DataFrame, n: int = 5) -> None:
    sample = manifest.head(max(n, 1))
    # Prefer sampling across datasets when available.
    if "dataset" in manifest.columns:
        pieces = []
        for _, group in manifest.groupby("dataset"):
            pieces.append(group.head(max(1, n // max(manifest["dataset"].nunique(), 1))))
        sample = pd.concat(pieces, ignore_index=True)

    missing: list[str] = []
    for path in sample["image_path"].astype(str).tolist():
        if not Path(path).exists():
            missing.append(path)
    if not missing:
        return

    example = missing[0]
    parent = Path(example).parent
    grandparent = parent.parent
    hint_lines = [
        f"Image path does not exist: {example}",
        f"Parent exists={parent.exists()}: {parent}",
        f"Grandparent exists={grandparent.exists()}: {grandparent}",
    ]
    if grandparent.exists():
        children = sorted([p.name for p in grandparent.iterdir()])[:20]
        hint_lines.append(f"Grandparent contents (first 20): {children}")
    if parent.exists():
        children = sorted([p.name for p in parent.iterdir()])[:20]
        hint_lines.append(f"Parent contents (first 20): {children}")
    hint_lines.append(
        "Fix: pass --image_roots dataset=/kaggle/input/... for each dataset, "
        "or rebuild results/manifest.csv with correct absolute paths."
    )
    raise FileNotFoundError("\n".join(hint_lines))


def make_loaders(
    train_datasets: list[str],
    eval_dataset: str,
    classes: list[str],
    image_size: int = 224,
    batch_size: int = 32,
    aug_pipeline: A.Compose | None = None,
    return_class_weights: bool = False,
    manifest_path: str | None = None,
    verify_images: bool = False,
    path_remap: tuple[str, str] | None = None,
    image_root: str | None = None,
    image_roots: dict[str, str] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    config = load_config()
    resolved_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else Path(config["results_root"]) / "manifest.csv"
    )
    manifest = pd.read_csv(resolved_manifest)
    manifest = resolve_manifest_paths(
        manifest,
        path_remap=path_remap,
        image_root=image_root,
        image_roots=image_roots,
    )
    _assert_sample_paths_exist(manifest, n=5)

    required = {"image_path", "dataset", "mapped_class", "split", "is_duplicate"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    canonical_classes = list(classes)
    class_to_index = {class_name: idx for idx, class_name in enumerate(canonical_classes)}
    index_to_class = {idx: class_name for class_name, idx in class_to_index.items()}

    train_rows = manifest[
        manifest["dataset"].isin(train_datasets)
        & (manifest["split"] == "train")
        & (manifest["mapped_class"].isin(canonical_classes))
    ].copy()
    val_rows = manifest[
        manifest["dataset"].isin(train_datasets)
        & (manifest["split"] == "val")
        & (manifest["mapped_class"].isin(canonical_classes))
        & (~manifest["is_duplicate"].astype(bool))
    ].copy()
    eval_rows = manifest[
        (manifest["dataset"] == eval_dataset)
        & (manifest["split"] == "test")
        & (manifest["mapped_class"].isin(canonical_classes))
        & (~manifest["is_duplicate"].astype(bool))
    ].copy()

    if train_rows.empty or val_rows.empty or eval_rows.empty:
        raise ValueError(
            "One or more loaders have no rows after filtering. "
            f"train={len(train_rows)}, val={len(val_rows)}, eval={len(eval_rows)}"
        )

    _assert_no_leakage(train_rows, eval_rows)
    _assert_identical_mapping(
        sorted(train_rows["mapped_class"].unique().tolist()),
        sorted(eval_rows["mapped_class"].unique().tolist()),
    )

    train_transform = aug_pipeline if aug_pipeline is not None else default_train_transform(image_size)
    eval_transform = default_eval_transform(image_size)

    train_dataset = ManifestImageDataset(
        train_rows, class_to_index, train_transform, verify_images=verify_images
    )
    val_dataset = ManifestImageDataset(
        val_rows, class_to_index, eval_transform, verify_images=verify_images
    )
    eval_dataset_obj = ManifestImageDataset(
        eval_rows, class_to_index, eval_transform, verify_images=verify_images
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_dataset_obj, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = _compute_class_weights(train_rows, class_to_index) if return_class_weights else None
    meta = LoaderMeta(
        class_to_index=class_to_index,
        index_to_class=index_to_class,
        train_size=len(train_rows),
        val_size=len(val_rows),
        eval_size=len(eval_rows),
        class_weights=class_weights,
    )
    return train_loader, val_loader, eval_loader, meta.__dict__


def _print_smoke_test(
    train_datasets: list[str],
    eval_dataset: str,
    classes: list[str],
    image_size: int,
    batch_size: int,
) -> None:
    train_loader, _, eval_loader, meta = make_loaders(
        train_datasets=train_datasets,
        eval_dataset=eval_dataset,
        classes=classes,
        image_size=image_size,
        batch_size=batch_size,
        return_class_weights=True,
    )

    train_images, train_labels, _ = next(iter(train_loader))
    _, eval_labels, _ = next(iter(eval_loader))
    print(f"Smoke test train={train_datasets} eval={eval_dataset}")
    print(f"  batch shape: {tuple(train_images.shape)}")
    print(f"  train label set: {sorted(set(train_labels.tolist()))}")
    print(f"  eval label set: {sorted(set(eval_labels.tolist()))}")
    print(f"  class->index mapping: {meta['class_to_index']}")


if __name__ == "__main__":
    cfg = load_config()
    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 32))
    _print_smoke_test(
        train_datasets=["riceleafbd"],
        eval_dataset="riceleafbd",
        classes=["brown_spot", "tungro"],
        image_size=image_size,
        batch_size=batch_size,
    )
    _print_smoke_test(
        train_datasets=["dhan_shomadhan"],
        eval_dataset="riceleafbd",
        classes=["brown_spot", "tungro"],
        image_size=image_size,
        batch_size=batch_size,
    )
