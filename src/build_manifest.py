from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.utils import load_config, set_seed


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Ambiguous / out-of-scope raw folders that must not enter the study.
# BRRI "Rice" is documented as "general rice images" and is not one of the
# seven leaf-disease/pest classes used in the peer-reviewed dataset paper.
EXCLUDED_RAW_CLASSES = {
    "brri_rice_disease_pest": {"Rice"},
}

CANONICAL_LABEL_MAP = {
    "healthy": ["Healthy", "Healthy Rice Leaf", "Healthy Leaf"],
    "bacterial_leaf_blight": ["Bacterial Leaf Blight", "BLB"],
    "brown_spot": ["Brown Spot", "Browon Spot"],
    "tungro": [
        "Tungro",
        "Tungro Virus",
        "Tungro Disease",
        "Rice Turngo",
        "Rice Tungro",
        "Rice Turgro",
    ],
    "rice_blast": ["Rice Blast", "Leaf Blast", "Blast"],
    "scald": ["Leaf Scald", "Leaf Scaled", "Scald"],
    "sheath_blight": ["Sheath Blight", "Steath Blight", "Stealth Blight", "Shath Blight"],
    "leaf_folder": ["Leaf-folder", "Leaffolder", "Leaf Folder", "Leaf-folder Injury", "Rice Leaffolder"],
    "insect": ["Insect Infestation", "Insect Damage", "Insect"],
    "stripes": ["Rice Stripes", "Leaf Stripes"],
}


def normalize_label(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"[-_]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def build_variant_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, variants in CANONICAL_LABEL_MAP.items():
        lookup[normalize_label(canonical)] = canonical
        for variant in variants:
            lookup[normalize_label(variant)] = canonical
    return lookup


def is_augmented_path(path: Path) -> bool:
    path_lower = path.as_posix().lower()
    stem_lower = path.stem.lower()
    aug_folder_markers = ["/augmented/", "/aug/"]
    aug_name_markers = ["aug", "rot", "flip", "scale", "bright"]
    if any(marker in path_lower for marker in aug_folder_markers):
        return True
    return any(marker in stem_lower for marker in aug_name_markers)


def infer_background(dataset: str, image_path: Path) -> str:
    if dataset != "dhan_shomadhan":
        return ""
    path_lower = image_path.as_posix().lower()
    if "field background" in path_lower or "/field/" in path_lower:
        return "field"
    if "white background" in path_lower or "/white/" in path_lower:
        return "white"
    return ""


def stratified_split_indices(count: int, seed: int) -> list[str]:
    if count <= 0:
        return []
    train_count = max(1, int(round(count * 0.70)))
    val_count = max(1, int(round(count * 0.15)))
    if train_count + val_count >= count:
        val_count = max(1, count - train_count - 1)
    test_count = count - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > val_count:
            train_count -= 1
        else:
            val_count -= 1

    splits = (["train"] * train_count) + (["val"] * val_count) + (["test"] * test_count)
    return splits


def build_manifest(config_path: str = "config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    # Split assignment only — never conflate with train_seed / training stochasticity.
    split_seed = int(config.get("split_seed", config.get("seed", 42)))
    set_seed(split_seed)
    print(
        f"Building stratified splits with split_seed={split_seed} "
        "(frozen for the published study; do not vary with train_seed)."
    )

    data_root = Path(config["data_root"])
    results_root = Path(config["results_root"])
    raw_root = data_root / "raw"
    results_root.mkdir(parents=True, exist_ok=True)

    variant_lookup = build_variant_lookup()
    records: list[dict[str, object]] = []
    excluded_augmented: dict[str, int] = defaultdict(int)
    excluded_raw_classes: dict[str, int] = defaultdict(int)
    unknown_classes: set[tuple[str, str]] = set()

    for dataset_name, dataset_cfg in config["datasets"].items():
        dataset_folder = raw_root / dataset_cfg["raw_folder"]
        if not dataset_folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")

        for image_path in dataset_folder.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if is_augmented_path(image_path):
                excluded_augmented[dataset_name] += 1
                continue

            original_class = image_path.parent.name
            if original_class in EXCLUDED_RAW_CLASSES.get(dataset_name, set()):
                excluded_raw_classes[f"{dataset_name}:{original_class}"] += 1
                continue

            normalized = normalize_label(original_class)
            mapped_class = variant_lookup.get(normalized)
            if mapped_class is None:
                unknown_classes.add((dataset_name, original_class))
                continue

            records.append(
                {
                    "image_path": str(image_path),
                    "dataset": dataset_name,
                    "original_class": original_class,
                    "mapped_class": mapped_class,
                    "background": infer_background(dataset_name, image_path),
                    "split": "",
                    "is_duplicate": False,
                }
            )

    if unknown_classes:
        unknown_msg = ", ".join(
            [f"{dataset}:{raw_class}" for dataset, raw_class in sorted(unknown_classes)]
        )
        raise ValueError(
            "Unmapped raw class names found. Extend canonical label map before proceeding: "
            f"{unknown_msg}"
        )

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise ValueError("No valid images found while building manifest.")

    split_assignments: dict[int, str] = {}
    grouped = manifest.groupby(["dataset", "mapped_class"], sort=True).groups
    for _, row_indices in grouped.items():
        row_idx_list = list(row_indices)
        shuffled_indices = pd.Series(row_idx_list).sample(frac=1.0, random_state=split_seed).tolist()
        splits = stratified_split_indices(len(shuffled_indices), split_seed)
        for idx, split in zip(shuffled_indices, splits):
            split_assignments[idx] = split

    manifest["split"] = manifest.index.map(split_assignments)

    manifest_path = results_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("Excluded augmented-like images per dataset:")
    for dataset_name in config["datasets"]:
        print(f"  {dataset_name}: {excluded_augmented.get(dataset_name, 0)}")

    if excluded_raw_classes:
        print("\nExcluded out-of-scope raw classes:")
        for key, count in sorted(excluded_raw_classes.items()):
            print(f"  {key}: {count}")

    pivot = (
        manifest.pivot_table(
            index="dataset",
            columns="mapped_class",
            values="image_path",
            aggfunc="count",
            fill_value=0,
        )
        .sort_index()
        .sort_index(axis=1)
    )
    print("\nPer-dataset class counts:")
    print(pivot)
    print(f"\nManifest written to: {manifest_path}")
    return manifest


if __name__ == "__main__":
    build_manifest()
