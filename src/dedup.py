from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.utils import load_config


@dataclass(frozen=True)
class HashEntry:
    idx: int
    dataset: str
    image_path: str
    phash_int: int


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, int] | None = None
        self.children: dict[int, dict[int, tuple[int, int]]] = {}

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def add(self, key: int, value: int) -> None:
        node = (key, value)
        if self.root is None:
            self.root = node
            self.children[id(node)] = {}
            return
        cur = self.root
        while True:
            cur_key, _ = cur
            dist = self._hamming(key, cur_key)
            cur_children = self.children[id(cur)]
            if dist in cur_children:
                cur = cur_children[dist]
                continue
            cur_children[dist] = node
            self.children[id(node)] = {}
            return

    def query(self, key: int, max_dist: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        out: list[tuple[int, int]] = []
        stack = [self.root]
        while stack:
            cur = stack.pop()
            cur_key, cur_value = cur
            dist = self._hamming(key, cur_key)
            if dist <= max_dist:
                out.append((cur_value, dist))
            cur_children = self.children[id(cur)]
            lo, hi = dist - max_dist, dist + max_dist
            for edge_dist, child in cur_children.items():
                if lo <= edge_dist <= hi:
                    stack.append(child)
        return out


def compute_phash_int(image_path: str, hash_size: int = 8) -> int:
    with Image.open(image_path) as img:
        return int(str(imagehash.phash(img.convert("RGB"), hash_size=hash_size)), 16)


def choose_keep_index(group_entries: list[HashEntry], duplicate_counts: dict[int, int]) -> int:
    # Prefer keeping entries that are least duplicated globally.
    return min(
        group_entries,
        key=lambda entry: (duplicate_counts.get(entry.idx, 0), entry.dataset, entry.image_path),
    ).idx


def deduplicate(config_path: str = "config.yaml", threshold: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    results_root = Path(config["results_root"])
    manifest_path = results_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required_cols = {
        "image_path",
        "dataset",
        "split",
        "is_duplicate",
    }
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    entries: list[HashEntry] = []
    for idx, row in manifest.iterrows():
        image_path = str(row["image_path"])
        try:
            phash_int = compute_phash_int(image_path, hash_size=8)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Failed to hash image {image_path}: {exc}") from exc
        entries.append(HashEntry(idx=idx, dataset=str(row["dataset"]), image_path=image_path, phash_int=phash_int))

    tree = BKTree()
    for position, entry in enumerate(entries):
        tree.add(entry.phash_int, position)

    collisions: list[dict[str, object]] = []
    parent = list(range(len(entries)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, entry in enumerate(entries):
        neighbors = tree.query(entry.phash_int, max_dist=threshold)
        for neighbor_pos, distance in neighbors:
            if neighbor_pos <= i:
                continue
            other = entries[neighbor_pos]
            if entry.dataset == other.dataset:
                continue
            collisions.append(
                {
                    "path_a": entry.image_path,
                    "dataset_a": entry.dataset,
                    "path_b": other.image_path,
                    "dataset_b": other.dataset,
                    "hamming_distance": distance,
                }
            )
            union(i, neighbor_pos)

    groups: dict[int, list[HashEntry]] = defaultdict(list)
    duplicate_counts: dict[int, int] = defaultdict(int)
    for i, entry in enumerate(entries):
        groups[find(i)].append(entry)
    for group_entries in groups.values():
        if len(group_entries) > 1:
            for entry in group_entries:
                duplicate_counts[entry.idx] += len(group_entries) - 1

    duplicate_indices: set[int] = set()
    for group_entries in groups.values():
        if len(group_entries) <= 1:
            continue
        keep_idx = choose_keep_index(group_entries, duplicate_counts)
        for entry in group_entries:
            if entry.idx != keep_idx:
                duplicate_indices.add(entry.idx)

    manifest["is_duplicate"] = manifest.index.isin(duplicate_indices)
    manifest.to_csv(manifest_path, index=False)

    report_columns = ["path_a", "dataset_a", "path_b", "dataset_b", "hamming_distance"]
    if collisions:
        duplicates_report = pd.DataFrame(collisions).sort_values(
            by=["dataset_a", "dataset_b", "hamming_distance", "path_a", "path_b"]
        )
    else:
        duplicates_report = pd.DataFrame(columns=report_columns)

    report_path = results_root / "duplicates_report.csv"
    duplicates_report.to_csv(report_path, index=False)

    if len(duplicates_report) > 0:
        pair_counts = (
            duplicates_report.assign(pair=lambda d: d["dataset_a"] + " -> " + d["dataset_b"])
            .groupby("pair")
            .size()
            .sort_index()
        )
        print("Duplicate pairs by dataset pair:")
        for pair, count in pair_counts.items():
            print(f"  {pair}: {int(count)}")
    else:
        print("Duplicate pairs by dataset pair: none")

    print(f"Total cross-dataset duplicate pairs: {len(duplicates_report)}")
    print(f"Flagged manifest rows as duplicates: {len(duplicate_indices)}")
    print(f"Updated manifest: {manifest_path}")
    print(f"Duplicates report: {report_path}")

    return manifest, duplicates_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-dataset near-duplicate detection with pHash.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Maximum Hamming distance for pHash duplicate match.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    deduplicate(config_path=args.config, threshold=args.threshold)
