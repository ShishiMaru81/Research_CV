"""Build HSV/ExG leaf masks for all non-duplicate manifest images (Week 12).

Deterministic morphological alternative to SAM. Reads frozen_results/manifest.csv.
Writes data/masked/hsv_leaf/ and frozen_results_v2/hsv_mask_quality.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "frozen_results" / "manifest.csv"
OUT_DIR = ROOT / "data" / "masked" / "hsv_leaf"
OUT_CSV = ROOT / "frozen_results_v2" / "hsv_mask_quality.csv"
SAM_QUALITY = ROOT / "frozen_results_v2" / "sam_mask_quality.csv"
KERNEL_SIZE = 5
SEED = 42


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    print(f"numpy.random.seed({seed})")


def _relative_under_raw(image_path: str) -> Path:
    posix = str(image_path).replace("\\", "/")
    marker = "data/raw/"
    if marker not in posix:
        raise ValueError(
            f"Expected image_path under data/raw/, got: {image_path!r}"
        )
    return Path(posix.split(marker, 1)[1])


def _load_manifest() -> pd.DataFrame:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing immutable manifest: {MANIFEST}")
    df = pd.read_csv(MANIFEST)
    nondup = df[df["is_duplicate"] == False].reset_index(drop=True)
    if len(nondup) != 5419:
        raise ValueError(
            f"Expected 5419 non-duplicate rows, got {len(nondup)} from {MANIFEST}"
        )
    return nondup


def _component_stats(mask: np.ndarray) -> tuple[int, float]:
    labeled, n_comp = ndimage.label(mask.astype(np.uint8))
    if n_comp == 0:
        return 0, 0.0
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(sizes.max())
    return int(n_comp), float(largest / mask.size)


def _mask_one(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (masked_rgb, largest_mask, no_leaf_detected)."""
    # HSV conversion retained for protocol completeness (leaf decision uses ExG).
    _ = rgb2hsv(img_rgb.astype(np.float64) / 255.0)

    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)
    b = img_rgb[:, :, 2].astype(np.float64)
    exg = 2.0 * g - r - b

    if np.allclose(exg.min(), exg.max()):
        # Uniform ExG — Otsu undefined; treat as no leaf.
        return img_rgb.copy(), np.zeros(exg.shape, dtype=bool), True

    t = threshold_otsu(exg)
    leaf = exg > t
    structure = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=bool)
    leaf = ndimage.binary_opening(leaf, structure=structure)
    leaf = ndimage.binary_closing(leaf, structure=structure)

    labeled, n_comp = ndimage.label(leaf.astype(np.uint8))
    if n_comp == 0:
        return img_rgb.copy(), np.zeros(leaf.shape, dtype=bool), True

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    largest_mask = labeled == largest_label
    masked = img_rgb.copy()
    masked[~largest_mask] = 0
    return masked, largest_mask, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: process only the first N rows (smoke test).",
    )
    args = parser.parse_args()

    _set_seeds(SEED)
    df = _load_manifest()
    print(f"Loaded manifest: {len(df)} non-duplicate rows from {MANIFEST}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(f"--limit must be positive, got {args.limit}")
        df = df.iloc[: args.limit].reset_index(drop=True)
        print(f"SMOKE TEST: limiting to first {len(df)} rows")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n_no_leaf = 0
    failures: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="HSV masks"):
        image_path = str(row["image_path"])
        src = ROOT / Path(str(image_path).replace("\\", "/"))
        if not src.is_file():
            failures.append(image_path)
            continue

        rel = _relative_under_raw(image_path)
        dst = OUT_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(src).convert("RGB")
        img_rgb = np.asarray(img)
        masked, leaf_mask, no_leaf = _mask_one(img_rgb)
        if no_leaf:
            n_no_leaf += 1

        Image.fromarray(masked).save(dst)
        n_comp, largest_frac = _component_stats(leaf_mask)
        rows.append(
            {
                "image_path": image_path,
                "dataset": row["dataset"],
                "foreground_fraction": float(leaf_mask.sum() / leaf_mask.size)
                if leaf_mask.size
                else 0.0,
                "n_components": n_comp,
                "largest_component_fraction": largest_frac,
            }
        )

    quality = pd.DataFrame(rows)
    quality.to_csv(OUT_CSV, index=False)

    print(f"\nSaved masks to {OUT_DIR}/")
    print(f"Processed: {len(quality)}, No-leaf-detected: {n_no_leaf}")
    print(f"Failures (missing images): {len(failures)}")
    if failures:
        print("Missing image paths (first 20):")
        for p in failures[:20]:
            print(f"  {p}")

    if len(quality) == 0:
        raise RuntimeError("No masks written — every image failed or was missing.")

    print("\nMean foreground_fraction by dataset:")
    for dataset, g in quality.groupby("dataset"):
        print(
            f"  {dataset}: {g['foreground_fraction'].mean():.3f} "
            f"(±{g['foreground_fraction'].std(ddof=0):.3f})"
        )
    print("\nMean n_components by dataset:")
    for dataset, g in quality.groupby("dataset"):
        print(
            f"  {dataset}: {g['n_components'].mean():.3f} "
            f"(±{g['n_components'].std(ddof=0):.3f})"
        )

    if SAM_QUALITY.is_file():
        sam = pd.read_csv(SAM_QUALITY)
        print("\nComparison to SAM (mean foreground_fraction by dataset):")
        for dataset in sorted(set(quality["dataset"]) & set(sam["dataset"])):
            hsv_m = quality.loc[
                quality["dataset"] == dataset, "foreground_fraction"
            ].mean()
            sam_m = sam.loc[sam["dataset"] == dataset, "foreground_fraction"].mean()
            print(f"  {dataset}: HSV={hsv_m:.3f}  SAM={sam_m:.3f}")
    else:
        print(
            f"\nNote: {SAM_QUALITY} not found yet — "
            "run scripts/build_sam_masks.py to enable HSV vs SAM comparison."
        )

    print(f"\nQuality CSV: {OUT_CSV} ({len(quality)} rows)")
    print(f"Shape: {quality.shape}")
    print(quality.head())
    print(quality.describe())


if __name__ == "__main__":
    main()
