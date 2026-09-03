"""Build MobileSAM leaf masks for all non-duplicate manifest images (Week 12).

Reads frozen_results/manifest.csv (immutable). Writes masked RGB images under
data/masked/sam_leaf/ and quality metrics to frozen_results_v2/sam_mask_quality.csv.

Never writes numeric findings into this file — all metrics are computed at runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "frozen_results" / "manifest.csv"
OUT_DIR = ROOT / "data" / "masked" / "sam_leaf"
OUT_CSV = ROOT / "frozen_results_v2" / "sam_mask_quality.csv"
CHECKPOINT = ROOT / "weights" / "mobile_sam.pt"
SEED = 42


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"numpy.random.seed({seed})")
    print(f"torch.manual_seed({seed})")


def _relative_under_raw(image_path: str) -> Path:
    """Strip data/raw/ (any slash style) to get dataset/.../file.jpg."""
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
    if "is_duplicate" not in df.columns:
        raise KeyError(f"manifest missing is_duplicate column: {MANIFEST}")
    nondup = df[df["is_duplicate"] == False].reset_index(drop=True)
    if len(nondup) != 5419:
        raise ValueError(
            f"Expected 5419 non-duplicate rows, got {len(nondup)} from {MANIFEST}"
        )
    return nondup


def _exg_score(img_rgb: np.ndarray, mask: np.ndarray) -> float:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return float("-inf")
    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)
    b = img_rgb[:, :, 2].astype(np.float64)
    exg = 2.0 * g - r - b
    return float(np.mean(exg[mask]))


def _component_stats(mask: np.ndarray) -> tuple[int, float]:
    labeled, n_comp = ndimage.label(mask.astype(np.uint8))
    if n_comp == 0:
        return 0, 0.0
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(sizes.max())
    return int(n_comp), float(largest / mask.size)


def _load_mobile_sam(device: torch.device):
    try:
        from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "MobileSAM is not installed. Install with:\n"
            "  pip install git+https://github.com/ChaoningZhang/MobileSAM.git\n"
            "Also place the checkpoint at weights/mobile_sam.pt "
            "(see https://github.com/ChaoningZhang/MobileSAM)."
        ) from exc

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"MobileSAM checkpoint not found: {CHECKPOINT}\n"
            "Download mobile_sam.pt from the MobileSAM release and place it at "
            "Research_CV/weights/mobile_sam.pt"
        )

    # Official MobileSAM registry key is vit_t (tiny).
    sam = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT))
    sam.to(device)
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def _select_best_mask(
    img_rgb: np.ndarray, amg_masks: list[dict]
) -> np.ndarray:
    if len(amg_masks) == 0:
        raise RuntimeError("MobileSAM returned zero masks for this image.")
    best_score = float("-inf")
    best_mask: np.ndarray | None = None
    for item in amg_masks:
        seg = item["segmentation"]
        score = _exg_score(img_rgb, seg)
        if score > best_score:
            best_score = score
            best_mask = seg.astype(bool)
    if best_mask is None:
        raise RuntimeError("Failed to select a mask (empty AMG list after scoring).")
    return best_mask


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = _load_manifest()
    print(f"Loaded manifest: {len(df)} non-duplicate rows from {MANIFEST}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(f"--limit must be positive, got {args.limit}")
        df = df.iloc[: args.limit].reset_index(drop=True)
        print(f"SMOKE TEST: limiting to first {len(df)} rows")

    mask_generator = _load_mobile_sam(device)
    print("Loaded MobileSAM (vit_t + SamAutomaticMaskGenerator)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="SAM masks"):
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
        amg_masks = mask_generator.generate(img_rgb)
        leaf_mask = _select_best_mask(img_rgb, amg_masks)

        masked = img_rgb.copy()
        masked[~leaf_mask] = 0
        Image.fromarray(masked).save(dst)

        n_comp, largest_frac = _component_stats(leaf_mask)
        rows.append(
            {
                "image_path": image_path,
                "dataset": row["dataset"],
                "foreground_fraction": float(leaf_mask.sum() / leaf_mask.size),
                "n_components": n_comp,
                "largest_component_fraction": largest_frac,
            }
        )

    quality = pd.DataFrame(rows)
    quality.to_csv(OUT_CSV, index=False)

    print(f"\nSaved masks to {OUT_DIR}/")
    print(f"Processed: {len(quality)}, Failures (missing images): {len(failures)}")
    if failures:
        print("Missing image paths (first 20):")
        for p in failures[:20]:
            print(f"  {p}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

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

    print(f"\nQuality CSV: {OUT_CSV} ({len(quality)} rows)")
    print(f"Shape: {quality.shape}")
    print(quality.head())
    print(quality.describe())


if __name__ == "__main__":
    main()
