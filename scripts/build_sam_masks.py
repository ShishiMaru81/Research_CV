"""Build MobileSAM leaf masks for all non-duplicate manifest images (Week 12).

Reads frozen_results/manifest.csv (immutable). Writes masked RGB images under
data/masked/sam_leaf/ and quality metrics to frozen_results_v2/sam_mask_quality.csv.

Mask selection: a single positive point prompt at the image centre, taking the
SAM output with the highest predicted IoU, then the largest connected component.
The original brief's rule (automatic masks, keep highest mean ExG) was replaced
because it selects the greenest fragment, which for disease images is always a
healthy background blade rather than the discoloured subject leaf. See
notes/week12_masking_plan.md, "SAM selection criterion".

Never writes numeric findings into this file — all metrics are computed at runtime.
"""

from __future__ import annotations

import argparse
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


def _component_stats(mask: np.ndarray) -> tuple[int, float]:
    labeled, n_comp = ndimage.label(mask.astype(np.uint8))
    if n_comp == 0:
        return 0, 0.0
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(sizes.max())
    return int(n_comp), float(largest / mask.size)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n_comp = ndimage.label(mask.astype(np.uint8))
    if n_comp == 0:
        raise RuntimeError("SAM mask has zero foreground pixels.")
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def _load_mobile_sam(device: torch.device):
    try:
        from mobile_sam import SamPredictor, sam_model_registry
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
    return SamPredictor(sam)


def _center_prompt_mask(predictor, img_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Segment the object under the image centre; return (mask, sam_iou_score)."""
    h, w = img_rgb.shape[:2]
    predictor.set_image(img_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[w // 2, h // 2]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )
    if masks.shape[0] == 0:
        raise RuntimeError("MobileSAM returned zero masks for centre prompt.")
    best = int(np.argmax(scores))
    mask = masks[best].astype(bool)
    if not mask.any():
        raise RuntimeError("MobileSAM centre-prompt mask is empty.")
    return _largest_component(mask), float(scores[best])


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

    predictor = _load_mobile_sam(device)
    print("Loaded MobileSAM (vit_t + SamPredictor, centre-point prompt)")

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
        leaf_mask, sam_score = _center_prompt_mask(predictor, img_rgb)

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
                "sam_score": sam_score,
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
