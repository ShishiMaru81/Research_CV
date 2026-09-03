"""Sample 60 stratified images and write 3-panel audit PNGs (Week 12).

Verdict columns in audit_sheet.csv are left EMPTY for hand fill.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "frozen_results" / "manifest.csv"
SAM_QUALITY = ROOT / "frozen_results_v2" / "sam_mask_quality.csv"
HSV_QUALITY = ROOT / "frozen_results_v2" / "hsv_mask_quality.csv"
RAW_ROOT = ROOT / "data" / "raw"
SAM_ROOT = ROOT / "data" / "masked" / "sam_leaf"
HSV_ROOT = ROOT / "data" / "masked" / "hsv_leaf"
OUT_DIR = ROOT / "notes" / "mask_audit"
AUDIT_CSV = OUT_DIR / "audit_sheet.csv"
SEED = 42
TARGET_PER_DATASET = 20
DATASETS = ("riceleafbd", "dhan_shomadhan", "brri_rice_disease_pest")


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


def _stratified_sample(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sample TARGET_PER_DATASET images per dataset, stratified by mapped_class."""
    selected: list[pd.DataFrame] = []
    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset].copy()
        if len(sub) == 0:
            raise ValueError(f"No rows for dataset={dataset}")
        classes = sorted(sub["mapped_class"].unique())
        # Allocate at least 1 per class when possible, then fill to TARGET_PER_DATASET.
        per_class = max(1, TARGET_PER_DATASET // len(classes))
        picks: list[pd.DataFrame] = []
        for cls in classes:
            pool = sub[sub["mapped_class"] == cls]
            n = min(per_class, len(pool))
            idxs = rng.choice(pool.index.to_numpy(), size=n, replace=False)
            picks.append(pool.loc[idxs])
        chosen = pd.concat(picks, axis=0)
        # Top up or trim to exactly TARGET_PER_DATASET.
        if len(chosen) < TARGET_PER_DATASET:
            remaining = sub.drop(index=chosen.index)
            need = TARGET_PER_DATASET - len(chosen)
            if len(remaining) < need:
                raise ValueError(
                    f"Cannot sample {TARGET_PER_DATASET} from {dataset}: "
                    f"only {len(sub)} available"
                )
            extra_idxs = rng.choice(
                remaining.index.to_numpy(), size=need, replace=False
            )
            chosen = pd.concat([chosen, remaining.loc[extra_idxs]], axis=0)
        elif len(chosen) > TARGET_PER_DATASET:
            keep = rng.choice(
                chosen.index.to_numpy(), size=TARGET_PER_DATASET, replace=False
            )
            chosen = chosen.loc[keep]
        selected.append(chosen.reset_index(drop=True))
    return pd.concat(selected, axis=0).reset_index(drop=True)


def _resize_panel(img: Image.Image, size: tuple[int, int] = (384, 384)) -> Image.Image:
    return img.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def _make_panel(
    raw: Image.Image, sam: Image.Image, hsv: Image.Image, title_prefix: str
) -> Image.Image:
    w, h = 384, 384
    label_h = 28
    canvas = Image.new("RGB", (w * 3, h + label_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    panels = [
        (_resize_panel(raw, (w, h)), "Raw"),
        (_resize_panel(sam, (w, h)), "SAM"),
        (_resize_panel(hsv, (w, h)), "HSV"),
    ]
    for i, (im, label) in enumerate(panels):
        canvas.paste(im, (i * w, label_h))
        draw.text((i * w + 8, 4), f"{title_prefix} | {label}", fill=(0, 0, 0))
    return canvas


def main() -> None:
    _set_seeds(SEED)
    rng = np.random.default_rng(SEED)

    for path in (SAM_QUALITY, HSV_QUALITY):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required quality CSV: {path}. "
                "Run build_sam_masks.py and build_hsv_masks.py first."
            )

    # Presence checks — quality CSVs must cover the sample universe.
    sam_q = pd.read_csv(SAM_QUALITY)
    hsv_q = pd.read_csv(HSV_QUALITY)
    print(f"Loaded SAM quality: {sam_q.shape}")
    print(f"Loaded HSV quality: {hsv_q.shape}")

    df = _load_manifest()
    # Only sample images that have both masks on disk (intersection of quality CSVs).
    common = set(sam_q["image_path"].astype(str)) & set(hsv_q["image_path"].astype(str))
    df = df[df["image_path"].astype(str).isin(common)].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(
            "No overlapping image_path between SAM and HSV quality CSVs."
        )

    sample = _stratified_sample(df, rng)
    print(f"\nSampled {len(sample)} images stratified by dataset and class:")
    for dataset, g in sample.groupby("dataset"):
        class_counts = g["mapped_class"].value_counts().to_dict()
        print(f"  {dataset}: {len(g)} images {class_counts}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict] = []

    for i, row in sample.iterrows():
        index = i + 1
        image_path = str(row["image_path"])
        rel = _relative_under_raw(image_path)
        raw_path = ROOT / Path(image_path)
        sam_path = SAM_ROOT / rel
        hsv_path = HSV_ROOT / rel
        for p in (raw_path, sam_path, hsv_path):
            if not p.is_file():
                raise FileNotFoundError(f"Missing image for audit panel: {p}")

        raw_img = Image.open(raw_path)
        sam_img = Image.open(sam_path)
        hsv_img = Image.open(hsv_path)
        panel = _make_panel(raw_img, sam_img, hsv_img, f"{index:03d}")
        out_png = OUT_DIR / f"panel_{index:03d}.png"
        panel.save(out_png)

        audit_rows.append(
            {
                "index": index,
                "image_path": image_path,
                "dataset": row["dataset"],
                "mapped_class": row["mapped_class"],
                "sam_verdict": "",
                "hsv_verdict": "",
                "reason_code": "",
                "notes": "",
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(AUDIT_CSV, index=False)

    print(f"\nSaved panels to: {OUT_DIR}/panel_001.png ... panel_{len(audit):03d}.png")
    print(f"Audit sheet: {AUDIT_CSV} ({len(audit)} rows, verdicts empty)")
    print("Ready for hand audit.")
    print(f"\nShape: {audit.shape}")
    print(audit.head())
    print(audit.groupby("dataset").size())


if __name__ == "__main__":
    main()
