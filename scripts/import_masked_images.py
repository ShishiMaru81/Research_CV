"""Import masked image trees from a Kaggle input folder.

Kaggle inputs are read-only and often have an extra dataset-slug directory in
front of the real files.  This script finds a tree containing the three dataset
folders, copies only the manifest-listed images into data/masked/{condition}/,
and verifies that every non-duplicate frozen image has a masked counterpart.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "frozen_results" / "manifest.csv"
MASKED_ROOT = ROOT / "data" / "masked"
DATASETS = ("riceleafbd", "dhan_shomadhan", "brri_rice_disease_pest")
CONDITIONS = ("sam_leaf", "hsv_leaf")
EXPECTED_NONDUP = 5419


def _relative_under_raw(image_path: str) -> Path:
    posix = str(image_path).replace("\\", "/")
    marker = "data/raw/"
    if marker not in posix:
        raise ValueError(f"Expected image_path under data/raw/, got: {image_path!r}")
    return Path(posix.split(marker, 1)[1])


def _load_manifest() -> pd.DataFrame:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing immutable manifest: {MANIFEST}")
    df = pd.read_csv(MANIFEST)
    if "is_duplicate" not in df.columns:
        raise KeyError(f"manifest missing is_duplicate column: {MANIFEST}")
    df = df[df["is_duplicate"] == False].reset_index(drop=True)
    if len(df) != EXPECTED_NONDUP:
        raise ValueError(
            f"Expected {EXPECTED_NONDUP} non-duplicate rows, got {len(df)}"
        )
    return df


def _candidate_roots(input_root: Path, condition: str) -> list[Path]:
    direct = [
        input_root,
        input_root / condition,
        input_root / "data" / "masked" / condition,
        input_root / "masked" / condition,
    ]
    seen: set[Path] = set()
    candidates: list[Path] = []
    for path in direct:
        resolved = path.resolve()
        if resolved not in seen:
            candidates.append(path)
            seen.add(resolved)

    for child in input_root.rglob("*"):
        if not child.is_dir():
            continue
        if child.name != condition and not all((child / ds).is_dir() for ds in DATASETS):
            continue
        path = child if child.name != condition else child
        resolved = path.resolve()
        if resolved not in seen:
            candidates.append(path)
            seen.add(resolved)
    return candidates


def _looks_like_mask_tree(path: Path) -> bool:
    return all((path / dataset).is_dir() for dataset in DATASETS)


def _find_source_root(input_root: Path, condition: str) -> Path:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root is not a directory: {input_root}")

    for candidate in _candidate_roots(input_root, condition):
        if _looks_like_mask_tree(candidate):
            return candidate

    tried = "\n  ".join(str(p) for p in _candidate_roots(input_root, condition)[:20])
    raise FileNotFoundError(
        "Could not find a masked image tree containing all dataset folders under "
        f"{input_root}.\nTried:\n  {tried}"
    )


def _copy_manifest_images(
    manifest: pd.DataFrame,
    source_root: Path,
    dest_root: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> tuple[int, int, list[Path]]:
    copied = 0
    already_present = 0
    missing: list[Path] = []

    for image_path in tqdm(
        manifest["image_path"].astype(str), total=len(manifest), desc="masked images"
    ):
        rel = _relative_under_raw(image_path)
        src = source_root / rel
        dst = dest_root / rel
        if not src.is_file():
            missing.append(src)
            continue
        if dst.is_file() and not overwrite:
            already_present += 1
            continue
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1

    return copied, already_present, missing


def _verify_dest(manifest: pd.DataFrame, dest_root: Path) -> list[Path]:
    missing: list[Path] = []
    for image_path in manifest["image_path"].astype(str):
        rel = _relative_under_raw(image_path)
        path = dest_root / rel
        if not path.is_file():
            missing.append(path)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        required=True,
        help="Masked-image variant to import.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Kaggle input folder, e.g. /kaggle/input/my-sam-masks.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=None,
        help="Destination masked tree. Defaults to data/masked/{condition}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find and count files without copying.",
    )
    args = parser.parse_args()

    manifest = _load_manifest()
    source_root = _find_source_root(args.input_root.resolve(), args.condition)
    dest_root = (
        args.dest_root.resolve()
        if args.dest_root is not None
        else (MASKED_ROOT / args.condition).resolve()
    )

    print(f"Condition: {args.condition}")
    print(f"Source root: {source_root}")
    print(f"Destination root: {dest_root}")
    print(f"Manifest rows: {len(manifest)}")

    copied, already_present, source_missing = _copy_manifest_images(
        manifest,
        source_root,
        dest_root,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(f"Copied: {copied}")
    print(f"Already present: {already_present}")
    print(f"Missing at source: {len(source_missing)}")
    if source_missing:
        for path in source_missing[:20]:
            print(f"  missing source: {path}")
        if len(source_missing) > 20:
            print(f"  ... and {len(source_missing) - 20} more")
        raise FileNotFoundError(
            f"Source tree is incomplete for {args.condition}: "
            f"{len(source_missing)} manifest images missing."
        )

    if args.dry_run:
        print("DRY RUN: destination verification skipped.")
        return

    dest_missing = _verify_dest(manifest, dest_root)
    print(f"Missing at destination after import: {len(dest_missing)}")
    if dest_missing:
        for path in dest_missing[:20]:
            print(f"  missing dest: {path}")
        if len(dest_missing) > 20:
            print(f"  ... and {len(dest_missing) - 20} more")
        raise FileNotFoundError(
            f"Destination tree is incomplete for {args.condition}: "
            f"{len(dest_missing)} manifest images missing."
        )

    file_count = sum(1 for path in dest_root.rglob("*") if path.is_file())
    print(f"Import verified: {len(manifest)}/{len(manifest)} manifest images present.")
    print(f"Destination file count: {file_count}")


if __name__ == "__main__":
    main()
