"""Package a partial or full results tree as week11_results.zip for Kaggle upload.

Expected zip layout (matches notebooks/kaggle_week11.md Cell 8):
  week11_bundle/
    checkpoints/
    predictions/
    multiseed/
    run_registry.csv
    manifest.csv

Usage:
  python scripts/package_week11_results.py
  python scripts/package_week11_results.py --src week11_light --out dist/week11_results.zip
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "week11_light"
DEFAULT_OUT = ROOT / "dist" / "week11_results.zip"


def package(src: Path, out_zip: Path) -> dict[str, int]:
    if not src.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src}")

    bundle = out_zip.parent / "_week11_bundle_staging"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    counts: dict[str, int] = {}
    for folder in ("multiseed", "checkpoints", "predictions"):
        src_dir = src / folder
        if src_dir.exists():
            shutil.copytree(src_dir, bundle / folder)
            if folder == "checkpoints":
                counts["checkpoints"] = len(list((bundle / folder).glob("*.pth")))
            elif folder == "predictions":
                counts["predictions"] = len(list((bundle / folder).glob("*.csv")))
            else:
                counts["multiseed_files"] = len(list((bundle / folder).rglob("*.csv")))
        else:
            counts[folder] = 0

    for name in ("run_registry.csv", "manifest.csv"):
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, bundle / name)
            counts[name] = 1
        else:
            counts[name] = 0

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(bundle).as_posix())

    shutil.rmtree(bundle)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    counts = package(args.src, args.out)
    print(f"Wrote: {args.out}")
    print(f"  checkpoints (.pth): {counts.get('checkpoints', 0)}")
    print(f"  predictions (.csv): {counts.get('predictions', 0)}")
    print(f"  run_registry.csv: {'yes' if counts.get('run_registry.csv') else 'no'}")
    print(f"  manifest.csv: {'yes' if counts.get('manifest.csv') else 'no'}")
    if counts.get("checkpoints", 0) == 0:
        print(
            "\nWARNING: No .pth files in package. "
            "This zip is metrics/predictions only — not sufficient for AdaBN or leaf-removal eval. "
            "Restore full checkpoints from a prior Kaggle session Output if you have one."
        )


if __name__ == "__main__":
    main()
