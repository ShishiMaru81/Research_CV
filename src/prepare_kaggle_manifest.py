from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _posix(path: str) -> str:
    return str(path).replace("\\", "/")


def _rel_under_dataset(path: str, dataset: str, original_class: str) -> str:
    posix = _posix(path)
    for marker in (f"data/raw/{dataset}/", f"/{dataset}/", f"{dataset}/"):
        if marker in posix:
            return posix.split(marker, 1)[1]
    return f"{original_class}/{posix.split('/')[-1]}"


def prepare_manifest(
    src_manifest: str | Path,
    out_manifest: str | Path,
    image_roots: dict[str, str],
) -> pd.DataFrame:
    df = pd.read_csv(src_manifest)
    new_paths: list[str] = []
    for _, row in df.iterrows():
        dataset = str(row["dataset"])
        if dataset not in image_roots:
            new_paths.append(str(row["image_path"]))
            continue
        rel = _rel_under_dataset(str(row["image_path"]), dataset, str(row["original_class"]))
        new_paths.append(str(Path(image_roots[dataset]) / rel))
    df = df.copy()
    df["image_path"] = new_paths

    out = Path(out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {out} ({len(df)} rows)")
    for dataset, root in image_roots.items():
        subset = df[df["dataset"] == dataset]
        if subset.empty:
            print(f"  {dataset}: no rows")
            continue
        sample = Path(subset.iloc[0]["image_path"])
        print(f"  {dataset}: sample={sample}")
        print(f"           exists={sample.exists()} | root={root}")
        if not sample.exists():
            parent = sample.parent
            print(f"           parent exists={parent.exists()} | {parent}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite manifest paths for Kaggle dataset roots.")
    parser.add_argument("--src", default="artifacts/manifest.csv")
    parser.add_argument("--out", default="results/manifest.csv")
    parser.add_argument(
        "--image_roots",
        nargs="+",
        required=True,
        help="dataset=/abs/path pairs, e.g. riceleafbd=/kaggle/input/.../riceleafbd",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    roots: dict[str, str] = {}
    for item in args.image_roots:
        key, value = item.split("=", 1)
        roots[key.strip()] = value.strip()
    prepare_manifest(args.src, args.out, roots)
