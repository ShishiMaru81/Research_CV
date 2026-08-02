"""Week 13 Phase-3 augmentation bucket ablation orchestrator.

Runs ResNet50 × 6 transfer pairs × 3 mechanism buckets × train seed 42
(18 training jobs). Each bucket is applied alone so we can attribute the
strong-augmentation gain to geometric, photometric, or occlusion transforms.

Outputs:
  results/ablation/augmentation_ablation.csv
  results/ablation/crossdataset_matrix_bucket_{geo,photo,occlusion}.csv
  results/ablation/generalization_gap_bucket_{geo,photo,occlusion}.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from run_crossdataset import TRANSFER_PAIRS, run_crossdataset
from src.data_loader import BUCKET_PROFILES
from src.run_registry import RunRegistry, default_registry_path, make_run_id
from src.utils import load_config


ABLATION_MODEL = "resnet50"
DEFAULT_TRAIN_SEED = 42
SPLIT_SEED = 42
MINUTES_PER_RUN = 13.0

BUCKET_SHORT = {
    "bucket-geo": "geo",
    "bucket-photo": "photo",
    "bucket-occlusion": "occlusion",
}


@dataclass(frozen=True)
class AblationJob:
    bucket: str
    train_dataset: str
    test_dataset: str
    classes: str
    run_tag: str
    train_seed: int = DEFAULT_TRAIN_SEED

    @property
    def run_id(self) -> str:
        return make_run_id(
            model=ABLATION_MODEL,
            train_datasets=[self.train_dataset],
            train_seed=self.train_seed,
            eval_dataset=self.test_dataset,
            run_tag=f"{self.run_tag}__aug-{self.bucket}",
            augmentation=self.bucket,
        )

    @property
    def label(self) -> str:
        return (
            f"{ABLATION_MODEL} | {self.train_dataset} -> {self.test_dataset} | "
            f"seed={self.train_seed} | {self.bucket}"
        )


def build_ablation_jobs(
    buckets: list[str] | None = None,
    train_seed: int = DEFAULT_TRAIN_SEED,
) -> list[AblationJob]:
    buckets = buckets or list(BUCKET_PROFILES)
    jobs: list[AblationJob] = []
    for bucket in buckets:
        if bucket not in BUCKET_PROFILES:
            raise ValueError(f"Unknown bucket '{bucket}'. Use {BUCKET_PROFILES}.")
        for pair in TRANSFER_PAIRS:
            jobs.append(
                AblationJob(
                    bucket=bucket,
                    train_dataset=pair.train_dataset,
                    test_dataset=pair.test_dataset,
                    classes=pair.class_string,
                    run_tag=pair.run_tag,
                    train_seed=train_seed,
                )
            )
    return jobs


def estimate_hours(n_jobs: int) -> float:
    return n_jobs * MINUTES_PER_RUN / 60.0


def _append_ablation_rows(
    ablation_path: Path,
    matrix_path: Path,
    gap_path: Path,
    bucket: str,
    train_seed: int,
) -> None:
    if not matrix_path.exists() or not gap_path.exists():
        return
    matrix = pd.read_csv(matrix_path)
    gap = pd.read_csv(gap_path)
    keys = ["train_dataset", "test_dataset", "model", "classes", "seed"]
    merged = matrix.merge(gap[keys + ["generalization_gap"]], on=keys, how="left")
    merged = merged[merged["seed"] == train_seed].copy()
    merged["augmentation"] = bucket
    merged["bucket"] = BUCKET_SHORT[bucket]
    cols = [
        "train_dataset",
        "test_dataset",
        "model",
        "classes",
        "seed",
        "augmentation",
        "bucket",
        "macro_f1",
        "accuracy",
        "generalization_gap",
        "n_samples",
        "checkpoint_path",
    ]
    for col in cols:
        if col not in merged.columns:
            merged[col] = ""
    out = merged[cols]
    if ablation_path.exists():
        existing = pd.read_csv(ablation_path)
        combined = pd.concat([existing, out], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["train_dataset", "test_dataset", "model", "seed", "augmentation"],
            keep="last",
        )
    else:
        combined = out
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(ablation_path, index=False)


def run_ablation(
    config_path: str = "config.yaml",
    buckets: list[str] | None = None,
    train_seed: int = DEFAULT_TRAIN_SEED,
    image_roots: dict[str, str] | None = None,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> Path:
    config = load_config(config_path)
    results_root = Path(config["results_root"])
    ablation_dir = results_root / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    ablation_csv = ablation_dir / "augmentation_ablation.csv"
    registry = RunRegistry(default_registry_path(results_root))
    jobs = build_ablation_jobs(buckets=buckets, train_seed=train_seed)

    print(f"Ablation jobs: {len(jobs)} (~{estimate_hours(len(jobs)):.1f} GPU-hrs)")
    if dry_run:
        for index, job in enumerate(jobs, start=1):
            status = "complete" if registry.is_complete(job.run_id) else "pending"
            print(f"  [{index:02d}/{len(jobs)}] {status}: {job.label}")
        print(f"Would write: {ablation_csv}")
        return ablation_csv

    buckets_to_run = buckets or list(BUCKET_PROFILES)
    for bucket in buckets_to_run:
        short = BUCKET_SHORT[bucket]
        print(f"\n======== BUCKET {bucket} ========")
        matrix_name = f"crossdataset_matrix_bucket_{short}.csv"
        gap_name = f"generalization_gap_bucket_{short}.csv"
        run_crossdataset(
            config_path=config_path,
            models=[ABLATION_MODEL],
            pair_keys=None,
            seeds=[train_seed],
            image_roots=image_roots,
            skip_existing=skip_existing,
            augmentation=bucket,
            matrix_filename=matrix_name,
            gap_filename=gap_name,
            output_dir=ablation_dir,
        )
        _append_ablation_rows(
            ablation_path=ablation_csv,
            matrix_path=ablation_dir / matrix_name,
            gap_path=ablation_dir / gap_name,
            bucket=bucket,
            train_seed=train_seed,
        )

    print(f"\nWrote: {ablation_csv}")
    if ablation_csv.exists():
        df = pd.read_csv(ablation_csv)
        print(df.groupby("bucket").size().to_string())
    return ablation_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 ResNet50 augmentation bucket ablation."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=None,
        choices=list(BUCKET_PROFILES),
        help="Subset of buckets (default: all three).",
    )
    parser.add_argument("--train_seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument(
        "--image_roots",
        nargs="*",
        default=None,
        help="dataset=/abs/path pairs for Kaggle mounts.",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-run combinations already present in outputs/registry.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List jobs and resume status without training.",
    )
    return parser.parse_args()


def _parse_image_roots(items: list[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    roots: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --image_roots item '{item}'. Use dataset=/abs/path"
            )
        key, value = item.split("=", 1)
        roots[key.strip()] = value.strip()
    return roots


if __name__ == "__main__":
    args = parse_args()
    run_ablation(
        config_path=args.config,
        buckets=args.buckets,
        train_seed=args.train_seed,
        image_roots=_parse_image_roots(args.image_roots),
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run,
    )
