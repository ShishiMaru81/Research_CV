"""Week 11 multi-seed replication orchestrator.

Runs transfer baseline, transfer augmentation, and LODO for new train seeds
(7, 2024) while split_seed stays fixed at 42. Seed 42 results live in
``frozen_results/`` and are not re-run here.

Execution order is seed-major: finish all experiments for seed 7, then seed 2024.
Uses ``results/run_registry.csv`` for resume after Kaggle session timeouts.
Summary CSVs are written under ``results/multiseed/``; checkpoints and
predictions stay under ``results/checkpoints`` and ``results/predictions``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from run_crossdataset import DEFAULT_MODELS, TRANSFER_PAIRS, run_crossdataset
from run_lodo import ALL_DATASETS, build_lodo_splits, run_lodo
from src.run_registry import RunRegistry, default_registry_path, make_run_id
from src.utils import load_config


DEFAULT_NEW_TRAIN_SEEDS = [7, 2024]
SPLIT_SEED = 42
MINUTES_PER_RUN = 13.0


@dataclass(frozen=True)
class MultiseedJob:
    experiment_type: str
    model: str
    train_seed: int
    train_datasets: str
    eval_dataset: str
    classes: str
    augmentation: str
    run_tag: str | None

    @property
    def run_id(self) -> str:
        datasets = self.train_datasets.split("+")
        return make_run_id(
            model=self.model,
            train_datasets=datasets,
            train_seed=self.train_seed,
            eval_dataset=self.eval_dataset,
            run_tag=self.run_tag,
            augmentation=self.augmentation,
        )

    @property
    def label(self) -> str:
        return (
            f"{self.experiment_type} | {self.model} | "
            f"train={self.train_datasets} -> eval={self.eval_dataset} | "
            f"seed={self.train_seed} | aug={self.augmentation}"
        )


def _effective_run_tag(pair_tag: str, augmentation: str) -> str:
    if augmentation == "default":
        return pair_tag
    return f"{pair_tag}__aug-{augmentation}"


def build_multiseed_jobs(
    train_seeds: list[int] | None = None,
    models: list[str] | None = None,
) -> list[MultiseedJob]:
    """Enumerate the 45 runs per seed (90 total for two new seeds)."""
    train_seeds = train_seeds or list(DEFAULT_NEW_TRAIN_SEEDS)
    models = models or list(DEFAULT_MODELS)
    jobs: list[MultiseedJob] = []

    for train_seed in train_seeds:
        for pair in TRANSFER_PAIRS:
            classes = pair.class_string
            for model in models:
                jobs.append(
                    MultiseedJob(
                        experiment_type="transfer_baseline",
                        model=model,
                        train_seed=train_seed,
                        train_datasets=pair.train_dataset,
                        eval_dataset=pair.test_dataset,
                        classes=classes,
                        augmentation="default",
                        run_tag=pair.run_tag,
                    )
                )
                jobs.append(
                    MultiseedJob(
                        experiment_type="transfer_aug",
                        model=model,
                        train_seed=train_seed,
                        train_datasets=pair.train_dataset,
                        eval_dataset=pair.test_dataset,
                        classes=classes,
                        augmentation="strong",
                        run_tag=_effective_run_tag(pair.run_tag, "strong"),
                    )
                )

        for held_out in ALL_DATASETS:
            train_datasets = "+".join(d for d in ALL_DATASETS if d != held_out)
            run_tag = f"lodo-holdout-{held_out}"
            for model in models:
                jobs.append(
                    MultiseedJob(
                        experiment_type="lodo",
                        model=model,
                        train_seed=train_seed,
                        train_datasets=train_datasets,
                        eval_dataset=held_out,
                        classes="",  # filled at runtime by LODO runner
                        augmentation="default",
                        run_tag=run_tag,
                    )
                )

    return jobs


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


def plan_multiseed(
    config_path: str = "config.yaml",
    train_seeds: list[int] | None = None,
    models: list[str] | None = None,
) -> tuple[list[MultiseedJob], list[MultiseedJob], Path]:
    config = load_config(config_path)
    results_root = Path(config["results_root"])
    registry_path = default_registry_path(results_root)
    registry = RunRegistry(registry_path)

    jobs = build_multiseed_jobs(train_seeds=train_seeds, models=models)
    pending = [job for job in jobs if not registry.is_complete(job.run_id)]
    return jobs, pending, registry_path


def print_dry_run(
    config_path: str = "config.yaml",
    train_seeds: list[int] | None = None,
    models: list[str] | None = None,
) -> int:
    jobs, pending, registry_path = plan_multiseed(
        config_path=config_path,
        train_seeds=train_seeds,
        models=models,
    )
    seeds = train_seeds or DEFAULT_NEW_TRAIN_SEEDS
    model_list = models or DEFAULT_MODELS

    print("=== Multi-seed campaign dry run ===")
    print(f"Registry: {registry_path}")
    print(f"New train seeds: {seeds} (split_seed fixed at {SPLIT_SEED})")
    print(f"Models: {model_list}")
    print(f"Pairs: {len(TRANSFER_PAIRS)} ordered x baseline + aug = {len(TRANSFER_PAIRS) * 2 * len(model_list)} per seed")
    print(f"LODO: {len(ALL_DATASETS)} holdouts x {len(model_list)} = {len(ALL_DATASETS) * len(model_list)} per seed")
    print(f"Runs per seed: {len(jobs) // len(seeds)}")
    print(f"Total jobs enumerated: {len(jobs)}")
    print(f"Already complete (registry): {len(jobs) - len(pending)}")
    print(f"Would execute: {len(pending)}")
    est_hours = len(pending) * MINUTES_PER_RUN / 60.0
    print(
        f"Estimated GPU time (@ {MINUTES_PER_RUN:.0f} min/run): "
        f"{est_hours:.1f} hours ({len(pending) * MINUTES_PER_RUN:.0f} min)"
    )
    print("\nSeed-major execution order:")
    current_seed: int | None = None
    for job in jobs:
        if job.train_seed != current_seed:
            current_seed = job.train_seed
            print(f"\n--- train_seed={current_seed} ---")
        status = "PENDING" if job in pending else "SKIP(complete)"
        print(f"  [{status}] {job.label}")
        print(f"           run_id={job.run_id}")

    if len(jobs) != 90:
        print(f"\nWARNING: expected 90 total jobs, got {len(jobs)}.")
    return len(pending)


def run_multiseed(
    config_path: str = "config.yaml",
    train_seeds: list[int] | None = None,
    models: list[str] | None = None,
    image_roots: dict[str, str] | None = None,
    skip_existing: bool = True,
) -> None:
    config = load_config(config_path)
    results_root = Path(config["results_root"])
    multiseed_dir = results_root / "multiseed"
    multiseed_dir.mkdir(parents=True, exist_ok=True)

    seeds = train_seeds or list(DEFAULT_NEW_TRAIN_SEEDS)
    jobs, pending, _ = plan_multiseed(
        config_path=config_path,
        train_seeds=seeds,
        models=models,
    )
    pending_ids = {job.run_id for job in pending}
    if not pending:
        print("All multi-seed jobs already complete in run_registry.csv.")
        return

    print(
        f"Starting multi-seed campaign: {len(pending)}/{len(jobs)} runs pending, "
        f"~{len(pending) * MINUTES_PER_RUN / 60.0:.1f} GPU-hrs estimated."
    )

    for train_seed in seeds:
        seed_jobs = [job for job in jobs if job.train_seed == train_seed]
        seed_pending = [job for job in seed_jobs if job.run_id in pending_ids]
        if not seed_pending:
            print(f"\n=== train_seed={train_seed}: all complete, skipping ===")
            continue

        print(
            f"\n=== train_seed={train_seed}: "
            f"{len(seed_pending)}/{len(seed_jobs)} runs pending ==="
        )

        baseline_pending = any(
            j.experiment_type == "transfer_baseline" for j in seed_pending
        )
        if baseline_pending:
            print("\n--- transfer_baseline ---")
            run_crossdataset(
                config_path=config_path,
                models=models,
                seeds=[train_seed],
                image_roots=image_roots,
                skip_existing=skip_existing,
                augmentation="default",
                matrix_filename="crossdataset_matrix_baseline.csv",
                gap_filename="generalization_gap_baseline.csv",
                output_dir=multiseed_dir,
            )

        aug_pending = any(j.experiment_type == "transfer_aug" for j in seed_pending)
        if aug_pending:
            print("\n--- transfer_aug (strong) ---")
            run_crossdataset(
                config_path=config_path,
                models=models,
                seeds=[train_seed],
                image_roots=image_roots,
                skip_existing=skip_existing,
                augmentation="strong",
                matrix_filename="crossdataset_matrix_aug.csv",
                gap_filename="generalization_gap_aug.csv",
                output_dir=multiseed_dir,
            )

        lodo_pending = any(j.experiment_type == "lodo" for j in seed_pending)
        if lodo_pending:
            print("\n--- lodo ---")
            run_lodo(
                config_path=config_path,
                models=models,
                seeds=[train_seed],
                image_roots=image_roots,
                skip_existing=skip_existing,
                augmentation="default",
                output_dir=multiseed_dir,
                results_filename="lodo_results.csv",
            )

        remaining = [
            job
            for job in seed_jobs
            if job.run_id in pending_ids
            and not RunRegistry(default_registry_path(results_root)).is_complete(
                job.run_id
            )
        ]
        if remaining:
            print(
                f"WARNING: after seed {train_seed}, {len(remaining)} jobs still "
                "not marked complete — check logs."
            )

    print(f"\nMulti-seed outputs: {multiseed_dir}")
    print(f"Registry: {default_registry_path(results_root)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 11 multi-seed replication (seeds 7 and 2024)."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train_seeds", nargs="+", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--image_roots",
        nargs="*",
        default=None,
        help="dataset=/abs/path pairs for Kaggle mounts.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List every run that would execute; do not train.",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Ignore registry/CSV skip logic and re-run everything.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        pending_count = print_dry_run(
            config_path=args.config,
            train_seeds=args.train_seeds,
            models=args.models,
        )
        if pending_count == 0:
            print("\nNothing to run.")
    else:
        run_multiseed(
            config_path=args.config,
            train_seeds=args.train_seeds,
            models=args.models,
            image_roots=_parse_image_roots(args.image_roots),
            skip_existing=not args.no_skip_existing,
        )
