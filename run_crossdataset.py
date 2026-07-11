from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.eval import evaluate
from src.train import train, training_stem
from src.utils import load_config


DEFAULT_MODELS = ["mobilenetv2_100", "efficientnet_b0", "resnet50"]


@dataclass(frozen=True)
class TransferPair:
    train_dataset: str
    test_dataset: str
    classes: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.train_dataset}:{self.test_dataset}"

    @property
    def class_string(self) -> str:
        return "|".join(self.classes)

    @property
    def run_tag(self) -> str:
        class_tag = "+".join(self.classes)
        return f"to-{self.test_dataset}__classes-{class_tag}"


TRANSFER_PAIRS = [
    TransferPair("riceleafbd", "dhan_shomadhan", ("brown_spot", "tungro")),
    TransferPair("dhan_shomadhan", "riceleafbd", ("brown_spot", "tungro")),
    TransferPair("riceleafbd", "brri_rice_disease_pest", ("healthy", "tungro")),
    TransferPair("brri_rice_disease_pest", "riceleafbd", ("healthy", "tungro")),
    TransferPair(
        "dhan_shomadhan",
        "brri_rice_disease_pest",
        ("rice_blast", "scald", "tungro"),
    ),
    TransferPair(
        "brri_rice_disease_pest",
        "dhan_shomadhan",
        ("rice_blast", "scald", "tungro"),
    ),
]


KEY_COLUMNS = ["train_dataset", "test_dataset", "model", "classes", "seed"]


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


def _select_pairs(pair_keys: list[str] | None) -> list[TransferPair]:
    if not pair_keys:
        return TRANSFER_PAIRS
    requested = set(pair_keys)
    available = {pair.key for pair in TRANSFER_PAIRS}
    unknown = requested - available
    if unknown:
        raise ValueError(
            f"Unknown transfer pair(s): {sorted(unknown)}. "
            f"Available: {sorted(available)}"
        )
    return [pair for pair in TRANSFER_PAIRS if pair.key in requested]


def _validate_pair_classes(manifest: pd.DataFrame, pair: TransferPair) -> None:
    expected = set(pair.classes)
    for dataset in (pair.train_dataset, pair.test_dataset):
        available = set(
            manifest.loc[manifest["dataset"] == dataset, "mapped_class"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        missing = expected - available
        if missing:
            raise ValueError(
                f"{pair.key} requires classes missing from {dataset}: "
                f"{sorted(missing)}"
            )


def _has_key(df: pd.DataFrame, row: dict[str, object]) -> bool:
    if df.empty or not set(KEY_COLUMNS).issubset(df.columns):
        return False
    mask = pd.Series(True, index=df.index)
    for column in KEY_COLUMNS:
        mask &= df[column].astype(str) == str(row[column])
    return bool(mask.any())


def _upsert_csv(path: Path, row: dict[str, object]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_csv(path)
        if set(KEY_COLUMNS).issubset(df.columns):
            mask = pd.Series(True, index=df.index)
            for column in KEY_COLUMNS:
                mask &= df[column].astype(str) == str(row[column])
            df = df.loc[~mask].copy()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)
    return df


def _print_summary(matrix_path: Path, gap_path: Path) -> None:
    if not matrix_path.exists() or not gap_path.exists():
        print("Cross-dataset outputs are not complete yet.")
        return

    matrix = pd.read_csv(matrix_path)
    gaps = pd.read_csv(gap_path)
    print("\n=== Cross-dataset results ===")
    print(
        matrix.sort_values(["model", "train_dataset", "test_dataset"]).to_string(
            index=False
        )
    )
    print("\n=== Generalization gaps ===")
    print(
        gaps.sort_values(["model", "train_dataset", "test_dataset"]).to_string(
            index=False
        )
    )

    if matrix.empty:
        return
    print("\n=== Mean cross-domain macro-F1 matrix (models pooled) ===")
    pivot = matrix.pivot_table(
        index="train_dataset",
        columns="test_dataset",
        values="macro_f1",
        aggfunc="mean",
    )
    print(pivot.to_string(float_format=lambda value: f"{value:.4f}"))

    print("\n=== Mean by model ===")
    model_summary = (
        matrix.groupby("model", as_index=False)
        .agg(
            cross_macro_f1=("macro_f1", "mean"),
            cross_accuracy=("accuracy", "mean"),
        )
        .merge(
            gaps.groupby("model", as_index=False).agg(
                mean_gap=("generalization_gap", "mean")
            ),
            on="model",
            how="left",
        )
    )
    print(model_summary.to_string(index=False))


def run_crossdataset(
    config_path: str = "config.yaml",
    models: list[str] | None = None,
    pair_keys: list[str] | None = None,
    seeds: list[int] | None = None,
    image_roots: dict[str, str] | None = None,
    skip_existing: bool = True,
    augmentation: str = "default",
    matrix_filename: str | None = None,
    gap_filename: str | None = None,
) -> tuple[Path, Path]:
    config = load_config(config_path)
    results_root = Path(config["results_root"])
    manifest_path = results_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Prepare Kaggle paths first "
            "(see notebooks/kaggle_week5.md)."
        )

    manifest = pd.read_csv(manifest_path)
    models = models or DEFAULT_MODELS
    pairs = _select_pairs(pair_keys)
    seeds = seeds or [
        int(seed)
        for seed in config.get("seeds", [config.get("seed", 42)])
    ]

    is_augmented = augmentation != "default"
    default_matrix = (
        "crossdataset_matrix_aug.csv"
        if is_augmented
        else "crossdataset_matrix.csv"
    )
    default_gap = (
        "generalization_gap_aug.csv" if is_augmented else "generalization_gap.csv"
    )
    matrix_path = results_root / (matrix_filename or default_matrix)
    gap_path = results_root / (gap_filename or default_gap)

    for pair in pairs:
        _validate_pair_classes(manifest, pair)
        classes = list(pair.classes)
        for model_name in models:
            for seed in seeds:
                key_row: dict[str, object] = {
                    "train_dataset": pair.train_dataset,
                    "test_dataset": pair.test_dataset,
                    "model": model_name,
                    "classes": pair.class_string,
                    "seed": seed,
                }
                effective_run_tag = (
                    f"{pair.run_tag}__aug-{augmentation}"
                    if is_augmented
                    else pair.run_tag
                )
                matrix_df = (
                    pd.read_csv(matrix_path)
                    if matrix_path.exists()
                    else pd.DataFrame()
                )
                gap_df = (
                    pd.read_csv(gap_path) if gap_path.exists() else pd.DataFrame()
                )
                if (
                    skip_existing
                    and _has_key(matrix_df, key_row)
                    and _has_key(gap_df, key_row)
                ):
                    print(
                        "SKIP completed: "
                        f"{model_name} | {pair.key} | seed={seed}"
                    )
                    continue

                artifact_stem = training_stem(
                    model_name,
                    [pair.train_dataset],
                    seed,
                    effective_run_tag,
                )
                checkpoint_path = (
                    results_root / "checkpoints" / f"{artifact_stem}.pth"
                )

                if checkpoint_path.exists():
                    print(
                        "\nREUSE checkpoint after interruption: "
                        f"{checkpoint_path}"
                    )
                else:
                    print(
                        f"\n=== TRAIN {model_name}: {pair.train_dataset} "
                        f"-> {pair.test_dataset} (seed={seed}) ==="
                    )
                    print(f"shared classes: {classes}")
                    checkpoint_path, _ = train(
                        model_name=model_name,
                        train_datasets=[pair.train_dataset],
                        classes=classes,
                        config=config,
                        seed=seed,
                        eval_dataset=pair.train_dataset,
                        image_roots=image_roots,
                        run_tag=effective_run_tag,
                        augmentation=augmentation,
                    )

                print(
                    f"=== REFERENCE EVAL {pair.train_dataset} "
                    f"({pair.class_string}) ==="
                )
                reference = evaluate(
                    checkpoint_path=checkpoint_path,
                    eval_dataset=pair.train_dataset,
                    classes=classes,
                    split="test",
                    seed=seed,
                    config_path=config_path,
                    sample_n=5,
                    image_roots=image_roots,
                )

                print(
                    f"=== CROSS EVAL {pair.train_dataset} "
                    f"-> {pair.test_dataset} ==="
                )
                cross = evaluate(
                    checkpoint_path=checkpoint_path,
                    eval_dataset=pair.test_dataset,
                    classes=classes,
                    split="test",
                    seed=seed,
                    config_path=config_path,
                    sample_n=10,
                    image_roots=image_roots,
                )

                if (
                    reference["class_to_index"]
                    != cross["class_to_index"]
                ):
                    raise AssertionError(
                        "Silent class-index swap detected: "
                        f"reference={reference['class_to_index']}, "
                        f"cross={cross['class_to_index']}"
                    )

                matrix_row = {
                    **key_row,
                    "augmentation": augmentation,
                    "accuracy": cross["accuracy"],
                    "macro_f1": cross["macro_f1"],
                    "n_samples": cross["n_samples"],
                    "checkpoint_path": str(checkpoint_path),
                }
                gap_row = {
                    **key_row,
                    "augmentation": augmentation,
                    "in_dataset_accuracy": reference["accuracy"],
                    "in_dataset_macro_f1": reference["macro_f1"],
                    "cross_accuracy": cross["accuracy"],
                    "cross_macro_f1": cross["macro_f1"],
                    "generalization_gap": (
                        reference["macro_f1"] - cross["macro_f1"]
                    ),
                    "in_dataset_n": reference["n_samples"],
                    "cross_n": cross["n_samples"],
                    "checkpoint_path": str(checkpoint_path),
                }
                _upsert_csv(matrix_path, matrix_row)
                _upsert_csv(gap_path, gap_row)
                print(
                    "RECORDED: "
                    f"{model_name} | {pair.key} | "
                    f"same={reference['macro_f1']:.4f} | "
                    f"cross={cross['macro_f1']:.4f} | "
                    f"gap={gap_row['generalization_gap']:.4f}"
                )

    _print_summary(matrix_path, gap_path)
    print(f"\nWrote: {matrix_path}")
    print(f"Wrote: {gap_path}")
    return matrix_path, gap_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 5 ordered cross-dataset transfer matrix."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Ordered pairs as train:test (e.g. riceleafbd:dhan_shomadhan).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--image_roots",
        nargs="*",
        default=None,
        help="dataset=/abs/path pairs for Kaggle mounts.",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-run combinations already present in both output CSVs.",
    )
    parser.add_argument(
        "--augmentation",
        default="default",
        choices=["default", "strong"],
        help="Train-time augmentation profile. 'strong' is the Week 7 mitigation.",
    )
    parser.add_argument(
        "--matrix_filename",
        default=None,
        help="Override matrix CSV filename inside results_root.",
    )
    parser.add_argument(
        "--gap_filename",
        default=None,
        help="Override gap CSV filename inside results_root.",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only print existing matrix and gap summaries.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["results_root"])
    if args.summary_only:
        is_aug = args.augmentation != "default"
        matrix_name = args.matrix_filename or (
            "crossdataset_matrix_aug.csv"
            if is_aug
            else "crossdataset_matrix.csv"
        )
        gap_name = args.gap_filename or (
            "generalization_gap_aug.csv"
            if is_aug
            else "generalization_gap.csv"
        )
        _print_summary(root / matrix_name, root / gap_name)
    else:
        run_crossdataset(
            config_path=args.config,
            models=args.models,
            pair_keys=args.pairs,
            seeds=args.seeds,
            image_roots=_parse_image_roots(args.image_roots),
            skip_existing=not args.no_skip_existing,
            augmentation=args.augmentation,
            matrix_filename=args.matrix_filename,
            gap_filename=args.gap_filename,
        )
