from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.data_loader import (
    ManifestImageDataset,
    default_eval_transform,
    resolve_manifest_paths,
)
from src.eval import collect_predictions, load_checkpoint
from src.gradcam import generate_gradcam_grid
from src.train import build_model, train, training_stem
from src.utils import get_device, load_config, set_seed


DIAGNOSIS_MODEL = "resnet50"
TRAIN_DATASET = "dhan_shomadhan"
EVAL_DATASET = "riceleafbd"
DIAGNOSIS_CLASSES = ["brown_spot", "tungro"]
RUN_TAG = "to-riceleafbd__classes-brown_spot+tungro"


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


def ensure_diagnostic_checkpoint(
    config: dict[str, Any],
    seed: int,
    image_roots: dict[str, str] | None,
    checkpoint_path: str | None,
) -> Path:
    """Reuse the Week 5 Dhan-to-Rice checkpoint or retrain only that run."""
    if checkpoint_path:
        supplied = Path(checkpoint_path)
        if not supplied.exists():
            raise FileNotFoundError(f"Checkpoint not found: {supplied}")
        return supplied

    stem = training_stem(
        DIAGNOSIS_MODEL, [TRAIN_DATASET], seed, RUN_TAG
    )
    expected = Path(config["results_root"]) / "checkpoints" / f"{stem}.pth"
    if expected.exists():
        print(f"Reusing Week 5 checkpoint: {expected}")
        return expected

    print(
        "Week 5 checkpoint is unavailable; retraining only the representative "
        "ResNet50 Dhan-to-Rice shared-class model."
    )
    trained_path, _ = train(
        model_name=DIAGNOSIS_MODEL,
        train_datasets=[TRAIN_DATASET],
        classes=DIAGNOSIS_CLASSES,
        config=config,
        seed=seed,
        eval_dataset=TRAIN_DATASET,
        image_roots=image_roots,
        run_tag=RUN_TAG,
    )
    return trained_path


def _make_condition_loader(
    rows: pd.DataFrame,
    class_to_index: dict[str, int],
    config: dict[str, Any],
) -> DataLoader:
    if rows.empty:
        raise ValueError("Diagnosis condition has zero rows.")
    present = set(rows["mapped_class"].astype(str).unique())
    missing = set(class_to_index) - present
    if missing:
        raise ValueError(
            f"Diagnosis condition is missing classes: {sorted(missing)}"
        )
    dataset = ManifestImageDataset(
        rows,
        class_to_index,
        default_eval_transform(int(config.get("image_size", 224))),
        verify_images=False,
    )
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=0,
    )


def _condition_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: list[str],
) -> dict[str, Any]:
    _, labels, predictions = collect_predictions(model, loader, device)
    per_class_values = f1_score(
        labels,
        predictions,
        labels=list(range(len(classes))),
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "n_samples": len(labels),
        "per_class_f1": {
            class_name: float(per_class_values[index])
            for index, class_name in enumerate(classes)
        },
    }


def run_background_confound(
    checkpoint_path: str | Path,
    config_path: str = "config.yaml",
    seed: int = 42,
    image_roots: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Evaluate the same Dhan-trained model across background conditions."""
    set_seed(seed)
    config = load_config(config_path)
    results_root = Path(config["results_root"])
    manifest = pd.read_csv(results_root / "manifest.csv")
    manifest = resolve_manifest_paths(manifest, image_roots=image_roots)

    checkpoint = load_checkpoint(Path(checkpoint_path), get_device())
    classes = list(checkpoint["classes"])
    if classes != DIAGNOSIS_CLASSES:
        raise ValueError(
            "Background diagnosis expects the Dhan-to-Rice shared classes "
            f"{DIAGNOSIS_CLASSES}, got {classes}."
        )
    class_to_index = dict(checkpoint["class_to_index"])
    expected_mapping = {
        name: index for index, name in enumerate(DIAGNOSIS_CLASSES)
    }
    if class_to_index != expected_mapping:
        raise AssertionError(
            f"Unexpected diagnosis class mapping: {class_to_index}"
        )

    duplicate_mask = ~manifest["is_duplicate"].astype(bool)
    shared_mask = manifest["mapped_class"].isin(classes)
    test_mask = manifest["split"].eq("test")
    dhan_mask = manifest["dataset"].eq(TRAIN_DATASET)
    rice_mask = manifest["dataset"].eq(EVAL_DATASET)

    conditions = {
        "dhan_field": manifest[
            duplicate_mask
            & shared_mask
            & test_mask
            & dhan_mask
            & manifest["background"].eq("field")
        ].copy(),
        "dhan_white": manifest[
            duplicate_mask
            & shared_mask
            & test_mask
            & dhan_mask
            & manifest["background"].eq("white")
        ].copy(),
        "riceleafbd_field": manifest[
            duplicate_mask & shared_mask & test_mask & rice_mask
        ].copy(),
    }

    device = get_device()
    model = build_model(
        str(checkpoint["model_name"]), len(classes), pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records: list[dict[str, Any]] = []
    for condition, rows in conditions.items():
        loader = _make_condition_loader(rows, class_to_index, config)
        metrics = _condition_metrics(model, loader, device, classes)
        records.append(
            {
                "condition": condition,
                "model": str(checkpoint["model_name"]),
                "train_dataset": TRAIN_DATASET,
                "classes": "|".join(classes),
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "n_samples": metrics["n_samples"],
                "per_class_f1": json.dumps(metrics["per_class_f1"]),
            }
        )
        print(
            f"{condition}: n={metrics['n_samples']} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    results = pd.DataFrame(records)
    csv_path = results_root / "background_confound.csv"
    results.to_csv(csv_path, index=False)

    figure_path = Path("paper/figures/background_confound.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    axis = sns.barplot(
        data=results,
        x="condition",
        y="macro_f1",
        hue="condition",
        palette="colorblind",
        legend=False,
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Evaluation condition")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Background-confound evaluation (Dhan-trained ResNet50)")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", padding=3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()

    score_by_condition = results.set_index("condition")["macro_f1"]
    white_score = float(score_by_condition["dhan_white"])
    field_score = float(score_by_condition["dhan_field"])
    cross_score = float(score_by_condition["riceleafbd_field"])
    print(
        "Background-confound hypothesis "
        + (
            "SUPPORTED by ordering white > field and white > cross."
            if white_score > field_score and white_score > cross_score
            else "NOT SUPPORTED by the prespecified ordering."
        )
    )
    print(
        "Interpret together with Grad-CAM overlays; this comparison is "
        "descriptive and does not by itself prove causality."
    )
    print(f"Background results: {csv_path}")
    print(f"Background figure: {figure_path}")
    return csv_path, figure_path


def run_diagnosis(
    config_path: str = "config.yaml",
    checkpoint_path: str | None = None,
    sample_size: int = 12,
    seed: int = 42,
    image_roots: dict[str, str] | None = None,
) -> None:
    config = load_config(config_path)
    checkpoint = ensure_diagnostic_checkpoint(
        config, seed, image_roots, checkpoint_path
    )
    generate_gradcam_grid(
        checkpoint_path=checkpoint,
        eval_dataset=EVAL_DATASET,
        config_path=config_path,
        sample_size=sample_size,
        seed=seed,
        image_roots=image_roots,
        output_path="paper/figures/gradcam_examples.png",
    )
    run_background_confound(
        checkpoint_path=checkpoint,
        config_path=config_path,
        seed=seed,
        image_roots=image_roots,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 6 Grad-CAM and background-confound diagnosis."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional ResNet50 Dhan-to-Rice Week 5 checkpoint.",
    )
    parser.add_argument("--sample_size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image_roots",
        nargs="*",
        default=None,
        help="dataset=/abs/path pairs for Kaggle mounts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diagnosis(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        sample_size=args.sample_size,
        seed=args.seed,
        image_roots=_parse_image_roots(args.image_roots),
    )

