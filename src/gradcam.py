from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data_loader import default_eval_transform, make_loaders
from src.eval import load_checkpoint
from src.train import build_model
from src.utils import get_device, load_config, set_seed


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the final Conv2d module, a robust Grad-CAM target for timm CNNs."""
    last_conv: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return last_conv


@torch.no_grad()
def collect_prediction_records(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    index_to_class: dict[int, str],
    device: torch.device,
) -> pd.DataFrame:
    """Collect path/label/prediction metadata for stratified CAM sampling."""
    model.eval()
    records: list[dict[str, Any]] = []
    for images, labels, paths in loader:
        logits = model(images.to(device))
        predictions = logits.argmax(dim=1).cpu().tolist()
        true_indices = labels.tolist()
        for path, true_idx, pred_idx in zip(
            paths, true_indices, predictions
        ):
            records.append(
                {
                    "image_path": str(path),
                    "true_index": int(true_idx),
                    "pred_index": int(pred_idx),
                    "true_label": index_to_class[int(true_idx)],
                    "pred_label": index_to_class[int(pred_idx)],
                    "correct": bool(true_idx == pred_idx),
                }
            )
    return pd.DataFrame(records)


def stratified_cam_sample(
    records: pd.DataFrame, sample_size: int, seed: int
) -> pd.DataFrame:
    """Sample both correct/incorrect examples across true classes."""
    if records.empty:
        raise ValueError("No prediction records available for Grad-CAM.")
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    targets = {
        False: sample_size // 2,
        True: sample_size - (sample_size // 2),
    }

    for correctness in (False, True):
        if targets[correctness] == 0:
            continue
        subset = records[records["correct"] == correctness]
        if subset.empty:
            continue
        chosen: list[int] = []
        for _, group in subset.groupby("true_label", sort=True):
            chosen.append(int(rng.choice(group.index.to_numpy())))
            if len(chosen) >= targets[correctness]:
                break

        remaining = subset.index.difference(chosen).to_numpy()
        need = min(targets[correctness] - len(chosen), len(remaining))
        if need > 0:
            chosen.extend(
                [int(value) for value in rng.choice(remaining, need, replace=False)]
            )
        selected.extend(chosen)

    if len(selected) < sample_size:
        remaining = records.index.difference(selected).to_numpy()
        need = min(sample_size - len(selected), len(remaining))
        if need > 0:
            selected.extend(
                [int(value) for value in rng.choice(remaining, need, replace=False)]
            )

    return records.loc[selected].reset_index(drop=True)


def _border_mask(
    heatmap: np.ndarray, border_fraction: float
) -> np.ndarray:
    height, width = heatmap.shape
    border_h = max(1, int(height * border_fraction))
    border_w = max(1, int(width * border_fraction))
    mask = np.zeros_like(heatmap, dtype=bool)
    mask[:border_h, :] = True
    mask[-border_h:, :] = True
    mask[:, :border_w] = True
    mask[:, -border_w:] = True
    return mask


def background_attention_metrics(
    heatmap: np.ndarray, border_fraction: float = 0.20
) -> tuple[float, float]:
    """Return border CAM mass and area-normalized enrichment.

    Enrichment is 1.0 for a spatially uniform heatmap, above 1.0 when CAM mass
    is disproportionately near borders, and below 1.0 when it favors the
    center. This remains a background-attention proxy, not lesion segmentation.
    """
    mask = _border_mask(heatmap, border_fraction)
    total = float(heatmap.sum())
    mass_fraction = (
        float(heatmap[mask].sum() / total) if total > 0 else 0.0
    )
    area_fraction = float(mask.mean())
    enrichment = (
        mass_fraction / area_fraction if area_fraction > 0 else 0.0
    )
    return mass_fraction, enrichment


def generate_gradcam_grid(
    checkpoint_path: str | Path,
    eval_dataset: str,
    config_path: str = "config.yaml",
    sample_size: int = 12,
    seed: int = 42,
    image_roots: dict[str, str] | None = None,
    output_path: str | Path = "paper/figures/gradcam_examples.png",
) -> tuple[Path, pd.DataFrame]:
    """Generate a labeled grid of cross-dataset Grad-CAM overlays."""
    set_seed(seed)
    config = load_config(config_path)
    device = get_device()
    checkpoint = load_checkpoint(Path(checkpoint_path), device)
    model_name = str(checkpoint["model_name"])
    train_datasets = list(checkpoint["train_datasets"])
    classes = list(checkpoint["classes"])
    class_to_index = dict(checkpoint["class_to_index"])
    index_to_class = {index: name for name, index in class_to_index.items()}

    _, _, eval_loader, meta = make_loaders(
        train_datasets=train_datasets,
        eval_dataset=eval_dataset,
        classes=classes,
        image_size=int(config.get("image_size", 224)),
        batch_size=int(config.get("batch_size", 32)),
        manifest_path=str(Path(config["results_root"]) / "manifest.csv"),
        image_roots=image_roots,
    )
    if meta["class_to_index"] != class_to_index:
        raise AssertionError(
            "Checkpoint/Grad-CAM loader class mapping mismatch: "
            f"{class_to_index} != {meta['class_to_index']}"
        )

    model = build_model(model_name, len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_records = collect_prediction_records(
        model, eval_loader, index_to_class, device
    )
    selected = stratified_cam_sample(all_records, sample_size, seed)

    target_layer = find_last_conv_layer(model)
    image_size = int(config.get("image_size", 224))
    transform = default_eval_transform(image_size)
    overlays: list[np.ndarray] = []
    border_fractions: list[float] = []
    border_enrichments: list[float] = []

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for _, row in selected.iterrows():
            bgr = cv2.imread(str(row["image_path"]))
            if bgr is None:
                raise ValueError(f"Failed to read image: {row['image_path']}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (image_size, image_size))
            rgb_float = resized.astype(np.float32) / 255.0
            input_tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(int(row["pred_index"]))]
            grayscale_cam = cam(
                input_tensor=input_tensor, targets=targets
            )[0]
            overlays.append(
                show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
            )
            fraction, enrichment = background_attention_metrics(grayscale_cam)
            border_fractions.append(fraction)
            border_enrichments.append(enrichment)

    selected["border_attention_fraction"] = border_fractions
    selected["border_attention_enrichment"] = border_enrichments
    selected["model_name"] = model_name
    selected["train_datasets"] = "+".join(train_datasets)
    selected["eval_dataset"] = eval_dataset

    columns = min(4, max(1, len(overlays)))
    rows = math.ceil(len(overlays) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 4.2 * rows), squeeze=False
    )
    for axis in axes.ravel():
        axis.axis("off")
    for axis, overlay, (_, record) in zip(
        axes.ravel(), overlays, selected.iterrows()
    ):
        status = "correct" if record["correct"] else "wrong"
        axis.imshow(overlay)
        axis.set_title(
            f"T: {record['true_label']}\nP: {record['pred_label']} ({status})\n"
            f"border enrichment={record['border_attention_enrichment']:.2f}",
            fontsize=9,
        )
        axis.axis("off")

    figure.suptitle(
        f"Grad-CAM: train={'+'.join(train_datasets)}, "
        f"test={eval_dataset}, model={model_name}",
        fontsize=13,
    )
    figure.tight_layout()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    records_path = Path(config["results_root"]) / "gradcam_records.csv"
    selected.to_csv(records_path, index=False)
    print(f"Grad-CAM figure: {out_path}")
    print(f"Grad-CAM records: {records_path}")
    return out_path, selected

