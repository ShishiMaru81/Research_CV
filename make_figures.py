from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "frozen_results"
DEFAULT_FIGURES = ROOT / "paper" / "figures"
DEFAULT_TABLES = ROOT / "paper" / "tables"

DATASETS = ["riceleafbd", "dhan_shomadhan", "brri_rice_disease_pest"]
MODELS = ["mobilenetv2_100", "efficientnet_b0", "resnet50"]
DATASET_LABELS = {
    "riceleafbd": "RiceLeafBD",
    "dhan_shomadhan": "Dhan-Shomadhan",
    "brri_rice_disease_pest": "BRRI",
}
MODEL_LABELS = {
    "mobilenetv2_100": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet50",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", metadata={"Software": "Research_CV"})
    plt.close(fig)


def _grouped_bar(
    ax: plt.Axes,
    values: np.ndarray,
    group_labels: list[str],
    series_labels: list[str],
    *,
    ylabel: str = "Macro-F1",
) -> None:
    colors = sns.color_palette("colorblind", n_colors=len(series_labels))
    x = np.arange(len(group_labels))
    width = 0.8 / len(series_labels)
    offsets = (np.arange(len(series_labels)) - (len(series_labels) - 1) / 2) * width
    for index, (label, color) in enumerate(zip(series_labels, colors)):
        bars = ax.bar(x + offsets[index], values[:, index], width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x, group_labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False)


def figure_indataset(results: Path, output: Path) -> None:
    df = pd.read_csv(results / "indataset_results.csv")
    pivot = df.pivot(index="dataset", columns="model", values="macro_f1").reindex(
        index=DATASETS, columns=MODELS
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    _grouped_bar(
        ax,
        pivot.to_numpy(),
        [DATASET_LABELS[item] for item in DATASETS],
        [MODEL_LABELS[item] for item in MODELS],
    )
    ax.set_title("In-dataset test performance")
    fig.tight_layout()
    _save(fig, output)


def _three_model_heatmaps(
    df: pd.DataFrame,
    value: str,
    title: str,
    output: Path,
    *,
    center: float | None = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharex=True, sharey=True)
    for ax, model in zip(axes, MODELS):
        matrix = (
            df[df["model"] == model]
            .pivot(index="train_dataset", columns="test_dataset", values=value)
            .reindex(index=DATASETS, columns=DATASETS)
        )
        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap="vlag" if center is not None else "Blues",
            vmin=-0.5 if center is not None else 0,
            vmax=0.5 if center is not None else 1,
            center=center,
            square=True,
            linewidths=0.5,
            cbar=ax is axes[-1],
            mask=matrix.isna(),
        )
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Test dataset")
        ax.set_ylabel("Train dataset" if ax is axes[0] else "")
        ax.set_xticklabels([DATASET_LABELS[item] for item in DATASETS], rotation=30)
        ax.set_yticklabels([DATASET_LABELS[item] for item in DATASETS], rotation=0)
    fig.suptitle(title, y=1.03, fontweight="bold")
    fig.tight_layout()
    _save(fig, output)


def figure_background(results: Path, output: Path) -> None:
    df = pd.read_csv(results / "background_confound.csv").set_index("condition")
    order = ["dhan_white", "dhan_field", "riceleafbd_field"]
    labels = ["Dhan white", "Dhan field", "RiceLeafBD field"]
    values = df.loc[order, "macro_f1"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(labels, values, color=sns.color_palette("colorblind", 3))
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Background-confound evaluation")
    fig.tight_layout()
    _save(fig, output)


def figure_aug_delta(results: Path, output: Path) -> None:
    df = pd.read_csv(results / "mitigation_pairwise_aug.csv").copy()
    pair_order = [
        (source, target)
        for source in DATASETS
        for target in DATASETS
        if source != target
    ]
    labels = [
        f"{DATASET_LABELS[source]} → {DATASET_LABELS[target]}"
        for source, target in pair_order
    ]
    df["pair"] = list(zip(df["train_dataset"], df["test_dataset"]))
    matrix = (
        df.pivot(index="pair", columns="model", values="cross_f1_improvement")
        .reindex(index=pair_order, columns=MODELS)
    )
    matrix.index = labels
    matrix.columns = [MODEL_LABELS[item] for item in MODELS]
    bound = max(0.45, float(np.nanmax(np.abs(matrix.to_numpy()))))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="+.3f",
        cmap="vlag",
        center=0,
        vmin=-bound,
        vmax=bound,
        linewidths=0.5,
        cbar_kws={"label": "Change in cross-domain macro-F1"},
        ax=ax,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Transfer direction")
    ax.set_title("Effect of strong augmentation")
    fig.tight_layout()
    _save(fig, output)


def figure_strategy(results: Path, output: Path) -> None:
    df = pd.read_csv(results / "mitigation_comparison.csv")
    columns = [
        "baseline_single_source_macro_f1",
        "aug_single_source_macro_f1",
        "lodo_macro_f1",
    ]
    labels = ["Baseline", "Strong augmentation", "LODO"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        subset = (
            df[df["model"] == model]
            .set_index("target_dataset")
            .reindex(DATASETS)
        )
        _grouped_bar(
            ax,
            subset[columns].to_numpy(),
            [DATASET_LABELS[item] for item in DATASETS],
            labels,
        )
        ax.set_title(MODEL_LABELS[model])
        ax.tick_params(axis="x", rotation=30)
        if ax is not axes[0]:
            ax.set_ylabel("")
            ax.get_legend().remove()
    fig.suptitle("Mitigation strategies by held-out target", y=1.03, fontweight="bold")
    fig.tight_layout()
    _save(fig, output)


def figure_lodo(results: Path, output: Path) -> None:
    df = pd.read_csv(results / "lodo_results.csv")
    pivot = df.pivot(
        index="held_out_dataset", columns="model", values="macro_f1"
    ).reindex(index=DATASETS, columns=MODELS)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    _grouped_bar(
        ax,
        pivot.to_numpy(),
        [DATASET_LABELS[item] for item in DATASETS],
        [MODEL_LABELS[item] for item in MODELS],
    )
    ax.set_title("Leave-one-dataset-out performance")
    ax.set_xlabel("Held-out dataset")
    fig.tight_layout()
    _save(fig, output)


def _write_table(df: pd.DataFrame, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    tex_path = stem.with_suffix(".tex")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    tex_path.write_text(
        df.to_latex(index=False, float_format=lambda value: f"{value:.3f}"),
        encoding="utf-8",
    )
    return [csv_path, tex_path]


def _summary_statistics(results: Path) -> pd.DataFrame:
    pairwise = pd.read_csv(results / "mitigation_pairwise_aug.csv")
    comparison = pd.read_csv(results / "mitigation_comparison.csv")
    lodo = pd.read_csv(results / "lodo_results.csv")
    rows = [
        ("baseline_mean_cross_macro_f1", pairwise["baseline_cross_macro_f1"].mean()),
        ("aug_mean_cross_macro_f1", pairwise["aug_cross_macro_f1"].mean()),
        ("mean_cross_f1_improvement", pairwise["cross_f1_improvement"].mean()),
        ("baseline_mean_gap", pairwise["baseline_gap"].mean()),
        ("aug_mean_gap", pairwise["aug_gap"].mean()),
        ("mean_gap_reduction", pairwise["gap_reduction"].mean()),
        ("augmentation_positive_pairs", float((pairwise["cross_f1_improvement"] > 0).sum())),
        ("gap_reduction_positive_pairs", float((pairwise["gap_reduction"] > 0).sum())),
        ("lodo_positive_target_model_cells", float((comparison["lodo_vs_single_source"] > 0).sum())),
    ]
    for dataset in DATASETS:
        mean_value = lodo.loc[lodo["held_out_dataset"] == dataset, "macro_f1"].mean()
        rows.append((f"lodo_mean_{dataset}", mean_value))
    return pd.DataFrame(rows, columns=["statistic", "value"])


def generate_tables(results: Path, tables: Path) -> list[Path]:
    outputs: list[Path] = []
    specs: list[tuple[str, pd.DataFrame]] = [
        ("table_indataset", pd.read_csv(results / "indataset_results.csv")),
        (
            "table_transfer_baseline",
            pd.read_csv(results / "crossdataset_matrix.csv"),
        ),
        ("table_gap_baseline", pd.read_csv(results / "generalization_gap.csv")),
        (
            "table_background_confound",
            pd.read_csv(results / "background_confound.csv"),
        ),
        (
            "table_mitigation_pairwise",
            pd.read_csv(results / "mitigation_pairwise_aug.csv"),
        ),
        (
            "table_mitigation_strategy",
            pd.read_csv(results / "mitigation_comparison.csv"),
        ),
        ("table_lodo", pd.read_csv(results / "lodo_results.csv")),
        ("table_summary_stats", _summary_statistics(results)),
    ]
    for name, frame in specs:
        outputs.extend(_write_table(frame, tables / name))
    return outputs


def generate_all(
    results: Path = DEFAULT_RESULTS,
    figures: Path = DEFAULT_FIGURES,
    tables: Path = DEFAULT_TABLES,
) -> dict[str, object]:
    required = [
        "indataset_results.csv",
        "crossdataset_matrix.csv",
        "generalization_gap.csv",
        "background_confound.csv",
        "crossdataset_matrix_aug.csv",
        "generalization_gap_aug.csv",
        "lodo_results.csv",
        "mitigation_pairwise_aug.csv",
        "mitigation_comparison.csv",
    ]
    for name in required:
        if not (results / name).exists():
            raise FileNotFoundError(
                f"Missing frozen input {results / name}. Run python -m freeze_results."
            )

    _configure_style()
    figure_jobs: list[tuple[str, Callable[[Path, Path], None]]] = [
        ("fig01_indataset_macro_f1.png", figure_indataset),
        (
            "fig02_crossdataset_heatmap_baseline.png",
            lambda source, output: _three_model_heatmaps(
                pd.read_csv(source / "crossdataset_matrix.csv"),
                "macro_f1",
                "Baseline cross-dataset macro-F1",
                output,
            ),
        ),
        (
            "fig03_generalization_gap_baseline.png",
            lambda source, output: _three_model_heatmaps(
                pd.read_csv(source / "generalization_gap.csv"),
                "generalization_gap",
                "Baseline generalization gap",
                output,
            ),
        ),
        ("fig04_background_confound.png", figure_background),
        (
            "fig05_crossdataset_heatmap_aug.png",
            lambda source, output: _three_model_heatmaps(
                pd.read_csv(source / "crossdataset_matrix_aug.csv"),
                "macro_f1",
                "Strong-augmentation cross-dataset macro-F1",
                output,
            ),
        ),
        (
            "fig06_generalization_gap_aug.png",
            lambda source, output: _three_model_heatmaps(
                pd.read_csv(source / "generalization_gap_aug.csv"),
                "generalization_gap",
                "Strong-augmentation generalization gap",
                output,
            ),
        ),
        ("fig07_augmentation_f1_delta.png", figure_aug_delta),
        ("fig08_mitigation_by_target.png", figure_strategy),
        ("fig09_lodo_heldout.png", figure_lodo),
    ]
    figure_outputs: list[Path] = []
    for filename, job in figure_jobs:
        output = figures / filename
        job(results, output)
        figure_outputs.append(output)

    table_outputs = generate_tables(results, tables)
    input_records = [
        {"path": name, "sha256": sha256(results / name)} for name in sorted(required)
    ]
    output_records = [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(path),
        }
        for path in sorted(figure_outputs + table_outputs)
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "inputs": input_records,
        "outputs": output_records,
        "gradcam_overlay": "not regenerated; local checkpoint/image bundle unavailable",
    }
    manifest_path = figures / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(figure_outputs)} figures to {figures}")
    print(f"Wrote {len(table_outputs)} table files to {tables}")
    print(f"Wrote: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic paper figures and tables."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all(args.results, args.figures, args.tables)
