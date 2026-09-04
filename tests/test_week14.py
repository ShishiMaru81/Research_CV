"""Focused tests for the Week-14 analysis and freeze audit."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import freeze_results_v2 as freeze
from scripts.analyze_adabn_labelshift import (
    AnalysisError,
    analyze_rows,
    divergence_metrics,
    parse_shared_classes,
)


def test_parse_classes_and_known_divergences() -> None:
    assert parse_shared_classes("healthy|tungro") == ["healthy", "tungro"]
    assert parse_shared_classes("healthy, tungro") == ["healthy", "tungro"]
    with pytest.raises(AnalysisError, match="duplicates"):
        parse_shared_classes("healthy|healthy")

    kl_sym, tv, chi2 = divergence_metrics([0.5, 0.5], [0.5, 0.5])
    assert kl_sym == pytest.approx(0.0)
    assert tv == pytest.approx(0.0)
    assert chi2 == pytest.approx(0.0)

    kl_sym, tv, chi2 = divergence_metrics([0.75, 0.25], [0.25, 0.75])
    assert kl_sym > 0
    assert tv == pytest.approx(0.5)
    assert chi2 > 0
    kl_sym, tv, chi2 = divergence_metrics([1.0, 0.0], [0.0, 1.0])
    assert np.isfinite(kl_sym)
    assert tv == pytest.approx(1.0)
    assert chi2 > 0
    with pytest.raises(AnalysisError, match="epsilon"):
        divergence_metrics([0.5, 0.5], [0.5, 0.5], eps=float("nan"))


def test_analysis_uses_target_train_and_rejects_count_mismatch() -> None:
    manifest = pd.DataFrame(
        {
            "image_path": [f"p{i}.jpg" for i in range(10)],
            "dataset": ["source"] * 4 + ["target"] * 6,
            "mapped_class": ["a", "a", "a", "b", "a", "b", "b", "b", "a", "b"],
            "split": ["train"] * 8 + ["test"] * 2,
            "is_duplicate": [False] * 10,
        }
    )
    adabn = pd.DataFrame(
        [
            {
                "train_dataset": "source",
                "test_dataset": "target",
                "model": "model",
                "classes": "a|b",
                "seed": 42,
                "n_adapt_images": 4,
                "delta_macro_f1": -0.1,
            }
        ]
    )
    result = analyze_rows(adabn, manifest, {"model": 3}, adapt_split="train")
    assert result.loc[0, "target_adapt_split"] == "train"
    assert result.loc[0, "n_target_adapt"] == 4
    assert json.loads(result.loc[0, "prior_tgt"]) == {"a": 0.25, "b": 0.75}

    adabn.loc[0, "n_adapt_images"] = 2
    with pytest.raises(AnalysisError, match="would not reconstruct"):
        analyze_rows(adabn, manifest, {"model": 3}, adapt_split="train")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_minimal_freeze(tmp_path: Path) -> tuple[Path, Path, Path]:
    v1 = tmp_path / "frozen_results"
    v2 = tmp_path / "frozen_results_v2"
    v1.mkdir()
    v2.mkdir()

    manifest_rows = [
        {
            "image_path": "data/source/a.jpg",
            "dataset": "source",
            "original_class": "a",
            "mapped_class": "a",
            "background": "",
            "split": "train",
            "is_duplicate": False,
        },
        {
            "image_path": "data/target/a.jpg",
            "dataset": "target",
            "original_class": "a",
            "mapped_class": "a",
            "background": "",
            "split": "test",
            "is_duplicate": False,
        },
    ]
    baseline_rows = [
        {
            "train_dataset": "source",
            "test_dataset": "target",
            "model": "model",
            "classes": "a",
            "seed": 42,
            "accuracy": 0.5,
            "macro_f1": 0.5,
            "n_samples": 1,
        }
    ]
    _write_csv(v1 / "manifest.csv", manifest_rows)
    _write_csv(v1 / "crossdataset_matrix.csv", baseline_rows)
    records = []
    for name in ("manifest.csv", "crossdataset_matrix.csv"):
        frame = pd.read_csv(v1 / name)
        records.append(
            {"path": name, "rows": len(frame), "sha256": freeze.sha256(v1 / name)}
        )
    v1_manifest = v1 / "freeze_manifest.json"
    v1_manifest.write_text(json.dumps({"files": records}), encoding="utf-8")
    shutil.copy2(v1 / "manifest.csv", v2 / "manifest.csv")
    shutil.copy2(v1 / "crossdataset_matrix.csv", v2 / "crossdataset_matrix.csv")

    _write_csv(
        v2 / "deployment_profile.csv",
        [
            {
                "model": "model",
                "n_params": 1,
                "n_trainable": 1,
                "gflops": 0.1,
                "cpu_latency_ms_mean": 1.0,
                "cpu_latency_ms_std": 0.0,
                "model_size_mb": 1.0,
            }
        ],
    )
    summary = {
        "n_total": 1,
        "n_correct": 1,
        "mean_enrichment_correct": 1.0,
        "std_enrichment_correct": 0.0,
        "n_incorrect": 0,
        "mean_enrichment_incorrect": 0.0,
        "std_enrichment_incorrect": 0.0,
        "mannwhitney_u_statistic": 0.0,
        "mannwhitney_p_value": 1.0,
        "n_enrichment_below_1p0": 0,
        "frac_enrichment_below_1p0": 0.0,
        "n_label_a": 1,
        "mean_enrichment_label_a": 1.0,
        "n_label_b": 0,
        "mean_enrichment_label_b": 0.0,
        "crosstab_correct_False_below1_False": 0,
        "crosstab_correct_False_below1_True": 0,
        "crosstab_correct_True_below1_False": 1,
        "crosstab_correct_True_below1_True": 0,
    }
    # The real summary uses these two labels; keep the minimal fixture aligned
    # with the production schema rather than weakening the validator.
    summary["n_label_brown_spot"] = summary.pop("n_label_a")
    summary["mean_enrichment_label_brown_spot"] = summary.pop("mean_enrichment_label_a")
    summary["n_label_tungro"] = summary.pop("n_label_b")
    summary["mean_enrichment_label_tungro"] = summary.pop("mean_enrichment_label_b")
    _write_csv(
        v2 / "gradcam_negative_summary.csv",
        [{"metric": metric, "value": value} for metric, value in summary.items()],
    )
    _write_csv(
        v2 / "dinov2_indataset.csv",
        [
            {
                "model": "dinov2",
                "dataset": dataset,
                "seed": 42,
                "accuracy": 0.5,
                "macro_f1": 0.5,
                "n_samples": 1,
            }
            for dataset in ("source", "target")
        ],
    )
    _write_csv(
        v2 / "dinov2_crossdataset.csv",
        [
            {
                "model": "dinov2",
                "train_dataset": "source",
                "test_dataset": "target",
                "classes": "a",
                "seed": 42,
                "accuracy": 0.5,
                "macro_f1": 0.5,
                "n_samples": 1,
            }
        ],
    )
    _write_csv(
        v2 / "sam_mask_quality.csv",
        [
            {
                "image_path": row["image_path"],
                "dataset": row["dataset"],
                "foreground_fraction": 0.5,
                "n_components": 1,
                "largest_component_fraction": 0.5,
                "sam_score": 1.01,
            }
            for row in manifest_rows
        ],
    )
    _write_csv(
        v2 / "crossdataset_matrix_masked_sam_leaf.csv",
        [{**baseline_rows[0], "condition": "sam_leaf"}],
    )
    adabn_row = {
        "train_dataset": "source",
        "test_dataset": "target",
        "model": "model",
        "classes": "a",
        "seed": 42,
        "delta_macro_f1": -0.1,
    }
    _write_csv(v2 / "adabn_results.csv", [adabn_row])
    _write_csv(
        v2 / "adabn_labelshift.csv",
        [
            {
                **adabn_row,
                "kl_sym": 0.0,
                "tv_distance": 0.0,
                "chi2": 0.0,
                "bn_layers": 1,
                "n_source_train": 1,
                "n_target_adapt": 1,
            }
        ],
    )
    _write_csv(v2 / "extra.csv", [{"kept": 1}])
    return v1, v1_manifest, v2


def test_v1_tamper_aborts_before_v2_writes(tmp_path: Path) -> None:
    v1, v1_manifest, v2 = _build_minimal_freeze(tmp_path)
    with (v1 / "manifest.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(freeze.FreezeError, match="V1 INTEGRITY FAILURE"):
        freeze.audit_and_freeze(v2, v1_dir=v1, v1_manifest=v1_manifest)
    assert not (v2 / freeze.MANIFEST_NAME).exists()
    assert not (v2 / freeze.REPORT_NAME).exists()


def test_freeze_hashes_every_csv_and_validates_exact_keys(tmp_path: Path) -> None:
    v1, v1_manifest, v2 = _build_minimal_freeze(tmp_path)
    manifest = freeze.audit_and_freeze(v2, v1_dir=v1, v1_manifest=v1_manifest)

    assert set(manifest["v2_results"]) == {path.name for path in v2.glob("*.csv")}
    assert manifest["v2_results"]["extra.csv"]["rows"] == 1
    assert manifest["v2_completeness"]["crossdataset_masked_pairs"]["found"] == 1
    assert manifest["v2_completeness"]["crossdataset_masked_pairs"]["expected"] == 1
    assert (
        manifest["v2_completeness"]["crossdataset_masked_pairs"]["conditions"][
            "hsv_leaf"
        ]["status"]
        == "not_run"
    )
    assert (v2 / freeze.MANIFEST_NAME).is_file()
    assert (v2 / freeze.REPORT_NAME).is_file()


def test_invalid_masked_metric_fails_without_writing_freeze(tmp_path: Path) -> None:
    v1, v1_manifest, v2 = _build_minimal_freeze(tmp_path)
    masked_path = v2 / "crossdataset_matrix_masked_sam_leaf.csv"
    masked = pd.read_csv(masked_path)
    masked.loc[0, "accuracy"] = np.inf
    masked.to_csv(masked_path, index=False)

    with pytest.raises(freeze.FreezeError, match="NaN or infinite"):
        freeze.audit_and_freeze(v2, v1_dir=v1, v1_manifest=v1_manifest)
    assert not (v2 / freeze.MANIFEST_NAME).exists()
    assert not (v2 / freeze.REPORT_NAME).exists()


def test_v2_core_mutation_and_missing_adabn_source_fail(tmp_path: Path) -> None:
    v1, v1_manifest, v2 = _build_minimal_freeze(tmp_path)
    with (v2 / "manifest.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(freeze.FreezeError, match="V2 CORE INTEGRITY FAILURE"):
        freeze.audit_and_freeze(v2, v1_dir=v1, v1_manifest=v1_manifest)

    # Restore the core copy, then remove the provenance source required by the
    # AdaBN label-shift audit.
    shutil.copy2(v1 / "manifest.csv", v2 / "manifest.csv")
    (v2 / "adabn_results.csv").unlink()
    with pytest.raises(freeze.FreezeError, match="Missing required"):
        freeze.audit_and_freeze(v2, v1_dir=v1, v1_manifest=v1_manifest)
