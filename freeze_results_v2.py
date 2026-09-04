"""Audit and freeze Week 10-14 result artifacts without touching the v1 freeze.

This command is intentionally an auditor, not a result generator.  It first
verifies every v1 CSV against the immutable Week-8 SHA-256 manifest, then
validates and hashes every CSV already present in the requested v2 directory.
Only after all checks pass does it atomically write the v2 manifest and report.

New result CSVs should be assembled in a staging directory and audited there
before they are promoted into ``frozen_results_v2/``.  The command never clears
or silently rebuilds either frozen directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
V1_DIR = ROOT / "frozen_results"
DEFAULT_OUT = ROOT / "frozen_results_v2"
DEFAULT_V1_MANIFEST = V1_DIR / "freeze_manifest.json"
MANIFEST_NAME = "freeze_manifest_v2.json"
REPORT_NAME = "audit_report_v2.md"

BASE_REQUIRED_WEEK10_14 = {
    "deployment_profile.csv",
    "gradcam_negative_summary.csv",
    "dinov2_indataset.csv",
    "dinov2_crossdataset.csv",
    "sam_mask_quality.csv",
    "crossdataset_matrix_masked_sam_leaf.csv",
    "adabn_results.csv",
    "adabn_labelshift.csv",
}
OPTIONAL_WEEK10_14 = {
    "hsv_mask_quality.csv",
    "crossdataset_matrix_masked_hsv_leaf.csv",
}
MASK_CONDITIONS = {
    "sam_leaf": ("sam_mask_quality.csv", "crossdataset_matrix_masked_sam_leaf.csv"),
    "hsv_leaf": ("hsv_mask_quality.csv", "crossdataset_matrix_masked_hsv_leaf.csv"),
}
MASKED_KEY_COLUMNS = [
    "train_dataset",
    "test_dataset",
    "model",
    "classes",
    "seed",
]

# Human-readable provenance for artifacts introduced after the prior v2 freeze.
KNOWN_SOURCES = {
    "dinov2_indataset.csv": "dinov2_indataset.csv",
    "dinov2_crossdataset.csv": "dinov2_crossdataset.csv",
    "sam_mask_quality.csv": "sam_mask_quality.csv",
    "hsv_mask_quality.csv": "hsv_mask_quality.csv",
    "crossdataset_matrix_masked_sam_leaf.csv": "crossdataset_matrix_masked_sam_leaf.csv",
    "crossdataset_matrix_masked_hsv_leaf.csv": "crossdataset_matrix_masked_hsv_leaf.csv",
    "adabn_labelshift.csv": "week14_results/adabn_labelshift.csv",
}


class FreezeError(RuntimeError):
    """Raised when an integrity, arithmetic, or completeness check fails."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FreezeError(f"{label} is missing required columns: {missing}")


def _numeric(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    try:
        values = pd.to_numeric(frame[column], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise FreezeError(f"{label}.{column} is not numeric: {exc}") from exc
    if not np.isfinite(values.to_numpy()).all():
        raise FreezeError(f"{label}.{column} contains NaN or infinite values.")
    return values


def _check_range(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> int:
    checked = 0
    for column in columns:
        values = _numeric(frame, column, label)
        invalid = ~values.between(lower, upper, inclusive="both")
        if invalid.any():
            indices = frame.index[invalid].tolist()[:5]
            raise FreezeError(
                f"{label}.{column} has values outside [{lower}, {upper}] "
                f"at rows {indices}."
            )
        checked += len(values)
    return checked


def _check_nonnegative(frame: pd.DataFrame, columns: Iterable[str], label: str) -> int:
    checked = 0
    for column in columns:
        values = _numeric(frame, column, label)
        invalid = values < 0
        if invalid.any():
            indices = frame.index[invalid].tolist()[:5]
            raise FreezeError(
                f"{label}.{column} has negative values at rows {indices}."
            )
        checked += len(values)
    return checked


def _check_positive_integer(frame: pd.DataFrame, column: str, label: str) -> int:
    values = _numeric(frame, column, label)
    invalid = (values <= 0) | (values != np.floor(values))
    if invalid.any():
        indices = frame.index[invalid].tolist()[:5]
        raise FreezeError(
            f"{label}.{column} must contain positive integers; bad rows {indices}."
        )
    return len(values)


def _check_nonnegative_integer(frame: pd.DataFrame, column: str, label: str) -> int:
    values = _numeric(frame, column, label)
    invalid = (values < 0) | (values != np.floor(values))
    if invalid.any():
        indices = frame.index[invalid].tolist()[:5]
        raise FreezeError(
            f"{label}.{column} must contain non-negative integers; bad rows {indices}."
        )
    return len(values)


def _duplicate_examples(
    frame: pd.DataFrame, keys: Sequence[str]
) -> list[dict[str, Any]]:
    duplicated = frame.duplicated(list(keys), keep=False)
    return frame.loc[duplicated, list(keys)].head().to_dict("records")


def _key_set(frame: pd.DataFrame, keys: Sequence[str]) -> set[tuple[Any, ...]]:
    return set(frame.loc[:, list(keys)].itertuples(index=False, name=None))


def _format_masked_key(key: tuple[Any, ...], condition: str) -> str:
    train, test, model, classes, seed = key
    return f"{train}->{test}|{model}|{classes}|seed={seed}|{condition}"


def _assert_paths_do_not_overlap(v1_dir: Path, v2_dir: Path) -> None:
    v1_resolved = v1_dir.resolve()
    v2_resolved = v2_dir.resolve()
    if (
        v1_resolved == v2_resolved
        or v1_resolved in v2_resolved.parents
        or v2_resolved in v1_resolved.parents
    ):
        raise FreezeError(
            f"Refusing overlapping v1/v2 paths: v1={v1_resolved}, v2={v2_resolved}"
        )


def verify_v1_integrity(v1_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify the exact v1 CSV set and hashes before any v2 output is written.

    Failure construction: changing one byte, removing a listed file, or adding an
    unlisted CSV makes this check fail.
    """

    if not v1_dir.is_dir():
        raise FreezeError(f"Missing v1 directory: {v1_dir}")
    if not manifest_path.is_file():
        raise FreezeError(f"Missing Week-8 freeze manifest: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError(f"Cannot parse {manifest_path}: {exc}") from exc
    records = payload.get("files")
    if not isinstance(records, list) or not records:
        raise FreezeError("Week-8 manifest has no nonempty 'files' list.")

    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise FreezeError("Week-8 manifest contains a non-object file record.")
        name = str(record.get("path", ""))
        digest = str(record.get("sha256", ""))
        if not name.endswith(".csv") or Path(name).name != name:
            raise FreezeError(f"Invalid v1 CSV path in manifest: {name!r}")
        if name in expected:
            raise FreezeError(f"Duplicate v1 file record in manifest: {name}")
        if len(digest) != 64:
            raise FreezeError(f"Invalid SHA-256 recorded for v1 file: {name}")
        expected[name] = record

    actual_names = {path.name for path in v1_dir.glob("*.csv") if path.is_file()}
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise FreezeError(
            f"v1 CSV set differs from Week-8 manifest; missing={missing}, extra={extra}."
        )

    verified: list[dict[str, Any]] = []
    for name in sorted(expected):
        path = v1_dir / name
        actual_hash = sha256(path)
        recorded_hash = str(expected[name]["sha256"]).lower()
        if actual_hash.lower() != recorded_hash:
            raise FreezeError(
                f"V1 INTEGRITY FAILURE: {name} SHA-256 is {actual_hash}, "
                f"expected {recorded_hash}. Aborting before v2 freeze."
            )
        try:
            row_count = len(pd.read_csv(path))
        except Exception as exc:  # pandas exposes several parser exception types
            raise FreezeError(f"Cannot read v1 CSV {path}: {exc}") from exc
        recorded_rows = expected[name].get("rows")
        if recorded_rows is not None and row_count != int(recorded_rows):
            raise FreezeError(
                f"V1 INTEGRITY FAILURE: {name} has {row_count} rows, "
                f"expected {recorded_rows}."
            )
        verified.append({"path": name, "rows": row_count, "sha256": actual_hash})

    return {
        "files_checked": len(verified),
        "all_hashes_match": True,
        "files": verified,
        "baseline_manifest": str(manifest_path),
    }


def _load_all_v2_csvs(v2_dir: Path) -> dict[str, pd.DataFrame]:
    if not v2_dir.is_dir():
        raise FreezeError(f"Missing v2 directory: {v2_dir}")
    paths = sorted(path for path in v2_dir.glob("*.csv") if path.is_file())
    if not paths:
        raise FreezeError(f"No CSV files found in {v2_dir}.")
    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        try:
            frames[path.name] = pd.read_csv(path)
        except Exception as exc:
            raise FreezeError(f"Cannot read v2 CSV {path}: {exc}") from exc
    return frames


def _require_week10_14_artifacts(frames: dict[str, pd.DataFrame]) -> set[str]:
    missing = sorted(BASE_REQUIRED_WEEK10_14 - set(frames))
    if missing:
        raise FreezeError(f"Missing required Week 10-14 v2 artifacts: {missing}")

    present = set(BASE_REQUIRED_WEEK10_14)
    for condition, (quality_name, matrix_name) in MASK_CONDITIONS.items():
        has_quality = quality_name in frames
        has_matrix = matrix_name in frames
        if has_quality != has_matrix:
            raise FreezeError(
                f"Condition {condition} is only partially frozen: "
                f"quality_present={has_quality}, matrix_present={has_matrix}."
            )
        if has_quality:
            present.update({quality_name, matrix_name})
    return present


def _validate_mask_quality(
    frame: pd.DataFrame, manifest: pd.DataFrame, label: str
) -> list[str]:
    required = {
        "image_path",
        "dataset",
        "foreground_fraction",
        "n_components",
        "largest_component_fraction",
    }
    _require_columns(frame, required, label)
    normalize = lambda value: str(value).replace("\\", "/")
    normalized_paths = frame["image_path"].map(normalize)
    if normalized_paths.duplicated().any():
        raise FreezeError(f"{label} contains duplicate normalized image_path values.")
    expected_paths = {normalize(value) for value in manifest["image_path"]}
    actual_paths = set(normalized_paths)
    missing = expected_paths - actual_paths
    extra = actual_paths - expected_paths
    if missing or extra:
        raise FreezeError(
            f"{label} image coverage differs from v1 manifest: "
            f"missing={len(missing)}, extra={len(extra)}."
        )
    expected_dataset = {
        normalize(row.image_path): str(row.dataset)
        for row in manifest.itertuples(index=False)
    }
    mismatched_dataset = [
        path
        for path, dataset in zip(normalized_paths, frame["dataset"].astype(str))
        if expected_dataset.get(path) != dataset
    ]
    if mismatched_dataset:
        raise FreezeError(
            f"{label} dataset labels disagree with v1 manifest at "
            f"{mismatched_dataset[:5]}"
        )
    _check_range(
        frame,
        ["foreground_fraction", "largest_component_fraction"],
        label,
    )
    _check_nonnegative_integer(frame, "n_components", label)
    if "sam_score" in frame.columns:
        # Some saved MobileSAM scores exceed one; finiteness, not [0,1], is the valid check.
        _numeric(frame, "sam_score", label)
    return [
        f"{label}: {len(frame)} unique image paths match the v1 manifest",
        f"{label}: mask fractions and component counts are valid",
    ]


def _validate_deployment(frame: pd.DataFrame, expected_models: set[str]) -> list[str]:
    name = "deployment_profile.csv"
    required = {
        "model",
        "n_params",
        "n_trainable",
        "gflops",
        "cpu_latency_ms_mean",
        "cpu_latency_ms_std",
        "model_size_mb",
    }
    _require_columns(frame, required, name)
    if (
        set(frame["model"].astype(str)) != expected_models
        or frame["model"].duplicated().any()
    ):
        raise FreezeError(
            f"{name} model coverage is not exactly {sorted(expected_models)}."
        )
    _check_positive_integer(frame, "n_params", name)
    _check_nonnegative_integer(frame, "n_trainable", name)
    trainable = _numeric(frame, "n_trainable", name)
    params = _numeric(frame, "n_params", name)
    if (trainable > params).any():
        raise FreezeError(f"{name}.n_trainable exceeds n_params.")
    for column in ("gflops", "cpu_latency_ms_mean", "model_size_mb"):
        _check_nonnegative(frame, [column], name)
    _check_nonnegative(frame, ["cpu_latency_ms_std"], name)
    return [f"{name}: schema, model coverage, and resource metrics are valid"]


GRADCAM_METRICS = {
    "n_total",
    "n_correct",
    "mean_enrichment_correct",
    "std_enrichment_correct",
    "n_incorrect",
    "mean_enrichment_incorrect",
    "std_enrichment_incorrect",
    "mannwhitney_u_statistic",
    "mannwhitney_p_value",
    "n_enrichment_below_1p0",
    "frac_enrichment_below_1p0",
    "n_label_brown_spot",
    "mean_enrichment_label_brown_spot",
    "n_label_tungro",
    "mean_enrichment_label_tungro",
    "crosstab_correct_False_below1_False",
    "crosstab_correct_False_below1_True",
    "crosstab_correct_True_below1_False",
    "crosstab_correct_True_below1_True",
}


def _validate_gradcam_summary(frame: pd.DataFrame) -> list[str]:
    name = "gradcam_negative_summary.csv"
    _require_columns(frame, {"metric", "value"}, name)
    if (
        frame["metric"].duplicated().any()
        or set(frame["metric"].astype(str)) != GRADCAM_METRICS
    ):
        raise FreezeError(f"{name} metric inventory is incomplete or duplicated.")
    values = {
        str(row.metric): float(row.value) for row in frame.itertuples(index=False)
    }
    if not np.isfinite(list(values.values())).all():
        raise FreezeError(f"{name} contains NaN or infinite values.")
    count_metrics = [
        metric
        for metric in GRADCAM_METRICS
        if metric.startswith("n_") or metric.startswith("crosstab_")
    ]
    for metric in count_metrics:
        value = values[metric]
        if value < 0 or value != np.floor(value):
            raise FreezeError(f"{name}.{metric} must be a non-negative integer.")
    probability_metrics = [
        "mannwhitney_p_value",
        "frac_enrichment_below_1p0",
    ]
    for metric in probability_metrics:
        if not 0 <= values[metric] <= 1:
            raise FreezeError(f"{name}.{metric} must be in [0,1].")
    if values["n_correct"] + values["n_incorrect"] != values["n_total"]:
        raise FreezeError(f"{name}: correct + incorrect does not equal n_total.")
    if values["n_enrichment_below_1p0"] > values["n_total"]:
        raise FreezeError(f"{name}: below-one count exceeds n_total.")
    if not np.isclose(
        values["frac_enrichment_below_1p0"],
        values["n_enrichment_below_1p0"] / values["n_total"],
        atol=1e-5,
    ):
        raise FreezeError(f"{name}: below-one fraction is arithmetically inconsistent.")
    if values["n_label_brown_spot"] + values["n_label_tungro"] != values["n_total"]:
        raise FreezeError(f"{name}: label counts do not equal n_total.")
    if (
        sum(
            values[metric]
            for metric in GRADCAM_METRICS
            if metric.startswith("crosstab_")
        )
        != values["n_total"]
    ):
        raise FreezeError(f"{name}: crosstab counts do not equal n_total.")
    return [f"{name}: 19 metrics and internal count/fraction arithmetic are valid"]


def _validate_dinov2(
    frames: dict[str, pd.DataFrame], baseline: pd.DataFrame
) -> list[str]:
    checks: list[str] = []
    in_name = "dinov2_indataset.csv"
    cross_name = "dinov2_crossdataset.csv"
    indataset = frames[in_name]
    cross = frames[cross_name]
    _require_columns(
        indataset,
        {"model", "dataset", "seed", "accuracy", "macro_f1", "n_samples"},
        in_name,
    )
    _require_columns(
        cross,
        {
            "model",
            "train_dataset",
            "test_dataset",
            "classes",
            "seed",
            "accuracy",
            "macro_f1",
            "n_samples",
        },
        cross_name,
    )
    for name, frame, keys in (
        (in_name, indataset, ["model", "dataset", "seed"]),
        (
            cross_name,
            cross,
            ["model", "train_dataset", "test_dataset", "classes", "seed"],
        ),
    ):
        duplicates = _duplicate_examples(frame, keys)
        if duplicates:
            raise FreezeError(f"{name} has duplicate keys: {duplicates}")
        metric_columns = ["accuracy", "macro_f1"]
        if "val_macro_f1" in frame.columns:
            metric_columns.append("val_macro_f1")
        _check_range(frame, metric_columns, name)
        _check_positive_integer(frame, "n_samples", name)
        if set(frame["model"].astype(str)) != {"dinov2"}:
            raise FreezeError(f"{name} contains models other than dinov2.")
        checks.append(f"{name}: metric ranges, sample counts, and unique keys pass")

    pair_columns = ["train_dataset", "test_dataset", "classes"]
    expected_pairs = _key_set(baseline, pair_columns)
    datasets = set(baseline["train_dataset"]) | set(baseline["test_dataset"])
    for seed, group in cross.groupby("seed", sort=True):
        found_pairs = _key_set(group, pair_columns)
        if found_pairs != expected_pairs:
            raise FreezeError(
                f"{cross_name} seed={seed} pair coverage differs from the v1 baseline."
            )
    for seed, group in indataset.groupby("seed", sort=True):
        if set(group["dataset"]) != datasets:
            raise FreezeError(
                f"{in_name} seed={seed} dataset coverage differs from the v1 baseline."
            )
    if set(cross["seed"]) != set(indataset["seed"]):
        raise FreezeError("DINOv2 in-dataset and cross-dataset seed sets differ.")
    checks.append("DINOv2 pair/dataset coverage is complete for every saved seed")
    return checks


def _validate_adabn_labelshift(frames: dict[str, pd.DataFrame]) -> list[str]:
    name = "adabn_labelshift.csv"
    frame = frames[name]
    required = set(MASKED_KEY_COLUMNS) | {
        "delta_macro_f1",
        "kl_sym",
        "tv_distance",
        "chi2",
        "bn_layers",
    }
    _require_columns(frame, required, name)
    duplicates = _duplicate_examples(frame, MASKED_KEY_COLUMNS)
    if duplicates:
        raise FreezeError(f"{name} has duplicate experiment keys: {duplicates}")
    _check_nonnegative(frame, ["kl_sym", "chi2"], name)
    _check_range(frame, ["tv_distance"], name)
    _check_positive_integer(frame, "bn_layers", name)
    _numeric(frame, "delta_macro_f1", name)
    if "n_target_adapt" in frame.columns:
        _check_positive_integer(frame, "n_target_adapt", name)
    if "n_source_train" in frame.columns:
        _check_positive_integer(frame, "n_source_train", name)

    checks = [
        f"{name}: divergence metrics and BN counts are valid for all {len(frame)} rows"
    ]
    if "adabn_results.csv" in frames:
        source = frames["adabn_results.csv"]
        _require_columns(
            source, set(MASKED_KEY_COLUMNS) | {"delta_macro_f1"}, "adabn_results.csv"
        )
        if _key_set(frame, MASKED_KEY_COLUMNS) != _key_set(source, MASKED_KEY_COLUMNS):
            raise FreezeError(f"{name} experiment keys differ from adabn_results.csv.")
        left = frame[MASKED_KEY_COLUMNS + ["delta_macro_f1"]]
        right = source[MASKED_KEY_COLUMNS + ["delta_macro_f1"]]
        merged = left.merge(
            right, on=MASKED_KEY_COLUMNS, suffixes=("_analysis", "_source")
        )
        if len(merged) != len(frame):
            raise FreezeError(
                f"{name} could not be matched one-to-one to adabn_results.csv."
            )
        if not np.allclose(
            _numeric(merged, "delta_macro_f1_analysis", name),
            _numeric(merged, "delta_macro_f1_source", "adabn_results.csv"),
            rtol=0.0,
            atol=1e-12,
        ):
            raise FreezeError(f"{name} delta values differ from adabn_results.csv.")
        checks.append(
            f"{name}: experiment keys and delta values match the frozen AdaBN results"
        )
    return checks


def _validate_masked_completeness(
    frames: dict[str, pd.DataFrame], baseline: pd.DataFrame
) -> tuple[dict[str, Any], list[str]]:
    _require_columns(baseline, set(MASKED_KEY_COLUMNS), "v1 crossdataset_matrix.csv")
    _require_columns(baseline, {"n_samples"}, "v1 crossdataset_matrix.csv")
    baseline_duplicates = _duplicate_examples(baseline, MASKED_KEY_COLUMNS)
    if baseline_duplicates:
        raise FreezeError(
            f"v1 crossdataset_matrix.csv has duplicate experiment keys: {baseline_duplicates}"
        )
    expected_keys = _key_set(baseline, MASKED_KEY_COLUMNS)
    condition_results: dict[str, Any] = {}
    checks: list[str] = []
    total_found = 0

    for condition, (quality_name, matrix_name) in MASK_CONDITIONS.items():
        if quality_name not in frames and matrix_name not in frames:
            condition_results[condition] = {
                "status": "not_run",
                "expected": 0,
                "found": 0,
                "missing": [],
                "extra": [],
            }
            continue
        if quality_name not in frames or matrix_name not in frames:
            raise FreezeError(
                f"Incomplete artifact pair for mask condition {condition}."
            )

        frame = frames[matrix_name]
        required = set(MASKED_KEY_COLUMNS) | {
            "condition",
            "accuracy",
            "macro_f1",
            "n_samples",
        }
        _require_columns(frame, required, matrix_name)
        conditions = set(frame["condition"].astype(str))
        if conditions != {condition}:
            raise FreezeError(
                f"{matrix_name}.condition is {sorted(conditions)}, expected only {condition!r}."
            )
        duplicates = _duplicate_examples(frame, MASKED_KEY_COLUMNS + ["condition"])
        if duplicates:
            raise FreezeError(
                f"{matrix_name} has duplicate experiment keys: {duplicates}"
            )
        _check_range(frame, ["accuracy", "macro_f1"], matrix_name)
        _check_positive_integer(frame, "n_samples", matrix_name)

        found_keys = _key_set(frame, MASKED_KEY_COLUMNS)
        missing = sorted(
            expected_keys - found_keys, key=lambda key: tuple(map(str, key))
        )
        extra = sorted(found_keys - expected_keys, key=lambda key: tuple(map(str, key)))
        condition_results[condition] = {
            "status": "complete" if not missing and not extra else "incomplete",
            "expected": len(expected_keys),
            "found": len(found_keys),
            "missing": [_format_masked_key(key, condition) for key in missing],
            "extra": [_format_masked_key(key, condition) for key in extra],
        }
        if missing or extra:
            raise FreezeError(
                f"{matrix_name} completeness failure: "
                f"missing={len(missing)}, extra={len(extra)}."
            )
        expected_samples = baseline.set_index(MASKED_KEY_COLUMNS)["n_samples"]
        actual_samples = frame.set_index(MASKED_KEY_COLUMNS)["n_samples"]
        aligned_expected = pd.to_numeric(
            expected_samples.reindex(actual_samples.index), errors="raise"
        )
        aligned_actual = pd.to_numeric(actual_samples, errors="raise")
        if not np.array_equal(aligned_expected.to_numpy(), aligned_actual.to_numpy()):
            raise FreezeError(
                f"{matrix_name}.n_samples differs from the immutable baseline evaluation populations."
            )
        total_found += len(found_keys)
        checks.append(
            f"{matrix_name}: {len(found_keys)}/{len(expected_keys)} exact masked-transfer keys"
        )

    expected_total = sum(item["expected"] for item in condition_results.values())
    completeness = {
        "crossdataset_masked_pairs": {
            "expected": expected_total,
            "found": total_found,
            "missing": [
                item
                for details in condition_results.values()
                for item in details["missing"]
            ],
            "extra": [
                item
                for details in condition_results.values()
                for item in details["extra"]
            ],
            "conditions": condition_results,
        }
    }
    return completeness, checks


def _validate_v2_core_against_v1(
    v2_dir: Path, frames: dict[str, pd.DataFrame], v1_integrity: dict[str, Any]
) -> list[str]:
    """Ensure every v2 copy of a v1 CSV is byte-identical to the verified v1 file."""

    checks: list[str] = []
    for record in v1_integrity["files"]:
        name = str(record["path"])
        path = v2_dir / name
        if name not in frames or not path.is_file():
            raise FreezeError(f"v2 is missing the immutable v1 core artifact {name}.")
        actual_hash = sha256(path)
        if actual_hash != str(record["sha256"]):
            raise FreezeError(
                f"V2 CORE INTEGRITY FAILURE: {name} differs from frozen_results/ "
                f"(v2={actual_hash}, v1={record['sha256']})."
            )
        if len(frames[name]) != int(record["rows"]):
            raise FreezeError(f"v2 core row count differs from v1 for {name}.")
        checks.append(f"{name}: byte-identical to verified v1 core")
    return checks


def validate_v2(
    frames: dict[str, pd.DataFrame],
    v1_dir: Path,
    v1_integrity: dict[str, Any],
    v2_dir: Path,
) -> tuple[dict[str, Any], list[str], set[str]]:
    expected_present = _require_week10_14_artifacts(frames)
    baseline = pd.read_csv(v1_dir / "crossdataset_matrix.csv")
    manifest = pd.read_csv(v1_dir / "manifest.csv")
    checks: list[str] = []

    checks.extend(_validate_v2_core_against_v1(v2_dir, frames, v1_integrity))
    expected_models = set(baseline["model"].astype(str))
    checks.extend(
        _validate_deployment(frames["deployment_profile.csv"], expected_models)
    )
    checks.extend(_validate_gradcam_summary(frames["gradcam_negative_summary.csv"]))
    checks.extend(_validate_dinov2(frames, baseline))
    checks.extend(_validate_adabn_labelshift(frames))
    for condition, (quality_name, _matrix_name) in MASK_CONDITIONS.items():
        if quality_name in frames:
            checks.extend(
                _validate_mask_quality(frames[quality_name], manifest, quality_name)
            )
            checks.append(f"{condition}: mask-quality artifact is present")
    completeness, completeness_checks = _validate_masked_completeness(frames, baseline)
    checks.extend(completeness_checks)
    checks.append("All specified arithmetic checks passed without filtering rows")
    return completeness, checks, expected_present


def _previous_sources(v2_dir: Path) -> dict[str, str]:
    path = v2_dir / MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    sources: dict[str, str] = {}
    for record in payload.get("files", []):
        if isinstance(record, dict) and record.get("path") and record.get("source"):
            sources[str(record["path"])] = str(record["source"])
    for name, record in payload.get("v2_results", {}).items():
        if isinstance(record, dict) and record.get("source"):
            sources[str(name)] = str(record["source"])
    return sources


def _inventory_v2(
    v2_dir: Path,
    frames: dict[str, pd.DataFrame],
    expected_present: set[str],
    v1_names: set[str],
) -> dict[str, dict[str, Any]]:
    previous_sources = _previous_sources(v2_dir)
    inventory: dict[str, dict[str, Any]] = {}
    for name in sorted(frames):
        frame = frames[name]
        if name in BASE_REQUIRED_WEEK10_14:
            expectation = "required"
        elif name in OPTIONAL_WEEK10_14:
            expectation = "optional_present"
        else:
            expectation = "additional_frozen_artifact"
        record: dict[str, Any] = {
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "sha256": sha256(v2_dir / name),
            "expected": expectation,
            "status": (
                "AUDITED" if name in expected_present or name in v1_names else "HASHED"
            ),
        }
        source = KNOWN_SOURCES.get(name) or previous_sources.get(name)
        if source:
            record["source"] = source
        inventory[name] = record

    missing_expected = expected_present - set(inventory)
    if missing_expected:  # defensive: validation should already make this impossible
        raise FreezeError(
            f"Expected artifacts escaped inventory: {sorted(missing_expected)}"
        )
    return inventory


def _inventory_auxiliary(v2_dir: Path) -> dict[str, dict[str, Any]]:
    excluded = {MANIFEST_NAME, REPORT_NAME, "AUDIT_REPORT_v2.md"}
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in v2_dir.iterdir() if item.is_file()):
        if path.name in excluded or path.suffix.lower() == ".csv":
            continue
        inventory[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "status": "PASS",
        }
    return inventory


def _mask_audit_gate_status() -> dict[str, str]:
    decision_path = ROOT / "notes" / "mask_audit" / "audit_decision.md"
    relative_path = "notes/mask_audit/audit_decision.md"
    if not decision_path.is_file():
        return {
            "status": "unverified_missing",
            "path": relative_path,
            "reason": "No human mask-audit decision file is present in the repository.",
        }
    decision_text = decision_path.read_text(encoding="utf-8")
    cleared_line = next(
        (
            line
            for line in decision_text.splitlines()
            if line.startswith("**Variants cleared for Week 13:**")
        ),
        "",
    )
    cleared_variants = {
        variant.strip()
        for variant in cleared_line.removeprefix(
            "**Variants cleared for Week 13:**"
        ).split(",")
        if variant.strip()
    }
    required_variants = {"sam_leaf", "hsv_leaf"}
    if required_variants.issubset(cleared_variants):
        return {
            "status": "verified",
            "path": relative_path,
            "reason": (
                "Human audit decision explicitly clears sam_leaf and hsv_leaf "
                "for Week 13."
            ),
        }
    return {
        "status": "unverified_present",
        "path": relative_path,
        "reason": (
            "A decision file exists, but it does not explicitly clear both "
            "sam_leaf and hsv_leaf for Week 13."
        ),
    }


def _atomic_write_outputs(
    manifest_path: Path,
    manifest_text: str,
    report_path: Path,
    report_text: str,
) -> None:
    """Prepare both outputs, then publish the manifest last as the commit marker."""

    manifest_temp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    report_temp = report_path.with_name(f".{report_path.name}.tmp")
    try:
        manifest_temp.write_text(manifest_text, encoding="utf-8", newline="\n")
        report_temp.write_text(report_text, encoding="utf-8", newline="\n")
        os.replace(report_temp, report_path)
        os.replace(manifest_temp, manifest_path)
    finally:
        for temporary in (manifest_temp, report_temp):
            if temporary.exists():
                temporary.unlink()


def _render_report(manifest: dict[str, Any]) -> str:
    v1 = manifest["v1_integrity"]
    completeness = manifest["v2_completeness"]["crossdataset_masked_pairs"]
    lines = [
        "# Week 10-14 Audit Report",
        "",
        f"Frozen at: `{manifest['frozen_at']}`",
        "",
        "## v1 Integrity",
        "",
        (
            f"All {v1['files_checked']} CSVs in `frozen_results/` match the "
            "Week-8 SHA-256 manifest: **PASS**"
        ),
        "",
        "## v2 Artifacts",
        "",
    ]
    for name, record in manifest["v2_results"].items():
        lines.append(
            f"- `{name}` ({record['rows']} rows x {record['columns']} columns): {record['status']}"
        )
    if manifest["auxiliary_results"]:
        lines.extend(["", "### Auxiliary non-CSV artifacts", ""])
        for name, record in manifest["auxiliary_results"].items():
            lines.append(f"- `{name}` ({record['bytes']} bytes): HASHED")

    lines.extend(["", "## v2 Completeness", ""])
    for condition, details in completeness["conditions"].items():
        if details["status"] == "not_run":
            lines.append(f"- `{condition}`: not run; no artifacts frozen (optional).")
        else:
            lines.append(
                f"- `{condition}`: {details['found']}/{details['expected']} exact "
                "pair/model/seed/class keys: PASS"
            )
    lines.extend(
        [
            (
                f"- All present masked conditions: {completeness['found']}/"
                f"{completeness['expected']} keys: PASS"
            ),
            "",
            "## Consistency Checks",
            "",
            *[f"- {check}: PASS" for check in manifest["checks_passed"]],
            "",
        ]
    )
    gate = manifest["mask_audit_gate"]
    if gate["status"] == "verified":
        decision = (
            "All implemented numerical and integrity checks pass. The saved v2 "
            "artifacts are internally consistent, and the human mask-audit gate "
            "is verified by the supplied decision record."
        )
    else:
        decision = (
            "All implemented numerical and integrity checks pass. The saved v2 "
            "artifacts are internally consistent. The human mask-audit gate is "
            "not verified by this repository state, so masked results must not "
            "be described as audit-cleared until its decision record is supplied."
        )
    lines.extend(
        [
            "## Interpretation Guardrails",
            "",
            "- The audit verifies saved-file integrity, arithmetic ranges, and key coverage; it is not an independent model re-run.",
            "- Presence of a mask artifact does not by itself reconstruct a missing human audit verdict.",
            "- The AdaBN label-shift analysis is observational and does not establish causality.",
            f"- Human mask-audit gate: **{gate['status']}** ({gate['reason']})",
            "",
            "## Decision",
            "",
            decision,
            "",
        ]
    )
    return "\n".join(lines)


def audit_and_freeze(
    v2_dir: Path = DEFAULT_OUT,
    *,
    v1_dir: Path = V1_DIR,
    v1_manifest: Path = DEFAULT_V1_MANIFEST,
) -> dict[str, Any]:
    """Run all checks and atomically write the manifest/report only on success."""

    _assert_paths_do_not_overlap(v1_dir, v2_dir)
    alternate_report = v2_dir / "AUDIT_REPORT_v2.md"
    lower_report = v2_dir / REPORT_NAME
    if alternate_report.is_file() and not lower_report.is_file():
        raise FreezeError(
            "Existing audit report uses uppercase AUDIT_REPORT_v2.md. Rename it to "
            "audit_report_v2.md before freezing so two report versions cannot coexist."
        )
    if (
        alternate_report.is_file()
        and lower_report.is_file()
        and alternate_report != lower_report
    ):
        try:
            same_file = alternate_report.samefile(lower_report)
        except OSError:
            same_file = False
        if not same_file:
            raise FreezeError(
                "Both AUDIT_REPORT_v2.md and audit_report_v2.md exist; remove the stale "
                "alias before freezing."
            )

    print("Verifying v1 integrity...")
    v1_integrity = verify_v1_integrity(v1_dir, v1_manifest)
    print(
        f"PASS: {v1_integrity['files_checked']}/{v1_integrity['files_checked']} "
        "v1 CSV hashes match."
    )

    print("Loading and validating every v2 CSV...")
    frames = _load_all_v2_csvs(v2_dir)
    completeness, checks, expected_present = validate_v2(
        frames, v1_dir, v1_integrity, v2_dir
    )
    v1_names = {str(record["path"]) for record in v1_integrity["files"]}
    inventory = _inventory_v2(v2_dir, frames, expected_present, v1_names)
    auxiliary = _inventory_auxiliary(v2_dir)
    total_rows = sum(record["rows"] for record in inventory.values())
    for name, record in inventory.items():
        if sha256(v2_dir / name) != record["sha256"]:
            raise FreezeError(
                f"CSV changed during validation before freeze publication: {name}"
            )

    gate = _mask_audit_gate_status()

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": (
            "PASS_WITH_UNVERIFIED_MASK_GATE" if gate["status"] != "verified" else "PASS"
        ),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "v1_integrity": v1_integrity,
        "v2_results": inventory,
        "auxiliary_results": auxiliary,
        "mask_audit_gate": gate,
        "v2_completeness": completeness,
        "checks_passed": ["v1_integrity", "v2_arithmetic", "v2_completeness", *checks],
        "summary": {
            "csv_files": len(inventory),
            "total_csv_rows": total_rows,
            "week10_14_artifacts_present": sorted(expected_present),
        },
    }
    report = _render_report(manifest)
    manifest_path = v2_dir / MANIFEST_NAME
    report_path = v2_dir / REPORT_NAME
    _atomic_write_outputs(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        report_path,
        report,
    )

    masked = completeness["crossdataset_masked_pairs"]
    print(f"PASS: v2 artifacts frozen: {len(inventory)} CSVs, {total_rows} total rows")
    print(f"PASS: masked completeness: {masked['found']}/{masked['expected']}")
    print("PASS: arithmetic consistency for every audited row")
    print(f"Freeze manifest: {manifest_path}")
    print(f"Audit report: {report_path}")
    if gate["status"] == "verified":
        print("FINAL STATUS: NUMERICAL FREEZE PASS; HUMAN MASK-AUDIT GATE VERIFIED")
    else:
        print("FINAL STATUS: NUMERICAL FREEZE PASS; HUMAN MASK-AUDIT GATE UNVERIFIED")
    return manifest


# Backward-compatible import name used by the previous script.  Semantics are
# deliberately safer: it audits an assembled directory and never copies or deletes it.
def assemble_v2(output_dir: Path = DEFAULT_OUT) -> dict[str, Any]:
    return audit_and_freeze(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2-dir",
        "--output",
        dest="v2_dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Already-assembled v2 directory to audit and freeze.",
    )
    parser.add_argument("--v1-dir", type=Path, default=V1_DIR)
    parser.add_argument("--v1-manifest", type=Path, default=DEFAULT_V1_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_and_freeze(
        args.v2_dir.resolve(),
        v1_dir=args.v1_dir.resolve(),
        v1_manifest=args.v1_manifest.resolve(),
    )


if __name__ == "__main__":
    main()
