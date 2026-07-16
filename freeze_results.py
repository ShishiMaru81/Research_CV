from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from run_mitigation import build_comparison_by_target, build_pairwise_augmentation


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "frozen_results"

SOURCES = {
    "manifest.csv": ROOT / "artifacts" / "manifest.csv",
    "indataset_results.csv": ROOT / "week4_results" / "indataset_results.csv",
    "crossdataset_matrix.csv": ROOT / "week5_progress" / "crossdataset_matrix.csv",
    "generalization_gap.csv": ROOT / "week5_progress" / "generalization_gap.csv",
    "background_confound.csv": (
        ROOT / "week6_results" / "results" / "background_confound.csv"
    ),
    "gradcam_records.csv": (
        ROOT / "week6_results" / "results" / "gradcam_records.csv"
    ),
    "crossdataset_matrix_aug.csv": (
        ROOT / "week7_results" / "crossdataset_matrix_aug.csv"
    ),
    "generalization_gap_aug.csv": (
        ROOT / "week7_results" / "generalization_gap_aug.csv"
    ),
    "lodo_results.csv": ROOT / "week7_results" / "lodo_results.csv",
    "mitigation_pairwise_aug.csv": (
        ROOT / "week7_results" / "mitigation_pairwise_aug.csv"
    ),
    "mitigation_comparison.csv": (
        ROOT / "week7_results" / "mitigation_comparison.csv"
    ),
}

EXPECTED_ROWS = {
    "manifest.csv": 5419,
    "indataset_results.csv": 9,
    "crossdataset_matrix.csv": 18,
    "generalization_gap.csv": 18,
    "background_confound.csv": 3,
    "gradcam_records.csv": 12,
    "crossdataset_matrix_aug.csv": 18,
    "generalization_gap_aug.csv": 18,
    "lodo_results.csv": 9,
    "mitigation_pairwise_aug.csv": 18,
    "mitigation_comparison.csv": 9,
}

PAIR_KEYS = ["train_dataset", "test_dataset", "model", "classes", "seed"]
MODELS = {"mobilenetv2_100", "efficientnet_b0", "resnet50"}
DATASETS = {"riceleafbd", "dhan_shomadhan", "brri_rice_disease_pest"}

EXPECTED_TEST_N = {
    ("riceleafbd", "brown_spot|tungro"): 133,
    ("dhan_shomadhan", "brown_spot|tungro"): 51,
    ("riceleafbd", "healthy|tungro"): 117,
    ("brri_rice_disease_pest", "healthy|tungro"): 127,
    ("dhan_shomadhan", "rice_blast|scald|tungro"): 103,
    ("brri_rice_disease_pest", "rice_blast|scald|tungro"): 204,
}
EXPECTED_INDATASET_N = {
    "riceleafbd": 235,
    "dhan_shomadhan": 167,
    "brri_rice_disease_pest": 414,
}
EXPECTED_LODO_N = {
    "riceleafbd": 171,
    "dhan_shomadhan": 124,
    "brri_rice_disease_pest": 295,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_columns(df: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise AssertionError(f"{name} missing columns: {sorted(missing)}")


def _assert_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    duplicated = df.duplicated(keys, keep=False)
    if duplicated.any():
        rows = df.loc[duplicated, keys].to_dict("records")
        raise AssertionError(f"{name} has duplicate keys: {rows[:5]}")


def _assert_metric_range(df: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        values = pd.to_numeric(df[column], errors="raise")
        if not values.between(0.0, 1.0).all():
            raise AssertionError(f"{name}.{column} contains values outside [0, 1]")


def _assert_close(actual: float, expected: float, context: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-9):
        raise AssertionError(f"{context}: {actual} != {expected}")


def _metric_path(
    folder: Path, checkpoint_path: str, eval_dataset: str, seed: int
) -> Path:
    stem = Path(checkpoint_path).stem
    stem = re.sub(r"__seed\d+$", "", stem)
    return folder / f"{stem}__eval-{eval_dataset}__seed{seed}__metrics.json"


def _check_metric_json(
    path: Path,
    *,
    accuracy: float,
    macro_f1: float,
    n_samples: int,
    context: str,
) -> None:
    if not path.exists():
        raise AssertionError(f"{context}: missing metrics JSON {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _assert_close(float(payload["accuracy"]), float(accuracy), f"{context} accuracy")
    _assert_close(
        float(payload["macro_f1"]), float(macro_f1), f"{context} macro_f1"
    )
    if int(payload["n_samples"]) != int(n_samples):
        raise AssertionError(
            f"{context} n_samples: {payload['n_samples']} != {n_samples}"
        )


def _manifest_identity(path: Path) -> Counter[tuple[str, ...]]:
    df = pd.read_csv(path).fillna("")
    required = {
        "image_path",
        "dataset",
        "original_class",
        "mapped_class",
        "background",
        "split",
        "is_duplicate",
    }
    _assert_columns(df, required, str(path))
    records: Counter[tuple[str, ...]] = Counter()
    for row in df.itertuples(index=False):
        image_path = str(getattr(row, "image_path")).replace("\\", "/")
        records[
            (
                str(getattr(row, "dataset")),
                str(getattr(row, "original_class")),
                str(getattr(row, "mapped_class")),
                str(getattr(row, "background")),
                str(getattr(row, "split")),
                str(getattr(row, "is_duplicate")).lower(),
                image_path.rsplit("/", 1)[-1],
            )
        ] += 1
    return records


def _validate_manifest_archives(checks: list[str]) -> None:
    canonical = _manifest_identity(SOURCES["manifest.csv"])
    for path in [
        ROOT / "week3_results" / "manifest.csv",
        ROOT / "week4_results" / "manifest.csv",
        ROOT / "week5_progress" / "manifest.csv",
    ]:
        if _manifest_identity(path) != canonical:
            raise AssertionError(f"Manifest split/label identity differs: {path}")
        checks.append(f"Manifest identity matches canonical: {path.relative_to(ROOT)}")


def _validate_frames(frames: dict[str, pd.DataFrame], checks: list[str]) -> None:
    manifest = frames["manifest.csv"]
    _assert_unique(manifest, ["image_path"], "manifest.csv")
    checks.append("Canonical manifest has 5,419 unique image paths")

    indataset = frames["indataset_results.csv"]
    _assert_columns(
        indataset,
        {"model", "dataset", "seed", "accuracy", "macro_f1", "n_samples"},
        "indataset_results.csv",
    )
    _assert_unique(indataset, ["model", "dataset", "seed"], "indataset_results.csv")
    _assert_metric_range(indataset, ["accuracy", "macro_f1"], "indataset_results.csv")
    if set(indataset["model"]) != MODELS or set(indataset["dataset"]) != DATASETS:
        raise AssertionError("In-dataset model/dataset coverage is incomplete")
    for row in indataset.itertuples(index=False):
        if int(row.n_samples) != EXPECTED_INDATASET_N[row.dataset]:
            raise AssertionError(f"Unexpected in-dataset n for {row.dataset}")
        _check_metric_json(
            _metric_path(
                ROOT / "week4_results", row.checkpoint_path, row.dataset, row.seed
            ),
            accuracy=row.accuracy,
            macro_f1=row.macro_f1,
            n_samples=row.n_samples,
            context=f"Week 4 {row.model}/{row.dataset}",
        )
    checks.append("All 9 in-dataset rows match their metrics JSON artifacts")

    for matrix_name, gap_name, artifact_dir in [
        (
            "crossdataset_matrix.csv",
            "generalization_gap.csv",
            ROOT / "week5_progress",
        ),
        (
            "crossdataset_matrix_aug.csv",
            "generalization_gap_aug.csv",
            ROOT / "week7_results",
        ),
    ]:
        matrix = frames[matrix_name]
        gap = frames[gap_name]
        _assert_columns(
            matrix,
            set(PAIR_KEYS) | {"accuracy", "macro_f1", "n_samples", "checkpoint_path"},
            matrix_name,
        )
        _assert_columns(
            gap,
            set(PAIR_KEYS)
            | {
                "in_dataset_accuracy",
                "in_dataset_macro_f1",
                "cross_accuracy",
                "cross_macro_f1",
                "generalization_gap",
                "in_dataset_n",
                "cross_n",
                "checkpoint_path",
            },
            gap_name,
        )
        _assert_unique(matrix, PAIR_KEYS, matrix_name)
        _assert_unique(gap, PAIR_KEYS, gap_name)
        _assert_metric_range(matrix, ["accuracy", "macro_f1"], matrix_name)
        _assert_metric_range(
            gap,
            [
                "in_dataset_accuracy",
                "in_dataset_macro_f1",
                "cross_accuracy",
                "cross_macro_f1",
                "generalization_gap",
            ],
            gap_name,
        )
        matrix_keys = set(map(tuple, matrix[PAIR_KEYS].to_numpy()))
        gap_keys = set(map(tuple, gap[PAIR_KEYS].to_numpy()))
        if matrix_keys != gap_keys:
            raise AssertionError(f"{matrix_name} and {gap_name} keys differ")
        for row in gap.itertuples(index=False):
            expected_gap = row.in_dataset_macro_f1 - row.cross_macro_f1
            _assert_close(row.generalization_gap, expected_gap, f"{gap_name} gap")
            expected_cross_n = EXPECTED_TEST_N[(row.test_dataset, row.classes)]
            expected_source_n = EXPECTED_TEST_N[(row.train_dataset, row.classes)]
            if int(row.cross_n) != expected_cross_n:
                raise AssertionError(f"{gap_name}: unexpected cross_n")
            if int(row.in_dataset_n) != expected_source_n:
                raise AssertionError(f"{gap_name}: unexpected in_dataset_n")
            _check_metric_json(
                _metric_path(
                    artifact_dir,
                    row.checkpoint_path,
                    row.train_dataset,
                    row.seed,
                ),
                accuracy=row.in_dataset_accuracy,
                macro_f1=row.in_dataset_macro_f1,
                n_samples=row.in_dataset_n,
                context=f"{gap_name} source evaluation",
            )
            _check_metric_json(
                _metric_path(
                    artifact_dir,
                    row.checkpoint_path,
                    row.test_dataset,
                    row.seed,
                ),
                accuracy=row.cross_accuracy,
                macro_f1=row.cross_macro_f1,
                n_samples=row.cross_n,
                context=f"{gap_name} cross evaluation",
            )
        checks.append(f"All 18 rows in {gap_name} pass arithmetic and JSON checks")

    baseline_keys = set(
        map(tuple, frames["crossdataset_matrix.csv"][PAIR_KEYS].to_numpy())
    )
    augmented_keys = set(
        map(tuple, frames["crossdataset_matrix_aug.csv"][PAIR_KEYS].to_numpy())
    )
    if baseline_keys != augmented_keys:
        raise AssertionError("Baseline and augmented transfer keys differ")
    checks.append("Baseline and augmented transfer keys align 18/18")

    lodo = frames["lodo_results.csv"]
    lodo_keys = ["held_out_dataset", "model", "seed"]
    _assert_unique(lodo, lodo_keys, "lodo_results.csv")
    _assert_metric_range(lodo, ["accuracy", "macro_f1"], "lodo_results.csv")
    if set(lodo["model"]) != MODELS or set(lodo["held_out_dataset"]) != DATASETS:
        raise AssertionError("LODO model/target coverage is incomplete")
    for row in lodo.itertuples(index=False):
        if int(row.n_samples) != EXPECTED_LODO_N[row.held_out_dataset]:
            raise AssertionError(f"Unexpected LODO n for {row.held_out_dataset}")
        _check_metric_json(
            _metric_path(
                ROOT / "week7_results",
                row.checkpoint_path,
                row.held_out_dataset,
                row.seed,
            ),
            accuracy=row.accuracy,
            macro_f1=row.macro_f1,
            n_samples=row.n_samples,
            context=f"LODO {row.model}/{row.held_out_dataset}",
        )
    checks.append("All 9 LODO rows match their metrics JSON artifacts")

    background = frames["background_confound.csv"]
    _assert_unique(background, ["condition", "model", "seed"], "background_confound.csv")
    _assert_metric_range(background, ["accuracy", "macro_f1"], "background_confound.csv")
    expected_conditions = {"dhan_field", "dhan_white", "riceleafbd_field"}
    if set(background["condition"]) != expected_conditions:
        raise AssertionError("Background-confound conditions are incomplete")
    checks.append("All 3 background-confound conditions are present")

    gradcam = frames["gradcam_records.csv"]
    if int(gradcam["correct"].astype(bool).sum()) == 0:
        raise AssertionError("Grad-CAM records contain no correct predictions")
    if int((~gradcam["correct"].astype(bool)).sum()) == 0:
        raise AssertionError("Grad-CAM records contain no incorrect predictions")
    checks.append("Grad-CAM records contain both correct and incorrect samples")


def _validate_derived_tables(
    frames: dict[str, pd.DataFrame], checks: list[str]
) -> None:
    expected_pairwise = build_pairwise_augmentation(
        frames["crossdataset_matrix.csv"],
        frames["generalization_gap.csv"],
        frames["crossdataset_matrix_aug.csv"],
        frames["generalization_gap_aug.csv"],
    )
    expected_comparison = build_comparison_by_target(
        frames["crossdataset_matrix.csv"],
        frames["crossdataset_matrix_aug.csv"],
        frames["lodo_results.csv"],
    )
    assert_frame_equal(
        expected_pairwise,
        frames["mitigation_pairwise_aug.csv"],
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )
    assert_frame_equal(
        expected_comparison,
        frames["mitigation_comparison.csv"],
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )
    checks.append("Both Week 7 mitigation tables re-derive exactly")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def freeze_results(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[str] = []

    for name, source in SOURCES.items():
        if not source.exists():
            raise FileNotFoundError(f"Missing freeze input: {source}")
        shutil.copy2(source, output_dir / name)

    frames = {name: pd.read_csv(output_dir / name) for name in SOURCES}
    for name, expected in EXPECTED_ROWS.items():
        actual = len(frames[name])
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected} rows, found {actual}")
        checks.append(f"{name}: {actual} rows")

    _validate_manifest_archives(checks)
    _validate_frames(frames, checks)
    _validate_derived_tables(frames, checks)

    missing_week5_log = (
        ROOT
        / "week5_progress"
        / (
            "mobilenetv2_100__train-riceleafbd__run-to-brri_rice_disease_pest"
            "__classes-healthy+tungro__seed42.json"
        )
    )
    limitations = [
        (
            "One Week 5 training-history JSON is absent; its checkpoint, "
            "evaluation artifacts, and summary rows are present."
            if not missing_week5_log.exists()
            else "The previously missing Week 5 training-history JSON is present."
        ),
        "Week 7 checkpoints were not included in the downloaded result bundle.",
        (
            "The Week 6 Grad-CAM overlay PNG/checkpoint is unavailable locally; "
            "the 12 sample records are frozen."
        ),
        "All experiments use one frozen split and seed 42.",
    ]

    file_records = []
    for name in sorted(SOURCES):
        path = output_dir / name
        file_records.append(
            {
                "path": name,
                "source": str(SOURCES[name].relative_to(ROOT)).replace("\\", "/"),
                "rows": len(frames[name]),
                "sha256": sha256(path),
            }
        )

    manifest = {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "seed": 42,
        "status": "PASS",
        "files": file_records,
        "checks_passed": len(checks),
        "limitations": limitations,
    }
    manifest_path = output_dir / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# Week 8 result-freeze audit",
        "",
        "**Status: PASS**",
        "",
        f"- Git commit at freeze: `{manifest['git_commit']}`",
        f"- Canonical manifest SHA-256: `{sha256(output_dir / 'manifest.csv')}`",
        f"- Checks passed: {len(checks)}",
        "",
        "## Checks",
        "",
        *[f"- {check}" for check in checks],
        "",
        "## Known limitations",
        "",
        *[f"- {limitation}" for limitation in limitations],
        "",
        "Frozen CSVs are immutable publication inputs. Regenerate this freeze",
        "through `python -m freeze_results`; do not edit them manually.",
        "",
    ]
    (output_dir / "audit_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(f"PASS: {len(checks)} checks")
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {output_dir / 'audit_report.md'}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and freeze publication-facing results from Weeks 4-7."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Tracked output directory (default: frozen_results).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    freeze_results(args.output)
