"""Assemble and audit ``frozen_results_v2/`` without mutating ``frozen_results/``.

Copies the immutable Week-8 core CSVs, then overlays revision-era aggregates
(multi-seed, AdaBN, ablation, stats) when present. Writes:

  frozen_results_v2/freeze_manifest_v2.json
  frozen_results_v2/AUDIT_REPORT_v2.md

Never edits ``frozen_results/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V1_DIR = ROOT / "frozen_results"
DEFAULT_OUT = ROOT / "frozen_results_v2"

# Core files that must match v1 seed-42 identity (bit-identical copy preferred).
V1_CORE = [
    "manifest.csv",
    "indataset_results.csv",
    "crossdataset_matrix.csv",
    "generalization_gap.csv",
    "background_confound.csv",
    "gradcam_records.csv",
    "crossdataset_matrix_aug.csv",
    "generalization_gap_aug.csv",
    "lodo_results.csv",
    "mitigation_pairwise_aug.csv",
    "mitigation_comparison.csv",
]

# Optional revision overlays (copied if present; absence is recorded, not fatal).
REVISION_SOURCES: dict[str, Path] = {
    "transfer_all_seeds.csv": ROOT / "week11_results" / "multiseed" / "transfer_all_seeds.csv",
    "transfer_cell_mean_std.csv": ROOT
    / "week11_results"
    / "multiseed"
    / "transfer_cell_mean_std.csv",
    "lodo_all_seeds.csv": ROOT / "week11_results" / "multiseed" / "lodo_all_seeds.csv",
    "adabn_results.csv": ROOT / "adabn_results.csv",
    "augmentation_ablation.csv": ROOT
    / "results"
    / "ablation"
    / "augmentation_ablation.csv",
    "stats_tests.csv": ROOT / "results" / "stats" / "stats_tests.csv",
    "seed_variance.csv": ROOT / "results" / "stats" / "seed_variance.csv",
    "bootstrap_ci.csv": ROOT / "results" / "stats" / "bootstrap_ci.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_commit_for_path(path: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "log", "-1", "--format=%H", "--", str(path)],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            or "untracked"
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _assert_close_frames(
    left: pd.DataFrame, right: pd.DataFrame, name: str, atol: float = 1e-9
) -> str:
    if list(left.columns) != list(right.columns):
        raise AssertionError(f"{name}: column mismatch v1 vs v2 copy")
    if len(left) != len(right):
        raise AssertionError(f"{name}: row count mismatch {len(left)} vs {len(right)}")
    for col in left.columns:
        if pd.api.types.is_numeric_dtype(left[col]):
            a = pd.to_numeric(left[col], errors="coerce")
            b = pd.to_numeric(right[col], errors="coerce")
            if not np.allclose(a.fillna(0), b.fillna(0), atol=atol, equal_nan=True):
                raise AssertionError(f"{name}.{col}: numeric mismatch vs v1")
    return f"{name}: v1 seed-42 values reproduced within tolerance"


def assemble_v2(output_dir: Path = DEFAULT_OUT) -> dict[str, Any]:
    if not V1_DIR.exists():
        raise FileNotFoundError(f"Missing {V1_DIR}; cannot build v2 without v1 freeze.")

    if output_dir.exists():
        # Replace contents but never touch frozen_results/.
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks: list[str] = []
    warnings: list[str] = []
    file_records: list[dict[str, Any]] = []

    for name in V1_CORE:
        src = V1_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"v1 missing required file: {src}")
        dst = output_dir / name
        shutil.copy2(src, dst)
        checks.append(_assert_close_frames(pd.read_csv(src), pd.read_csv(dst), name))
        file_records.append(
            {
                "path": name,
                "tier": "v1_core",
                "source": f"frozen_results/{name}",
                "rows": len(pd.read_csv(dst)),
                "sha256": sha256(dst),
                "matches_v1_sha256": sha256(src) == sha256(dst),
            }
        )

    for name, src in REVISION_SOURCES.items():
        dst = output_dir / name
        if not src.exists():
            warnings.append(f"optional missing: {src.as_posix()}")
            continue
        shutil.copy2(src, dst)
        rows = len(pd.read_csv(dst)) if dst.suffix == ".csv" else None
        file_records.append(
            {
                "path": name,
                "tier": "revision",
                "source": str(src.relative_to(ROOT)).replace("\\", "/"),
                "rows": rows,
                "sha256": sha256(dst),
                "matches_v1_sha256": None,
            }
        )
        checks.append(f"{name}: copied from revision overlay ({rows} rows)")

    # Soft seed-coverage check on transfer_all_seeds if present.
    transfer_path = output_dir / "transfer_all_seeds.csv"
    if transfer_path.exists():
        transfer = pd.read_csv(transfer_path)
        coverage = (
            transfer.groupby(["seed", "augmentation"]).size().to_dict()
        )
        expected = {
            (42, "default"): 18,
            (42, "strong"): 18,
            (7, "default"): 18,
            (7, "strong"): 18,
            (2024, "default"): 18,
            (2024, "strong"): 18,
        }
        for key, want in expected.items():
            got = int(coverage.get(key, 0))
            label = f"seed={key[0]} aug={key[1]}"
            if got == want:
                checks.append(f"seed coverage OK: {label} = {got}")
            else:
                warnings.append(
                    f"seed coverage incomplete: {label} has {got}, expected {want}"
                )

    ablation_path = output_dir / "augmentation_ablation.csv"
    if ablation_path.exists():
        abl = pd.read_csv(ablation_path)
        n = len(abl)
        if n >= 18:
            checks.append(f"ablation rows: {n} (>=18)")
        else:
            warnings.append(f"ablation incomplete: {n} rows (want >=18)")

    status = "PASS" if not warnings else "PASS_WITH_WARNINGS"
    manifest = {
        "schema_version": 2,
        "status": status,
        "git_commit": _git_commit(),
        "v1_dir": "frozen_results",
        "v1_commit_hint": _git_commit_for_path(V1_DIR / "freeze_manifest.json"),
        "split_seed": 42,
        "files": file_records,
        "checks_passed": checks,
        "warnings": warnings,
    }
    manifest_path = output_dir / "freeze_manifest_v2.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = [
        "# Freeze v2 audit",
        "",
        f"**Status: {status}**",
        "",
        f"- Git commit at freeze: `{manifest['git_commit']}`",
        f"- v1 freeze manifest commit hint: `{manifest['v1_commit_hint']}`",
        f"- Checks: {len(checks)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Checks",
        "",
        *[f"- {item}" for item in checks],
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        report.extend(f"- {item}" for item in warnings)
    else:
        report.append("- (none)")
    report.extend(
        [
            "",
            "## Policy",
            "",
            "- `frozen_results/` remains immutable.",
            "- This directory is the replacement freeze for revision claims.",
            "- See `notes/freeze_v2_changelog.md` for what changed and why.",
            "",
        ]
    )
    report_path = output_dir / "AUDIT_REPORT_v2.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Status: {status}")
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {report_path}")
    for warning in warnings:
        print(f"WARN: {warning}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assemble_v2(args.output)
