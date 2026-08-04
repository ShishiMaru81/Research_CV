"""Resumable experiment run registry for multi-seed / Kaggle campaigns.

Writes ``results/run_registry.csv``. Every training campaign should call
``is_complete`` before launching and ``mark_complete`` after eval finishes so
a killed Kaggle session can resume without re-running finished work.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.train import training_stem

REGISTRY_COLUMNS = [
    "run_id",
    "experiment_type",
    "model",
    "train_datasets",
    "eval_dataset",
    "classes",
    "split_seed",
    "train_seed",
    "augmentation",
    "status",
    "checkpoint_path",
    "predictions_path",
    "started_at",
    "completed_at",
]

STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"


def default_registry_path(results_root: str | Path) -> Path:
    return Path(results_root) / "run_registry.csv"


def make_run_id(
    model: str,
    train_datasets: list[str] | str,
    train_seed: int,
    eval_dataset: str,
    run_tag: str | None = None,
    augmentation: str = "default",
) -> str:
    """Stable id aligned with checkpoint naming, plus eval target.

    Format:
    ``{model}__train-{srcs}[__run-{tag}]__seed{N}__eval-{eval}__aug-{aug}``
    """
    datasets = (
        [train_datasets]
        if isinstance(train_datasets, str)
        else list(train_datasets)
    )
    stem = training_stem(model, datasets, int(train_seed), run_tag)
    aug_tag = augmentation if augmentation else "default"
    return f"{stem}__eval-{eval_dataset}__aug-{aug_tag}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in REGISTRY_COLUMNS})


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in REGISTRY_COLUMNS:
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].astype("object")
        out[column] = out[column].where(pd.notna(out[column]), "")
    return out[REGISTRY_COLUMNS]


class RunRegistry:
    """CSV-backed registry with upsert semantics keyed by ``run_id``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return _empty_frame()
        df = pd.read_csv(self.path)
        if df.empty:
            return _empty_frame()
        return _ensure_columns(df)

    def save(self, df: pd.DataFrame) -> None:
        _ensure_columns(df).to_csv(self.path, index=False)

    def _upsert(self, row: dict[str, Any]) -> None:
        df = self.load()
        run_id = str(row["run_id"])
        payload = {
            column: ("" if row.get(column) is None else row.get(column, ""))
            for column in REGISTRY_COLUMNS
        }
        if df.empty:
            df = pd.DataFrame([payload])
        else:
            mask = df["run_id"].astype(str) == run_id
            if mask.any():
                for column, value in payload.items():
                    if column == "started_at":
                        existing = df.loc[mask, "started_at"].iloc[0]
                        if existing not in ("", None) and pd.notna(existing):
                            # Keep original start time on re-register.
                            continue
                    df.loc[mask, column] = value
            else:
                df = pd.concat([df, pd.DataFrame([payload])], ignore_index=True)
        self.save(df)

    def register_run(
        self,
        run_id: str,
        experiment_type: str,
        model: str,
        train_datasets: str | list[str],
        eval_dataset: str,
        classes: str | list[str],
        split_seed: int,
        train_seed: int,
        augmentation: str = "default",
        checkpoint_path: str | Path | None = None,
        predictions_path: str | Path | None = None,
        status: str = STATUS_RUNNING,
    ) -> str:
        """Insert or update a run as running (or the given status)."""
        train_tag = (
            train_datasets
            if isinstance(train_datasets, str)
            else "+".join(train_datasets)
        )
        class_tag = classes if isinstance(classes, str) else "|".join(classes)
        self._upsert(
            {
                "run_id": run_id,
                "experiment_type": experiment_type,
                "model": model,
                "train_datasets": train_tag,
                "eval_dataset": eval_dataset,
                "classes": class_tag,
                "split_seed": int(split_seed),
                "train_seed": int(train_seed),
                "augmentation": augmentation,
                "status": status,
                "checkpoint_path": (
                    str(checkpoint_path) if checkpoint_path is not None else ""
                ),
                "predictions_path": (
                    str(predictions_path) if predictions_path is not None else ""
                ),
                "started_at": _utc_now(),
                "completed_at": "",
            }
        )
        return run_id

    def mark_complete(
        self,
        run_id: str,
        checkpoint_path: str | Path | None = None,
        predictions_path: str | Path | None = None,
    ) -> None:
        df = self.load()
        if df.empty or not (df["run_id"].astype(str) == run_id).any():
            raise KeyError(f"run_id not found in registry: {run_id}")
        mask = df["run_id"].astype(str) == run_id
        df.loc[mask, "status"] = STATUS_COMPLETE
        df.loc[mask, "completed_at"] = _utc_now()
        if checkpoint_path is not None:
            df.loc[mask, "checkpoint_path"] = str(checkpoint_path)
        if predictions_path is not None:
            df.loc[mask, "predictions_path"] = str(predictions_path)
        self.save(df)

    def mark_failed(self, run_id: str, note: str | None = None) -> None:
        del note  # reserved for a future notes column
        df = self.load()
        if df.empty or not (df["run_id"].astype(str) == run_id).any():
            raise KeyError(f"run_id not found in registry: {run_id}")
        mask = df["run_id"].astype(str) == run_id
        df.loc[mask, "status"] = STATUS_FAILED
        df.loc[mask, "completed_at"] = _utc_now()
        self.save(df)

    def is_complete(self, run_id: str) -> bool:
        df = self.load()
        if df.empty:
            return False
        hits = df.loc[df["run_id"].astype(str) == run_id]
        if hits.empty:
            return False
        return str(hits.iloc[-1]["status"]) == STATUS_COMPLETE

    def list_pending(
        self, experiment_type: str | None = None
    ) -> list[dict[str, Any]]:
        df = self.load()
        if df.empty:
            return []
        pending = df[df["status"].astype(str) != STATUS_COMPLETE].copy()
        if experiment_type is not None:
            pending = pending[
                pending["experiment_type"].astype(str) == experiment_type
            ]
        return pending.to_dict(orient="records")


# Module-level helpers matching the Phase 0 prompt API.


def register_run(
    registry_path: str | Path,
    run_id: str,
    **kwargs: Any,
) -> str:
    return RunRegistry(registry_path).register_run(run_id=run_id, **kwargs)


def mark_complete(
    registry_path: str | Path,
    run_id: str,
    checkpoint_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
) -> None:
    RunRegistry(registry_path).mark_complete(
        run_id, checkpoint_path=checkpoint_path, predictions_path=predictions_path
    )


def is_complete(registry_path: str | Path, run_id: str) -> bool:
    return RunRegistry(registry_path).is_complete(run_id)


def list_pending(
    registry_path: str | Path, experiment_type: str | None = None
) -> list[dict[str, Any]]:
    return RunRegistry(registry_path).list_pending(experiment_type=experiment_type)
