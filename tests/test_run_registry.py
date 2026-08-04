"""Unit tests for the resumable run registry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.run_registry import (
    REGISTRY_COLUMNS,
    RunRegistry,
    is_complete,
    list_pending,
    make_run_id,
    mark_complete,
    register_run,
)


class TestRunRegistry(unittest.TestCase):
    def test_make_run_id_is_stable_and_distinct(self) -> None:
        a = make_run_id(
            "resnet50",
            ["dhan_shomadhan"],
            train_seed=7,
            eval_dataset="riceleafbd",
            run_tag="to-riceleafbd__classes-brown_spot+tungro",
            augmentation="default",
        )
        b = make_run_id(
            "resnet50",
            ["dhan_shomadhan"],
            train_seed=7,
            eval_dataset="riceleafbd",
            run_tag="to-riceleafbd__classes-brown_spot+tungro",
            augmentation="default",
        )
        c = make_run_id(
            "resnet50",
            ["dhan_shomadhan"],
            train_seed=2024,
            eval_dataset="riceleafbd",
            run_tag="to-riceleafbd__classes-brown_spot+tungro",
            augmentation="default",
        )
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertIn("__seed7__", a)
        self.assertIn("__eval-riceleafbd__", a)

    def test_register_complete_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_registry.csv"
            run_id = make_run_id(
                "mobilenetv2_100",
                ["riceleafbd"],
                train_seed=42,
                eval_dataset="riceleafbd",
            )
            self.assertFalse(is_complete(path, run_id))

            register_run(
                path,
                run_id=run_id,
                experiment_type="indataset",
                model="mobilenetv2_100",
                train_datasets=["riceleafbd"],
                eval_dataset="riceleafbd",
                classes=["brown_spot", "tungro"],
                split_seed=42,
                train_seed=42,
                augmentation="default",
            )
            self.assertFalse(is_complete(path, run_id))
            pending = list_pending(path)
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["status"], "running")

            mark_complete(
                path,
                run_id,
                checkpoint_path="results/checkpoints/x.pth",
                predictions_path="results/predictions/x.csv",
            )
            self.assertTrue(is_complete(path, run_id))
            self.assertEqual(list_pending(path), [])

            registry = RunRegistry(path)
            df = registry.load()
            self.assertEqual(list(df.columns), REGISTRY_COLUMNS)
            self.assertEqual(df.iloc[0]["checkpoint_path"], "results/checkpoints/x.pth")
            self.assertEqual(
                df.iloc[0]["predictions_path"], "results/predictions/x.csv"
            )

            # Re-registering the same run_id upserts; still one row.
            register_run(
                path,
                run_id=run_id,
                experiment_type="indataset",
                model="mobilenetv2_100",
                train_datasets=["riceleafbd"],
                eval_dataset="riceleafbd",
                classes=["brown_spot", "tungro"],
                split_seed=42,
                train_seed=42,
            )
            self.assertEqual(len(registry.load()), 1)


if __name__ == "__main__":
    unittest.main()
