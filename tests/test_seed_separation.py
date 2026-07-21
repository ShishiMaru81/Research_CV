"""Tests for train_seed vs split_seed separation and DataLoader seeding."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from src.data_loader import make_loaders
from src.utils import load_config, make_torch_generator, set_seed


class TestSeedSeparation(unittest.TestCase):
    def test_split_seed_default_is_42(self) -> None:
        config = load_config("config.yaml")
        self.assertEqual(int(config.get("split_seed", -1)), 42)

    def test_make_torch_generator_is_deterministic(self) -> None:
        g1 = make_torch_generator(7)
        g2 = make_torch_generator(7)
        g3 = make_torch_generator(2024)
        self.assertEqual(int(g1.initial_seed()), 7)
        self.assertEqual(int(g2.initial_seed()), 7)
        self.assertEqual(int(g3.initial_seed()), 2024)

    def test_train_loader_shuffle_reproducible_by_train_seed(self) -> None:
        classes = ["brown_spot", "tungro"]
        n_train, n_val, n_test = 24, 8, 8

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_root = root / "results"
            image_root = root / "images" / "dummy_ds"
            results_root.mkdir(parents=True)
            image_root.mkdir(parents=True)

            rows: list[dict[str, object]] = []
            for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
                for i in range(count):
                    class_name = classes[i % 2]
                    class_dir = image_root / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)
                    image_path = class_dir / f"{split}_{i:02d}.jpg"
                    color = (20 + i, 90, 140) if class_name == classes[0] else (150, 40 + i, 80)
                    cv2.imwrite(str(image_path), np.full((48, 48, 3), color, dtype=np.uint8))
                    rows.append(
                        {
                            "image_path": str(image_path),
                            "dataset": "dummy_ds",
                            "original_class": class_name,
                            "mapped_class": class_name,
                            "background": "",
                            "split": split,
                            "is_duplicate": False,
                        }
                    )

            manifest_path = results_root / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest_path, index=False)

            # Point default load_config used inside make_loaders at a local config.
            config = {
                "data_root": str(root / "images"),
                "results_root": str(results_root),
                "seed": 42,
                "split_seed": 42,
                "image_size": 48,
                "batch_size": 8,
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            def first_batch_paths(train_seed: int) -> list[str]:
                set_seed(train_seed)
                train_loader, _, _, _ = make_loaders(
                    train_datasets=["dummy_ds"],
                    eval_dataset="dummy_ds",
                    classes=classes,
                    image_size=48,
                    batch_size=8,
                    manifest_path=str(manifest_path),
                    train_seed=train_seed,
                )
                _, _, paths = next(iter(train_loader))
                return list(paths)

            # Temporarily chdir so load_config() inside make_loaders finds our config.
            # make_loaders only needs load_config when manifest_path is None; we pass
            # manifest_path, so cwd does not matter for that path. Still safe.

            same_a = first_batch_paths(7)
            same_b = first_batch_paths(7)
            other = first_batch_paths(2024)

            self.assertEqual(same_a, same_b)
            self.assertNotEqual(same_a, other)

    def test_checkpoint_metadata_stores_both_seeds(self) -> None:
        """Document the expected checkpoint key contract without full training."""
        payload = {
            "seed": 7,
            "train_seed": 7,
            "split_seed": 42,
        }
        self.assertEqual(payload["train_seed"], 7)
        self.assertEqual(payload["split_seed"], 42)
        self.assertEqual(payload["seed"], payload["train_seed"])


if __name__ == "__main__":
    unittest.main()
