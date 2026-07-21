"""Unit tests for full per-sample prediction logging in src.eval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from src.eval import evaluate, save_predictions_csv
from src.train import build_model


REQUIRED_COLUMNS = [
    "image_path",
    "dataset",
    "true_index",
    "pred_index",
    "true_label",
    "pred_label",
    "correct",
]


class TestEvalPredictions(unittest.TestCase):
    def test_save_predictions_csv_schema(self) -> None:
        paths = [f"img_{i}.jpg" for i in range(20)]
        labels = [i % 2 for i in range(20)]
        preds = [0 if i < 10 else 1 for i in range(20)]
        probs = np.tile(np.array([[0.7, 0.3]], dtype=np.float64), (20, 1))
        index_to_class = {0: "brown_spot", 1: "tungro"}

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "predictions" / "dummy_run.csv"
            save_predictions_csv(
                out_path,
                paths,
                labels,
                preds,
                probs,
                index_to_class,
                dataset="riceleafbd",
            )
            df = pd.read_csv(out_path)

        self.assertEqual(len(df), 20)
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, df.columns)
        self.assertIn("prob_0", df.columns)
        self.assertIn("prob_1", df.columns)
        self.assertTrue((df["dataset"] == "riceleafbd").all())

    def test_evaluate_writes_predictions_csv(self) -> None:
        classes = ["brown_spot", "tungro"]
        class_to_index = {name: idx for idx, name in enumerate(classes)}
        n_test = 20
        n_train = 8
        n_val = 4

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
                    filename = f"{split}_{i:02d}.jpg"
                    image_path = class_dir / filename
                    # Distinct solid colors so OpenCV can read a valid image.
                    color = (30 + i, 80 + (i % 3) * 20, 120) if class_name == classes[0] else (
                        160,
                        40 + i,
                        90,
                    )
                    img = np.full((64, 64, 3), color, dtype=np.uint8)
                    cv2.imwrite(str(image_path), img)
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

            manifest = pd.DataFrame(rows)
            manifest_path = results_root / "manifest.csv"
            manifest.to_csv(manifest_path, index=False)

            config = {
                "data_root": str(root / "images"),
                "results_root": str(results_root),
                "seed": 42,
                "image_size": 64,
                "batch_size": 8,
                "model_name": "mobilenetv2_100",
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            model = build_model("mobilenetv2_100", num_classes=2, pretrained=False)
            ckpt_dir = results_root / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = ckpt_dir / "mobilenetv2_100__train-dummy_ds__seed42.pth"
            torch.save(
                {
                    "model_name": "mobilenetv2_100",
                    "train_datasets": ["dummy_ds"],
                    "eval_dataset": "dummy_ds",
                    "classes": classes,
                    "class_to_index": class_to_index,
                    "seed": 42,
                    "run_tag": None,
                    "augmentation": "default",
                    "best_val_macro_f1": 0.0,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            metrics = evaluate(
                checkpoint_path=checkpoint_path,
                eval_dataset="dummy_ds",
                classes=classes,
                split="test",
                seed=42,
                config_path=str(config_path),
                sample_n=0,
            )

            predictions_path = Path(metrics["predictions_path"])
            self.assertTrue(predictions_path.exists())
            self.assertEqual(predictions_path.parent.name, "predictions")
            self.assertEqual(
                predictions_path.name,
                "mobilenetv2_100__train-dummy_ds__eval-dummy_ds__seed42.csv",
            )

            df = pd.read_csv(predictions_path)
            self.assertEqual(len(df), n_test)
            for col in REQUIRED_COLUMNS:
                self.assertIn(col, df.columns)
            for class_idx in range(len(classes)):
                self.assertIn(f"prob_{class_idx}", df.columns)
            self.assertEqual(int(metrics["n_samples"]), n_test)
            # Softmax rows should sum to ~1.
            prob_sum = df[[f"prob_{i}" for i in range(len(classes))]].sum(axis=1)
            self.assertTrue(np.allclose(prob_sum.to_numpy(), 1.0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
