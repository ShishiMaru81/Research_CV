"""Unit tests for Adaptive BatchNorm helpers."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.adabn import (
    adapt_batch_norm,
    count_batch_norm_layers,
    set_batch_norm_train_mode,
)


class TinyBNNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        x = x.mean(dim=(2, 3))
        return self.fc(x)


class TestAdaBN(unittest.TestCase):
    def test_count_and_mode(self) -> None:
        model = TinyBNNet()
        self.assertEqual(count_batch_norm_layers(model), 1)
        n = set_batch_norm_train_mode(model)
        self.assertEqual(n, 1)
        self.assertTrue(model.bn.training)
        self.assertFalse(model.conv.training)
        self.assertFalse(model.fc.training)

    def test_adapt_updates_running_stats(self) -> None:
        torch.manual_seed(0)
        model = TinyBNNet()
        before_mean = model.bn.running_mean.clone()
        before_var = model.bn.running_var.clone()

        # Distinct-from-init target batch so stats must move.
        images = torch.randn(16, 3, 8, 8) * 2.0 + 1.5
        labels = torch.zeros(16, dtype=torch.long)
        loader = DataLoader(TensorDataset(images, labels), batch_size=8)

        info = adapt_batch_norm(model, loader, torch.device("cpu"))
        self.assertEqual(info["n_bn_layers"], 1)
        self.assertEqual(info["n_adapt_images"], 16)
        self.assertFalse(torch.allclose(model.bn.running_mean, before_mean))
        self.assertFalse(torch.allclose(model.bn.running_var, before_var))
        self.assertFalse(model.training)
        self.assertFalse(model.bn.training)


if __name__ == "__main__":
    unittest.main()
