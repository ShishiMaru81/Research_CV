"""Adaptive Batch Normalization (AdaBN) for cross-dataset evaluation.

AdaBN recalibrates BatchNorm running mean/variance on unlabeled *target-domain
train* images, then evaluates on the target test split. No gradients, no
label use during adaptation (Li et al., 2016).
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from tqdm import tqdm


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def iter_batch_norm_modules(model: nn.Module) -> Iterable[nn.modules.batchnorm._BatchNorm]:
    for module in model.modules():
        if isinstance(module, BN_TYPES):
            yield module  # type: ignore[misc]


def count_batch_norm_layers(model: nn.Module) -> int:
    return sum(1 for _ in iter_batch_norm_modules(model))


def set_batch_norm_train_mode(model: nn.Module) -> int:
    """Put only BatchNorm modules in train mode; leave everything else in eval.

    Returns the number of BN modules switched. Dropout / stochastic depth stay
    off so adaptation only updates BN running statistics.
    """
    model.eval()
    n = 0
    for module in iter_batch_norm_modules(model):
        module.train()
        n += 1
    return n


@torch.no_grad()
def adapt_batch_norm(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    reset_running_stats: bool = True,
    max_batches: int | None = None,
) -> dict[str, int]:
    """Recalibrate BN running stats on ``loader`` (target train images).

    Parameters
    ----------
    reset_running_stats:
        If True, clear source-domain running mean/var before accumulating
        target statistics (standard AdaBN).
    max_batches:
        Optional cap for debugging / dry runs.
    """
    n_bn = set_batch_norm_train_mode(model)
    if n_bn == 0:
        raise ValueError("Model has no BatchNorm layers to adapt.")

    if reset_running_stats:
        for module in iter_batch_norm_modules(model):
            module.reset_running_stats()

    n_images = 0
    n_batches = 0
    for images, *_rest in tqdm(loader, leave=False, desc="AdaBN adapt"):
        images = images.to(device)
        model(images)
        n_images += int(images.shape[0])
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    model.eval()
    return {"n_bn_layers": n_bn, "n_adapt_images": n_images, "n_adapt_batches": n_batches}
