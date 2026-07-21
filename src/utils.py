from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for training stochasticity.

    This controls initialization, shuffling, and augmentation sampling.
    It must not be used to rebuild frozen train/val/test splits — those are
    fixed by ``split_seed`` at manifest-build time.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn so each worker has a deterministic RNG stream."""
    del worker_id  # torch.initial_seed() already encodes worker id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    """Return a torch.Generator seeded for DataLoader shuffle reproducibility."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    """Load YAML config with env-var overrides for roots."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    data_root = os.getenv("DATA_ROOT")
    results_root = os.getenv("RESULTS_ROOT")

    if data_root:
        config["data_root"] = data_root
    if results_root:
        config["results_root"] = results_root

    return config


def get_device() -> torch.device:
    """Return CUDA device when available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_results_path(*parts: str) -> Path:
    """Create and return a path under configured results root."""
    config = load_config(str(DEFAULT_CONFIG_PATH))
    root = Path(config.get("results_root", "results"))
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
