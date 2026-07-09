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
    """Set all relevant RNG seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
