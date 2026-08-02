"""Unit tests for Phase-3 augmentation bucket profiles."""

from __future__ import annotations

import numpy as np

from src.data_loader import (
    AUGMENTATION_PROFILES,
    BUCKET_PROFILES,
    build_train_transform,
)


def test_all_profiles_build_and_run() -> None:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[40:80, 40:80] = 180
    for name in AUGMENTATION_PROFILES:
        transform = build_train_transform(224, name)
        out = transform(image=image)["image"]
        assert out.shape[0] == 3
        assert out.shape[1] == 224
        assert out.shape[2] == 224


def test_bucket_profiles_match_workflow_names() -> None:
    assert BUCKET_PROFILES == (
        "bucket-geo",
        "bucket-photo",
        "bucket-occlusion",
    )


def test_unknown_profile_raises() -> None:
    try:
        build_train_transform(224, "bucket-magic")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unknown augmentation" in str(exc)
