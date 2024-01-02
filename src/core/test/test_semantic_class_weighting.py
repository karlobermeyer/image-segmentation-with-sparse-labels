from collections import Counter
import pytest
from typing import List

import numpy as np

from core.semantic_class_weighting import (
    compute_semantic_class_num_pixels,
    compute_semantic_class_frequencies,
    smooth_semantic_class_frequencies,
    compute_semantic_class_weights,
)
from core.identifiers import SemanticClassId
from core.semantic_class_mapping import SemanticClassMapping


np.random.seed(13)


# ------------------ Fixtures ------------------


@pytest.fixture(scope="module")
def class_mapping() -> SemanticClassMapping:
    class_ids_raw: np.ndarray = np.array([0, 1, 2, 3], dtype=np.uint8)
    class_ids_raw_included: np.ndarray = class_ids_raw.copy()
    class_names_included: List[str] = ["unlabeled", "a", "b", "c"]
    return SemanticClassMapping(
        class_ids_raw=class_ids_raw,
        class_ids_raw_included=class_ids_raw_included,
        class_names_included=class_names_included,
    )


# ------------------ Tests ------------------


def test_compute_semantic_class_num_pixels(
    class_mapping: SemanticClassMapping,
) -> None:
    target_class_num_pixels: List[Counter[SemanticClassId]] = []
    target_class_num_pixels.append(Counter([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]))
    target_class_num_pixels.append(Counter([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]))
    target_class_num_pixels.append(Counter([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]))
    class_num_pixels: Counter[SemanticClassId] = \
        compute_semantic_class_num_pixels(
            class_mapping=class_mapping,
            target_class_num_pixels=target_class_num_pixels,
        )
    assert class_num_pixels[0] == 3
    assert class_num_pixels[1] == 6
    assert class_num_pixels[2] == 9
    assert class_num_pixels[3] == 12


def test_compute_semantic_class_frequencies() -> None:
    class_num_pixels: Counter[SemanticClassId] = Counter({
        0: 3, 1: 6, 2: 9, 3: 12,
    })
    class_frequencies: np.ndarray = \
        compute_semantic_class_frequencies(class_num_pixels)
    total: int = sum(class_num_pixels.values())
    assert class_frequencies[0] == class_num_pixels[0]/total
    assert class_frequencies[1] == class_num_pixels[1]/total
    assert class_frequencies[2] == class_num_pixels[2]/total
    assert class_frequencies[3] == class_num_pixels[3]/total


def test_smooth_semantic_class_frequencies() -> None:
    class_num_pixels: Counter[SemanticClassId] = Counter({
        0: 3, 1: 6, 2: 9, 3: 12,
    })
    frequencies: np.ndarray = \
        compute_semantic_class_frequencies(class_num_pixels)
    smooth_frequencies: np.ndarray = smooth_semantic_class_frequencies(
        class_frequencies=frequencies,
        frequency_max_to_min_ratio_ubnd=10.0,
    )
    assert np.all(smooth_frequencies <= frequencies)
    assert abs(1.0 - sum(smooth_frequencies)) < 0.0001


def test_compute_semantic_class_weights() -> None:
    class_frequencies: np.ndarray = np.random.random(4)
    class_frequencies /= sum(class_frequencies)
    weights: np.ndarray = compute_semantic_class_weights(class_frequencies)
    assert abs(1.0 - sum(weights)) < 0.0001
