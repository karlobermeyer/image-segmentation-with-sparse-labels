import itertools
import pytest
from typing import List, Dict

import numpy as np
import torch

from core.semantic_class_mapping import SemanticClassMapping


np.random.seed(13)
torch.manual_seed(13)


# ------------------ Fixtures ------------------


@pytest.fixture(scope="module")
def class_mapping_all_included_unlabeled0() -> SemanticClassMapping:
    class_ids_raw: np.ndarray = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    class_ids_raw_included: np.ndarray = class_ids_raw.copy()
    class_names_included: List[str] = ["unlabeled", "a", "b", "c", "d"]
    return SemanticClassMapping(
        class_ids_raw=class_ids_raw,
        class_ids_raw_included=class_ids_raw_included,
        class_names_included=class_names_included,
    )


@pytest.fixture(scope="module")
def class_mapping_all_included_unlabeled255() -> SemanticClassMapping:
    class_ids_raw: np.ndarray = np.array([255, 0, 1, 2, 3], dtype=np.uint8)
    class_ids_raw_included: np.ndarray = class_ids_raw.copy()
    class_names_included: List[str] = ["unlabeled", "a", "b", "c", "d"]
    return SemanticClassMapping(
        class_ids_raw=class_ids_raw,
        class_ids_raw_included=class_ids_raw_included,
        class_names_included=class_names_included,
    )


@pytest.fixture(scope="module")
def class_mapping_strict_subset_included_unlabeled0() -> SemanticClassMapping:
    class_ids_raw: np.ndarray = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    class_ids_raw_included: np.ndarray = np.array([0, 2, 3], dtype=np.uint8)
    class_names_included: List[str] = ["unlabeled", "b", "c"]
    return SemanticClassMapping(
        class_ids_raw=class_ids_raw,
        class_ids_raw_included=class_ids_raw_included,
        class_names_included=class_names_included,
    )


@pytest.fixture(scope="module")
def class_mapping_strict_subset_included_unlabeled255() -> SemanticClassMapping:
    class_ids_raw: np.ndarray = np.array([255, 0, 1, 2, 3], dtype=np.uint8)
    class_ids_raw_included: np.ndarray = np.array([255, 1, 2], dtype=np.uint8)
    class_names_included: List[str] = ["unlabeled", "b", "c"]
    return SemanticClassMapping(
        class_ids_raw=class_ids_raw,
        class_ids_raw_included=class_ids_raw_included,
        class_names_included=class_names_included,
    )


# ------------------ Mock Sampling Functions ------------------


def sample_mock_segmentation_map_int(
    class_ids: np.ndarray,
    height: int = 50,
    width: int = 100,
) -> np.ndarray:
    segmentation_map_int: np.ndarray = np.zeros((height, width), dtype=np.uint8)
    for i, j in itertools.product(range(height), range(width)):
        segmentation_map_int[i, j] = np.random.choice(class_ids)
    return segmentation_map_int


def sample_mock_segmentation_map_logits(
    num_classes: int,
    height: int = 50,
    width: int = 100,
) -> torch.Tensor:
    return torch.rand(num_classes, height, width, dtype=torch.float32)


def sample_mock_segmentation_map_categorical(
    num_classes: int,
    height: int = 50,
    width: int = 100,
) -> torch.Tensor:
    segmentation_map_logits: torch.Tensor = \
        sample_mock_segmentation_map_logits(num_classes, height, width)
    return torch.softmax(segmentation_map_logits, dim=0)


# ------------------ Test Helpers ------------------


def segmentation_map_int_is_feasible(
    segmentation_map_int: np.ndarray,
    class_ids: np.ndarray,
) -> bool:
    """
    Check whether every element of a segmentation map is a feasible semantic
    class ID.
    """
    height, width = segmentation_map_int.shape
    for i, j in itertools.product(range(height), range(width)):
        if segmentation_map_int[i, j] not in class_ids:
            return False
    return True


# ------------------ Tests ------------------


def test_class_ids_raw(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    class_ids_raw: np.ndarray = class_mapping.class_ids_raw
    assert isinstance(class_ids_raw, np.ndarray)
    assert len(class_ids_raw) == 5

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    assert isinstance(class_ids_raw, np.ndarray)
    assert len(class_ids_raw) == 5

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    assert isinstance(class_ids_raw, np.ndarray)
    assert len(class_ids_raw) == 5

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    assert isinstance(class_ids_raw, np.ndarray)
    assert len(class_ids_raw) == 5


def test_num_classes_raw(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    assert class_mapping.num_classes_raw == 5

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    assert class_mapping.num_classes_raw == 5

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    assert class_mapping.num_classes_raw == 5

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    assert class_mapping.num_classes_raw == 5


def test_class_ids(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    class_ids: np.ndarray = class_mapping.class_ids
    assert isinstance(class_ids, np.ndarray)
    assert len(class_ids) == 5

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    class_ids: np.ndarray = class_mapping.class_ids
    assert isinstance(class_ids, np.ndarray)
    assert len(class_ids) == 5

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    class_ids: np.ndarray = class_mapping.class_ids
    assert isinstance(class_ids, np.ndarray)
    assert len(class_ids) == 3

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    class_ids: np.ndarray = class_mapping.class_ids
    assert isinstance(class_ids, np.ndarray)
    assert len(class_ids) == 3


def test_num_classes(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    assert class_mapping.num_classes == 5

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    assert class_mapping.num_classes == 5

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    assert class_mapping.num_classes == 3

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    assert class_mapping.num_classes == 3


def test_class_names(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    class_names: List[str] = class_mapping.class_names
    assert isinstance(class_names, list)
    for item in class_names:
        assert isinstance(item, str)
    assert len(class_names) == 5

    class_mapping: SemanticClassMapping = \
    class_mapping_all_included_unlabeled255
    class_names: List[str] = class_mapping.class_names
    assert isinstance(class_names, list)
    for item in class_names:
        assert isinstance(item, str)
    assert len(class_names) == 5

    class_mapping: SemanticClassMapping = \
    class_mapping_strict_subset_included_unlabeled0
    class_names: List[str] = class_mapping.class_names
    assert isinstance(class_names, list)
    for item in class_names:
        assert isinstance(item, str)
    assert len(class_names) == 3

    class_mapping: SemanticClassMapping = \
    class_mapping_strict_subset_included_unlabeled255
    class_names: List[str] = class_mapping.class_names
    assert isinstance(class_names, list)
    for item in class_names:
        assert isinstance(item, str)
    assert len(class_names) == 3


def test_class_names_dict(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    class_names_dict: Dict[int, str] = class_mapping.class_names_dict
    assert isinstance(class_names_dict, dict)
    for ix, name in class_names_dict.items():
        assert isinstance(ix, int)
        assert isinstance(name, str)
    assert len(class_names_dict) == 5

    class_mapping: SemanticClassMapping = \
    class_mapping_all_included_unlabeled255
    class_names_dict: Dict[int, str] = class_mapping.class_names_dict
    assert isinstance(class_names_dict, dict)
    for ix, name in class_names_dict.items():
        assert isinstance(ix, int)
        assert isinstance(name, str)
    assert len(class_names_dict) == 5

    class_mapping: SemanticClassMapping = \
    class_mapping_strict_subset_included_unlabeled0
    class_names_dict: Dict[int, str] = class_mapping.class_names_dict
    assert isinstance(class_names_dict, dict)
    for ix, name in class_names_dict.items():
        assert isinstance(ix, int)
        assert isinstance(name, str)
    assert len(class_names_dict) == 3

    class_mapping: SemanticClassMapping = \
    class_mapping_strict_subset_included_unlabeled255
    class_names_dict: Dict[int, str] = class_mapping.class_names_dict
    assert isinstance(class_names_dict, dict)
    for ix, name in class_names_dict.items():
        assert isinstance(ix, int)
        assert isinstance(name, str)
    assert len(class_names_dict) == 3


def test_class_colors_rgb(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    class_colors_rgb: np.ndarray = class_mapping.class_colors_rgb
    assert isinstance(class_colors_rgb, np.ndarray)
    assert class_colors_rgb.shape == (class_mapping.num_classes, 3)

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    class_colors_rgb: np.ndarray = class_mapping.class_colors_rgb
    assert isinstance(class_colors_rgb, np.ndarray)
    assert class_colors_rgb.shape == (class_mapping.num_classes, 3)

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    class_colors_rgb: np.ndarray = class_mapping.class_colors_rgb
    assert isinstance(class_colors_rgb, np.ndarray)
    assert class_colors_rgb.shape == (class_mapping.num_classes, 3)

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    class_colors_rgb: np.ndarray = class_mapping.class_colors_rgb
    assert isinstance(class_colors_rgb, np.ndarray)
    assert class_colors_rgb.shape == (class_mapping.num_classes, 3)


def test_class_colors_bgr(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    class_colors_bgr: np.ndarray = class_mapping.class_colors_bgr
    assert isinstance(class_colors_bgr, np.ndarray)
    assert class_colors_bgr.shape == (class_mapping.num_classes, 3)

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    class_colors_bgr: np.ndarray = class_mapping.class_colors_bgr
    assert isinstance(class_colors_bgr, np.ndarray)
    assert class_colors_bgr.shape == (class_mapping.num_classes, 3)

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    class_colors_bgr: np.ndarray = class_mapping.class_colors_bgr
    assert isinstance(class_colors_bgr, np.ndarray)
    assert class_colors_bgr.shape == (class_mapping.num_classes, 3)

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    class_colors_bgr: np.ndarray = class_mapping.class_colors_bgr
    assert isinstance(class_colors_bgr, np.ndarray)
    assert class_colors_bgr.shape == (class_mapping.num_classes, 3)


def test_segmentation_map_int_from_raw(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    segmentation_map_int_raw: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids_raw)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_raw(segmentation_map_int_raw)
    assert segmentation_map_int.shape == segmentation_map_int_raw.shape
    assert segmentation_map_int_is_feasible(
        segmentation_map_int,
        class_mapping.class_ids,
    )

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    segmentation_map_int_raw: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids_raw)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_raw(segmentation_map_int_raw)
    assert segmentation_map_int.shape == segmentation_map_int_raw.shape
    assert segmentation_map_int_is_feasible(
        segmentation_map_int,
        class_mapping.class_ids,
    )

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    segmentation_map_int_raw: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids_raw)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_raw(segmentation_map_int_raw)
    assert segmentation_map_int.shape == segmentation_map_int_raw.shape
    assert segmentation_map_int_is_feasible(
        segmentation_map_int,
        class_mapping.class_ids,
    )

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    segmentation_map_int_raw: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids_raw)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_raw(segmentation_map_int_raw)
    assert segmentation_map_int.shape == segmentation_map_int_raw.shape
    assert segmentation_map_int_is_feasible(
        segmentation_map_int,
        class_mapping.class_ids,
    )


def test_segmentation_map_int_to_raw(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    segmentation_map_int: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids)
    segmentation_map_int_raw: np.ndarray = \
        class_mapping.segmentation_map_int_to_raw(segmentation_map_int)
    assert segmentation_map_int.shape == segmentation_map_int_raw.shape
    assert segmentation_map_int_is_feasible(
        segmentation_map_int_raw,
        class_mapping.class_ids_raw,
    )

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    segmentation_map_int: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids)
    segmentation_map_int_raw: np.ndarray = \
        class_mapping.segmentation_map_int_to_raw(segmentation_map_int)
    assert segmentation_map_int.shape == segmentation_map_int_raw.shape
    assert segmentation_map_int_is_feasible(
        segmentation_map_int_raw,
        class_mapping.class_ids_raw,
    )

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    segmentation_map_int: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids)
    segmentation_map_int_raw: np.ndarray = \
        class_mapping.segmentation_map_int_to_raw(segmentation_map_int)
    assert segmentation_map_int.shape == segmentation_map_int_raw.shape
    assert segmentation_map_int_is_feasible(
        segmentation_map_int_raw,
        class_mapping.class_ids_raw,
    )

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    segmentation_map_int: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids)
    segmentation_map_int_raw: np.ndarray = \
        class_mapping.segmentation_map_int_to_raw(segmentation_map_int)
    assert segmentation_map_int.shape == segmentation_map_int_raw.shape
    assert segmentation_map_int_is_feasible(
        segmentation_map_int_raw,
        class_mapping.class_ids_raw,
    )


def test_segmentation_map_int_from_logits(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    segmentation_map_logits: torch.Tensor = \
        sample_mock_segmentation_map_logits(class_mapping.num_classes)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_logits(segmentation_map_logits)
    assert segmentation_map_int.shape[0] == segmentation_map_logits.shape[1]
    assert segmentation_map_int.shape[1] == segmentation_map_logits.shape[2]
    assert segmentation_map_int_is_feasible(
        segmentation_map_int,
        class_mapping.class_ids,
    )

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    segmentation_map_logits: torch.Tensor = \
        sample_mock_segmentation_map_logits(class_mapping.num_classes)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_logits(segmentation_map_logits)
    assert segmentation_map_int.shape[0] == segmentation_map_logits.shape[1]
    assert segmentation_map_int.shape[1] == segmentation_map_logits.shape[2]
    assert segmentation_map_int_is_feasible(
        segmentation_map_int,
        class_mapping.class_ids,
    )

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    segmentation_map_logits: torch.Tensor = \
        sample_mock_segmentation_map_logits(class_mapping.num_classes)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_logits(segmentation_map_logits)
    assert segmentation_map_int.shape[0] == segmentation_map_logits.shape[1]
    assert segmentation_map_int.shape[1] == segmentation_map_logits.shape[2]
    assert segmentation_map_int_is_feasible(
        segmentation_map_int,
        class_mapping.class_ids,
    )

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    segmentation_map_logits: torch.Tensor = \
        sample_mock_segmentation_map_logits(class_mapping.num_classes)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_logits(segmentation_map_logits)
    assert segmentation_map_int.shape[0] == segmentation_map_logits.shape[1]
    assert segmentation_map_int.shape[1] == segmentation_map_logits.shape[2]
    assert segmentation_map_int_is_feasible(
        segmentation_map_int,
        class_mapping.class_ids,
    )


def test_segmentation_map_int_from_categorical(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    segmentation_map_categorical: torch.Tensor = \
        sample_mock_segmentation_map_categorical(class_mapping.num_classes)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_categorical(segmentation_map_categorical)
    assert segmentation_map_int.shape[0] == segmentation_map_categorical.shape[1]
    assert segmentation_map_int.shape[1] == segmentation_map_categorical.shape[2]
    assert segmentation_map_int_is_feasible(
        segmentation_map_int,
        class_mapping.class_ids,
    )

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    segmentation_map_categorical: torch.Tensor = \
        sample_mock_segmentation_map_categorical(class_mapping.num_classes)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_categorical(segmentation_map_categorical)
    assert segmentation_map_int.shape[0] == segmentation_map_categorical.shape[1]
    assert segmentation_map_int.shape[1] == segmentation_map_categorical.shape[2]
    assert segmentation_map_int_is_feasible(
        segmentation_map_int,
        class_mapping.class_ids,
    )

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    segmentation_map_categorical: torch.Tensor = \
        sample_mock_segmentation_map_categorical(class_mapping.num_classes)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_categorical(segmentation_map_categorical)
    assert segmentation_map_int.shape[0] == segmentation_map_categorical.shape[1]
    assert segmentation_map_int.shape[1] == segmentation_map_categorical.shape[2]
    assert segmentation_map_int_is_feasible(
        segmentation_map_int,
        class_mapping.class_ids,
    )

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    segmentation_map_categorical: torch.Tensor = \
        sample_mock_segmentation_map_categorical(class_mapping.num_classes)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_categorical(segmentation_map_categorical)
    assert segmentation_map_int.shape[0] == segmentation_map_categorical.shape[1]
    assert segmentation_map_int.shape[1] == segmentation_map_categorical.shape[2]
    assert segmentation_map_int_is_feasible(
        segmentation_map_int,
        class_mapping.class_ids,
    )


def test_segmentation_map_int_from_rgb(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    # TODO: Finish this if/when
    # `SemanticClassMapping.segmentation_map_int_from_rgb` is implemented.
    assert True


def test_segmentation_map_categorical_from_logits(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    segmentation_map_logits: torch.Tensor = \
        sample_mock_segmentation_map_logits(class_mapping.num_classes)
    segmentation_map_categorical: torch.Tensor = \
        class_mapping.segmentation_map_categorical_from_logits(
            segmentation_map_logits,
        )
    assert segmentation_map_logits.shape == segmentation_map_categorical.shape

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    segmentation_map_logits: torch.Tensor = \
        sample_mock_segmentation_map_logits(class_mapping.num_classes)
    segmentation_map_categorical: torch.Tensor = \
        class_mapping.segmentation_map_categorical_from_logits(
            segmentation_map_logits,
        )
    assert segmentation_map_logits.shape == segmentation_map_categorical.shape

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    segmentation_map_logits: torch.Tensor = \
        sample_mock_segmentation_map_logits(class_mapping.num_classes)
    segmentation_map_categorical: torch.Tensor = \
        class_mapping.segmentation_map_categorical_from_logits(
            segmentation_map_logits,
        )
    assert segmentation_map_logits.shape == segmentation_map_categorical.shape

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    segmentation_map_logits: torch.Tensor = \
        sample_mock_segmentation_map_logits(class_mapping.num_classes)
    segmentation_map_categorical: torch.Tensor = \
        class_mapping.segmentation_map_categorical_from_logits(
            segmentation_map_logits,
        )
    assert segmentation_map_logits.shape == segmentation_map_categorical.shape


def test_segmentation_map_categorical_from_int(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    segmentation_map_int: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids)
    segmentation_map_categorical: torch.Tensor = \
        class_mapping.segmentation_map_categorical_from_int(
            segmentation_map_int,
        )
    assert segmentation_map_categorical.shape[0] == class_mapping.num_classes
    assert segmentation_map_categorical.shape[1] == \
        segmentation_map_int.shape[0]
    assert segmentation_map_categorical.shape[2] == \
        segmentation_map_int.shape[1]

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    segmentation_map_int: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids)
    segmentation_map_categorical: torch.Tensor = \
        class_mapping.segmentation_map_categorical_from_int(
            segmentation_map_int,
        )
    assert segmentation_map_categorical.shape[0] == class_mapping.num_classes
    assert segmentation_map_categorical.shape[1] == \
        segmentation_map_int.shape[0]
    assert segmentation_map_categorical.shape[2] == \
        segmentation_map_int.shape[1]

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    segmentation_map_int: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids)
    segmentation_map_categorical: torch.Tensor = \
        class_mapping.segmentation_map_categorical_from_int(
            segmentation_map_int,
        )
    assert segmentation_map_categorical.shape[0] == class_mapping.num_classes
    assert segmentation_map_categorical.shape[1] == \
        segmentation_map_int.shape[0]
    assert segmentation_map_categorical.shape[2] == \
        segmentation_map_int.shape[1]

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    segmentation_map_int: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids)
    segmentation_map_categorical: torch.Tensor = \
        class_mapping.segmentation_map_categorical_from_int(
            segmentation_map_int,
        )
    assert segmentation_map_categorical.shape[0] == class_mapping.num_classes
    assert segmentation_map_categorical.shape[1] == \
        segmentation_map_int.shape[0]
    assert segmentation_map_categorical.shape[2] == \
        segmentation_map_int.shape[1]


def test_segmentation_map_categorical_from_raw(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    segmentation_map_int_raw: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids_raw)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_raw(segmentation_map_int_raw)
    segmentation_map_categorical: torch.Tensor = \
        class_mapping.segmentation_map_categorical_from_raw(
            segmentation_map_int_raw,
        )
    assert segmentation_map_categorical.shape[0] == class_mapping.num_classes
    assert segmentation_map_categorical.shape[1] == \
        segmentation_map_int.shape[0]
    assert segmentation_map_categorical.shape[2] == \
        segmentation_map_int.shape[1]

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    segmentation_map_int_raw: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids_raw)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_raw(segmentation_map_int_raw)
    segmentation_map_categorical: torch.Tensor = \
        class_mapping.segmentation_map_categorical_from_raw(
            segmentation_map_int_raw,
        )
    assert segmentation_map_categorical.shape[0] == class_mapping.num_classes
    assert segmentation_map_categorical.shape[1] == \
        segmentation_map_int.shape[0]
    assert segmentation_map_categorical.shape[2] == \
        segmentation_map_int.shape[1]

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    segmentation_map_int_raw: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids_raw)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_raw(segmentation_map_int_raw)
    segmentation_map_categorical: torch.Tensor = \
        class_mapping.segmentation_map_categorical_from_raw(
            segmentation_map_int_raw,
        )
    assert segmentation_map_categorical.shape[0] == class_mapping.num_classes
    assert segmentation_map_categorical.shape[1] == \
        segmentation_map_int.shape[0]
    assert segmentation_map_categorical.shape[2] == \
        segmentation_map_int.shape[1]

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    segmentation_map_int_raw: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids_raw)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_raw(segmentation_map_int_raw)
    segmentation_map_categorical: torch.Tensor = \
        class_mapping.segmentation_map_categorical_from_raw(
            segmentation_map_int_raw,
        )
    assert segmentation_map_categorical.shape[0] == class_mapping.num_classes
    assert segmentation_map_categorical.shape[1] == \
        segmentation_map_int.shape[0]
    assert segmentation_map_categorical.shape[2] == \
        segmentation_map_int.shape[1]


def test_segmentation_map_rgb_from_int(
    class_mapping_all_included_unlabeled0: SemanticClassMapping,
    class_mapping_all_included_unlabeled255: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled0: SemanticClassMapping,
    class_mapping_strict_subset_included_unlabeled255: SemanticClassMapping,
) -> None:
    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled0
    segmentation_map_int: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids)
    segmentation_map_rgb: np.ndarray = \
        class_mapping.segmentation_map_rgb_from_int(segmentation_map_int)
    assert segmentation_map_int.shape == segmentation_map_rgb.shape[:2]
    assert segmentation_map_rgb.shape[2] == 3

    class_mapping: SemanticClassMapping = \
        class_mapping_all_included_unlabeled255
    segmentation_map_int: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids)
    segmentation_map_rgb: np.ndarray = \
        class_mapping.segmentation_map_rgb_from_int(segmentation_map_int)
    assert segmentation_map_int.shape == segmentation_map_rgb.shape[:2]
    assert segmentation_map_rgb.shape[2] == 3

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled0
    segmentation_map_int: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids)
    segmentation_map_rgb: np.ndarray = \
        class_mapping.segmentation_map_rgb_from_int(segmentation_map_int)
    assert segmentation_map_int.shape == segmentation_map_rgb.shape[:2]
    assert segmentation_map_rgb.shape[2] == 3

    class_mapping: SemanticClassMapping = \
        class_mapping_strict_subset_included_unlabeled255
    segmentation_map_int: np.ndarray = \
        sample_mock_segmentation_map_int(class_mapping.class_ids)
    segmentation_map_rgb: np.ndarray = \
        class_mapping.segmentation_map_rgb_from_int(segmentation_map_int)
    assert segmentation_map_int.shape == segmentation_map_rgb.shape[:2]
    assert segmentation_map_rgb.shape[2] == 3
