# Standard
import os
from collections import Counter
from typing import Any, Dict, List, Sequence

# Machine Learning
import albumentations as albm

# Project-Specific
from core import (
    SemanticClassId,
    SparseLabelSimulatingDataset,
)
from data.cityscapes.preprocesses import preprocesses_from
from data.cityscapes.dataset_factories import (
    subcityscapes_dataset,
    cityscapes_dataset,
)


REPO_ROOT: str = os.environ.get("REPO_ROOT")
assert REPO_ROOT is not None, \
    "REPO_ROOT not found! Did you run `setenv.sh` before serving the notebook?"

preprocesses: Dict[str, albm.Compose] = preprocesses_from(
    input_height=320,
    input_width=640,
    mean_for_input_normalization=(0.485, 0.456, 0.406),
    std_for_input_normalization=(0.229, 0.224, 0.225),
    #do_shift_scale_rotate=...,  # Not relevant here since we're only using
    #ignore_index=...,           # the "infer" preprocessing.
)
PREPROCESS: albm.Compose = preprocesses["infer"]


class TargetClassNumPixelsCache:
    """
    Container for precomputing and caching semantic class occurrences.

    See instructions in header of `precompute_target_class_num_pixels.py`.
    """
    def __init__(
        self,
        dataset_names: Sequence[str],
        len_scale_factors: Sequence[float],
        label_densities:  Sequence[float],
    ) -> None:
        self._data_scale_density: \
            Dict[str, Dict[float, Dict[float,
                List[Counter[SemanticClassId]],
            ]]] = {}

        for dataset_name in dataset_names:
            assert dataset_name in (
                "subcityscapes_train",
                "subcityscapes_val",
                "subcityscapes_trainval",  # == cityscapes_train
                "subcityscapes_test",  # == cityscapes_val
                "cityscapes_train",
                "cityscapes_val",
                "cityscapes_trainval",
            )
        for factor in len_scale_factors:
            assert 0.0 < factor <= 1.0
        for density in label_densities:
            assert 0.0 < density <= 1.0

        self._dataset_names: List[str] = list(dataset_names)
        self._len_scale_factors: List[float] = list(len_scale_factors)
        self._label_densities: List[float] = list(label_densities)

        for dataset_name in self._dataset_names:

            if dataset_name == "subcityscapes_trainval" \
                    and "cityscapes_train" in self:
                self["subcityscapes_trainval"] = self["cityscapes_train"]
                continue
            if dataset_name == "cityscapes_train" \
                    and "subcityscapes_trainval" in self:
                self["cityscapes_train"] = self["subcityscapes_trainval"]
                continue
            if dataset_name == "subcityscapes_test" \
                    and "cityscapes_val" in self:
                self["subcityscapes_test"] = self["cityscapes_val"]
                continue
            if dataset_name == "cityscapes_val" \
                    and "subcityscapes_test" in self:
                self["cityscapes_val"] = self["subcityscapes_test"]
                continue

            self[dataset_name] = {}

            dataset_name_parts: List[str] = dataset_name.split("_")
            base_dataset_name: str = dataset_name_parts[0]
            split: str = dataset_name_parts[1]
            for scale in self._len_scale_factors:
                self[dataset_name][scale] = {}
                for density in self._label_densities:
                    if base_dataset_name == "subcityscapes":
                        dataset: SparseLabelSimulatingDataset = \
                            subcityscapes_dataset(
                                split=split,
                                preprocess=PREPROCESS,
                                len_scale_factor=scale,
                                label_density=density,
                                ignore_index=255,
                                shuffle=False,
                            )
                    else:
                        assert base_dataset_name == "cityscapes"
                        dataset: SparseLabelSimulatingDataset = \
                            cityscapes_dataset(
                                split=split,
                                preprocess=PREPROCESS,
                                len_scale_factor=scale,
                                label_density=density,
                                ignore_index=255,
                                shuffle=False,
                            )
                    self[dataset_name][scale][density] = \
                        dataset.target_class_num_pixels

    def __getitem__(
        self,
        key: str,
    ) -> Dict[float, Dict[float, List[Counter[SemanticClassId]]]]:
        return self._data_scale_density[key]

    def __setitem__(
        self,
        key: str,
        value: Dict[Any, Any],
    ) -> None:
        self._data_scale_density[key] = value

    def __contains__(self, dataset_name: str) -> bool:
        if dataset_name in self._data_scale_density:
            return True
        return False

    def has(
        self,
        dataset_name: str,
        len_scale_factor: float,
        label_density: float,
    ) -> bool:
        if dataset_name in self._dataset_names \
                and len_scale_factor in self._len_scale_factors \
                and label_density in self._label_densities:
            return True
        return False

    def get(
        self,
        dataset_name: str,
        len_scale_factor: float = 1.0,
        label_density: float = 1.0,
    ) -> List[Counter[SemanticClassId]]:
        assert self.has(dataset_name, len_scale_factor, label_density)
        return self._data_scale_density \
            [dataset_name][len_scale_factor][label_density]
