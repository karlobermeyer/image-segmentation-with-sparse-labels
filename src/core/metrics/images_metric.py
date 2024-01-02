"""
This module provides `ImagesMetric`, a base type for computing pixel-based
semantic segmentation metrics from pairs of semantic segmentation maps (model
outputs and targets).
"""
# Standard
from abc import ABC, abstractmethod
from typing import List, Optional, Set

# Numerics
import numpy as np

# Machine Learning
import pandas as pd

# Other 3rd-Party
#import texttable  # https://pypi.org/project/texttable/

# Project-Specific
from core.identifiers import SemanticClassId
from core.semantic_class_mapping import SemanticClassMapping


class ImagesMetric(ABC):
    """
    Base type for computing pixel-based semantic segmentation metrics from pairs
    of semantic segmentation maps (model outputs and targets).
    """
    def __init__(
        self,
        class_mapping: SemanticClassMapping,
        class_ids_excluded: Optional[Set[SemanticClassId]] = None,
    ) -> None:
        """
        Initialize instance.

        Args:
            class_mapping: for converting between cooked and raw class IDs.
            class_ids_excluded: the (cooked) IDs of the subset of classes that
                should not be evaluated. The Cityscapes project, for example,
                has the convention that pixels with the background class are
                excluded from the evaluation of metrics.
        """
        self._class_mapping: SemanticClassMapping = class_mapping

        if class_ids_excluded is None:
            self._class_ids_excluded: Set[SemanticClassId] = set()
        else:
            self._class_ids_excluded: Set[SemanticClassId] = class_ids_excluded

        self._class_ids: List[SemanticClassId] = \
            sorted(set(class_mapping.class_ids) - self._class_ids_excluded)

    @property
    def class_mapping(self) -> SemanticClassMapping:
        return self._class_mapping

    @property
    def class_ids(self) -> List[SemanticClassId]:
        """
        Ascending list of (cooked) class IDs included in metrics.

        For a full list of all class IDs, use `self.class_mapping.class_ids`.
        """
        return self._class_ids

    @property
    def class_ids_excluded(self) -> Set[SemanticClassId]:
        """Set of (cooked) class IDs excluded from metrics."""
        return self._class_ids_excluded

    def compute_include_array(self, target: np.ndarray) -> np.ndarray:
        """
        Boolean numpy array, in correspondence with `target`, which encodes
        which pixels to count towards metrics.
        """
        include: np.ndarray = np.ones_like(target, dtype=bool)
        for class_id in self.class_ids_excluded:
            include = np.logical_and(include, target != class_id)
        return include

    @abstractmethod
    def update_class(
        self,
        class_id: SemanticClassId,
        model_output: np.ndarray,
        target: np.ndarray,
        include: Optional[bool] = None,
    ) -> None:
        """
        Update statistics for the single (cooked) class `class_id`.

        Args:
            class_id: cooked ID of class to update the statistics of.
            model_output: segmentation map of cooked int class IDs output by
                model to be evaluated.
            target: reference segmentation map of cooked int class IDs. This is
                what we evaluate `model_output` against.
            include: optionally precomputed Boolean numpy array which encodes
                which pixels to count towards metrics.
        """
        pass

    def update(
        self,
        model_output: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """
        Use this method to accumulate statistics for computing metrics.

        Args:
            model_output: segmentation map of cooked int class IDs output by
                model to be evaluated.
            target: reference segmentation map of cooked int class IDs. This is
                what we are evaluate `model_output` against.
        """
        for class_id in self.class_ids:
            self.update_class(
                class_id,
                model_output,
                target,
                include=self.compute_include_array(target),
            )

    @abstractmethod
    def __getitem__(self, class_id: SemanticClassId) -> float:
        """Compute and return the metric of (cooked) class `class_id`."""
        pass

    @abstractmethod
    def df(self) -> pd.DataFrame:
        """Construct and return a pandas dataframe of metrics."""
        pass

    def write_csv(self, filename: str) -> None:
        """Write pandas dataframe to a CSV file."""
        df: pd.DataFrame = self.df()
        df.to_csv(filename, index=False)

    def df_str(self) -> str:
        """Pandas dataframe as a string."""
        return self.df().to_string(index=False) + "\n"

    @abstractmethod
    def texttable_str(self) -> str:
        """Construct and return an ASCII table of metrics."""
        pass

    def write_txt(self, filename: str) -> None:
        """Write ASCII table of metrics to text file."""
        with open(filename, "w") as fout:
            fout.write(self.texttable_str())
