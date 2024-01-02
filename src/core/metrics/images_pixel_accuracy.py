"""
This module provides `ImagesPixelAccuracy`, an object for computing pixel
accuracies per class, mean pixel accuracy, and pixel accuracy from pairs of
semantic segmentation maps (model outputs and targets).

_References_

"Fully Convolutional Networks for Semantic Segmentation"
by J. Long and E. Shelhamer and T. Darrell, 2015
"""
# Standard
from math import isnan
from typing import Any, Dict, List, Optional, Set

# Numerics
import numpy as np

# Machine Learning
import pandas as pd

# Other 3rd-Party
import texttable  # https://pypi.org/project/texttable/

# Project-Specific
from core.identifiers import SemanticClassId
from core.semantic_class_mapping import SemanticClassMapping
from core.metrics.images_metric import ImagesMetric


class ImagesPixelAccuracy(ImagesMetric):
    """
    An object for computing pixel accuracies per class, mean pixel accuracy, and
    pixel accuracy from pairs of semantic segmentation maps (model outputs and
    targets).

    Assume all semantic classes are cooked unless otherwise specified as raw.

    Example Usage:
    ```
    images_pixel_accuracy: ImagesPixelAccuracy = ImagesPixelAccuracy(
        class_mapping,
        class_ids_excluded,
    )
    for model_output, target in pairs:
        images_pixel_accuracy.update(model_output, target)
    class_pixel_accuracies: Dict[SemanticClassId, float] = \
        images_pixel_accuracy.class_pixel_accuracies()
    mean_pixel_accuracy: float = \
        images_pixel_accuracy.mean_pixel_accuracy(class_pixel_accuracies)
    pixel_accuracy: float = \
        images_pixel_accuracy.pixel_accuracy()
    print(images_pixel_accuracy.texttable_str())
    ```
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
            class_ids_excluded: the cooked IDs of the subset of classes that
                should not be evaluated. The Cityscapes project, for example,
                has the convention that pixels with the background class are
                excluded from the evaluation of metrics.
        """
        super().__init__(class_mapping, class_ids_excluded)

        # For tallying the total number of pixels included in metrics. This is
        # the total number of pixels less the pixels which are labeled with an
        # excluded class.
        self._num_pixels: np.uint64 = np.uint64(0)

        # For tallying the total number of correctly classified pixels.
        self._num_pixels_correct: np.uint64 = np.uint64(0)

        # For tallying target pixels of each included class.
        self._class_num_pixels: Dict[SemanticClassId, np.uint64] = \
            { class_id: np.uint64(0) for class_id in self.class_ids }

        # For tallying model output pixels correctly classified with respect to
        # target pixels of each included class.
        self._class_num_pixels_correct: Dict[SemanticClassId, np.uint64] = \
            { class_id: np.uint64(0) for class_id in self.class_ids }

    def update_class(
        self,
        class_id: SemanticClassId,
        model_output: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """
        Update statistics for the single included class `class_id`.

        Args:
            class_id: cooked ID of class to update the statistics of.
            model_output: segmentation map of cooked int class IDs output by
                model to be evaluated.
            target: reference segmentation map of cooked int class IDs. This is
                what we evaluate `model_output` against.
        """
        assert class_id not in self.class_ids_excluded, "Included classes only!"
        target_bool: np.ndarray = target == class_id
        num_pixels: np.uint64 = np.sum(target_bool).astype(np.uint64)
        num_pixels_correct: np.uint64 = np.sum(
            np.logical_and(model_output == class_id, target_bool),
        ).astype(np.uint64)
        self._class_num_pixels[class_id] += num_pixels
        self._class_num_pixels_correct[class_id] += num_pixels_correct

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
        include: np.ndarray = self.compute_include_array(target)
        num_pixels: np.uint64 = np.sum(include).astype(np.uint64)
        self._num_pixels += num_pixels

        # Compute the total number of correct included pixels.
        correct_bool: np.ndarray = np.logical_and(
            model_output == target,
            include,
        )
        self._num_pixels_correct += np.sum(correct_bool).astype(np.uint64)

        # Update pixel counts for individual classes.
        for class_id in self.class_ids:
            self.update_class(
                class_id,
                model_output,
                target,
            )

    def __getitem__(self, class_id: SemanticClassId) -> float:
        """
        Compute and return the pixel accuracy with respect to an included
        class `class_id`.
        """
        assert class_id not in self.class_ids_excluded, "Included classes only!"
        numerator: np.uint64 = self._class_num_pixels_correct[class_id]
        denominator: np.uint64 = self._class_num_pixels[class_id]
        if denominator == 0:
            return float("nan")
        return float(numerator/denominator)

    def class_pixel_accuracies(self) -> Dict[SemanticClassId, float]:
        """Dictionary of class pixel accuracies."""
        return { class_id: self[class_id] for class_id in self.class_ids }

    def mean_pixel_accuracy(
        self,
        class_pixel_accuracies: Optional[Dict[SemanticClassId, float]] = None,
    ) -> float:
        """
        Compute and return mean pixel accuracy over included classes.

        Args:
            class_pixel_accuracies: optional precomputed class pixel accuracies.

        Returns:
            mean pixel accuracy
        """
        if class_pixel_accuracies is None:
            class_pixel_accuracies: Dict[SemanticClassId, float] = \
                self.class_pixel_accuracies()
        mean_pixel_accuracy: float = 0.0
        num_mean_contributors: int = 0
        for class_id, accuracy in class_pixel_accuracies.items():
            if not isnan(accuracy):
                mean_pixel_accuracy += accuracy
                num_mean_contributors += 1
        if 0 < num_mean_contributors:
            mean_pixel_accuracy /= num_mean_contributors
        else:
            mean_pixel_accuracy = float("nan")
        return mean_pixel_accuracy

    def pixel_accuracy(self) -> float:
        """
        Compute and return pixel accuracy, which is the total number of
        correctly classified pixels divided by the total number of pixels.

        Returns:
            pixel accuracy
        """
        if 0 < self._num_pixels:
            return float(self._num_pixels_correct/self._num_pixels)
        return float("nan")

    def df(self) -> pd.DataFrame:
        """Construct and return a pandas dataframe of pixel accuracies."""
        columns: List[str] = \
            ["class name", "class ID", "raw class ID", "pixel accuracy"]
        rows: List[List[Any]] = []
        class_pixel_accuracies: Dict[SemanticClassId, float] = \
            self.class_pixel_accuracies()
        mean_pixel_accuracy: float = \
            self.mean_pixel_accuracy(class_pixel_accuracies)
        pixel_accuracy: float = self.pixel_accuracy()
        for class_id in self.class_ids:
            class_name: str = self.class_mapping.class_names[class_id]
            class_id_raw: SemanticClassId = \
                self.class_mapping.class_ids_raw_included[class_id]
            accuracy: float = class_pixel_accuracies[class_id]
            rows.append([class_name, class_id, class_id_raw, accuracy])
        rows.append(["all", "", "", pixel_accuracy])
        rows.append(["mean", "", "", mean_pixel_accuracy])
        return pd.DataFrame(rows, columns=columns)

    def texttable_str(self) -> str:
        """Construct and return an ASCII table of pixel accuracies."""
        table = texttable.Texttable()
        table.set_cols_align(["r", "c", "c", "c"])  # l, c, r
        table.set_cols_valign(["c", "c", "c", "c"])  # t, m, b
        table.set_cols_dtype(["a", "i", "i", "f"])  # t, f, e, i, a
        rows: List[List[Any]] = [
            ["class name", "class ID", "raw class ID", "pixel accuracy"]
        ]
        class_pixel_accuracies: Dict[SemanticClassId, float] = \
            self.class_pixel_accuracies()
        mean_pixel_accuracy: float = \
            self.mean_pixel_accuracy(class_pixel_accuracies)
        pixel_accuracy: float = self.pixel_accuracy()
        for class_id in self.class_ids:
            row: List[Any] = []
            row.append(self.class_mapping.class_names[class_id])
            row.append(class_id)
            row.append(self.class_mapping.class_ids_raw_included[class_id])
            accuracy: float = class_pixel_accuracies[class_id]
            row.append(accuracy)
            rows.append(row)
        rows.append(["all", "", "", pixel_accuracy])
        rows.append(["mean", "", "", mean_pixel_accuracy])
        table.add_rows(rows)
        return table.draw() + "\n"
