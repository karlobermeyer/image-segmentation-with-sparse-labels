"""
This module provides `ImagesIou`, an object for computing IoU (Intersection over
Union) per class and mIoU (mean IoU) from pairs of semantic segmentation maps
(model outputs and targets).

_References_

"Fully Convolutional Networks for Semantic Segmentation"
by J. Long and E. Shelhamer and T. Darrell, 2015

Cityscapes evaluation script:
https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py
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


class ImagesIou(ImagesMetric):
    """
    An object for computing IoU (Intersection over Union) and mIoU (mean
    IoU) from pairs of semantic segmentation maps (model outputs and
    targets).

    Assume all semantic classes are cooked unless otherwise specified as raw.

    Example Usage:
    ```
    images_iou: ImagesIou = ImagesIou(
        class_mapping,
        class_ids_excluded,
    )
    for model_output, target in pairs:
        images_iou.update(model_output, target)
    class_ious: Dict[SemanticClassId, float] = images_iou.class_ious()
    miou: float = images_iou.miou(class_ious)
    print(images_iou.texttable_str())
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

        # For tallying intersection pixels with respect to each class.
        self._class_intersections: Dict[SemanticClassId, np.uint64] = \
            { class_id: np.uint64(0) for class_id in self.class_ids }

        # For tallying union pixels with respect to each class.
        self._class_unions: Dict[SemanticClassId, np.uint64] = \
            { class_id: np.uint64(0) for class_id in self.class_ids }

    def update_class(
        self,
        class_id: SemanticClassId,
        model_output: np.ndarray,
        target: np.ndarray,
        include: Optional[bool] = None,
    ) -> None:
        """
        Update statistics for the single class `class_id`.

        Args:
            class_id: cooked ID of class to update the statistics of.
            model_output: segmentation map of cooked int class IDs output by
                model to be evaluated.
            target: reference segmentation map of cooked int class IDs. This is
                what we evaluate `model_output` against.
            include: optionally precomputed Boolean numpy array which encodes
                which pixels to count towards metrics.
        """
        assert class_id not in self.class_ids_excluded, "Included classes only!"

        if include is None:
            include: np.ndarray = self.compute_include_array(target)

        model_output_bool: np.ndarray = np.logical_and(
            model_output == class_id,
            include,
        )
        target_bool: np.ndarray = target == class_id

        intersection: np.uint64 = np.sum(
            np.logical_and(model_output_bool, target_bool),
        ).astype(np.uint64)
        self._class_intersections[class_id] += intersection

        union: np.uint64 = np.sum(
            np.logical_or(model_output_bool, target_bool),
        ).astype(np.uint64)
        self._class_unions[class_id] += union

    def __getitem__(self, class_id: SemanticClassId) -> float:
        """Compute and return the IoU of an included class `class_id`."""
        assert class_id not in self.class_ids_excluded, "Included classes only!"
        numerator: np.uint64 = self._class_intersections[class_id]
        denominator: np.uint64 = self._class_unions[class_id]
        if denominator == 0:
            return float("nan")
        return float(numerator/denominator)

    def class_ious(self) -> Dict[SemanticClassId, float]:
        """Dictionary of class IoUs."""
        return { class_id: self[class_id] for class_id in self.class_ids }

    def miou(
        self,
        class_ious: Optional[Dict[SemanticClassId, float]] = None,
    ) -> float:
        """
        Compute and return mIoU over included classes.

        Args:
            class_ious: optional precomputed class IoUs.

        Returns:
            mIoU
        """
        if class_ious is None:
            class_ious: Dict[SemanticClassId, float] = self.class_ious()
        miou: float = 0.0
        num_mean_contributors: int = 0
        for class_id, iou in class_ious.items():
            if not isnan(iou):
                miou += iou
                num_mean_contributors += 1
        if 0 < num_mean_contributors:
            miou /= num_mean_contributors
        else:
            miou = float("nan")
        return miou

    def df(self) -> pd.DataFrame:
        """Construct and return a pandas dataframe of IoUs and mIoU."""
        columns: List[str] = \
            ["class name", "class ID", "raw class ID", "IoU"]
        rows: List[List[Any]] = []
        class_ious: Dict[SemanticClassId, float] = self.class_ious()
        miou: float = self.miou(class_ious)
        for class_id in self.class_ids:
            class_name: str = self.class_mapping.class_names[class_id]
            class_id_raw: SemanticClassId = \
                self.class_mapping.class_ids_raw_included[class_id]
            iou: float = class_ious[class_id]
            rows.append([class_name, class_id, class_id_raw, iou])
        rows.append(["mean", "", "", miou])
        return pd.DataFrame(rows, columns=columns)

    def texttable_str(self) -> str:
        """Construct and return an ASCII table of IoUs and mIoU."""
        table = texttable.Texttable()
        table.set_cols_align(["r", "c", "c", "c"])  # l, c, r
        table.set_cols_valign(["c", "c", "c", "c"])  # t, m, b
        table.set_cols_dtype(["a", "i", "i", "f"])  # t, f, e, i, a
        rows: List[List[Any]] = [
            ["class name", "class ID", "raw class ID", "IoU"]
        ]
        class_ious: Dict[SemanticClassId, float] = self.class_ious()
        miou: float = self.miou(class_ious)
        for class_id in self.class_ids:
            row: List[Any] = []
            row.append(self.class_mapping.class_names[class_id])
            row.append(class_id)
            row.append(self.class_mapping.class_ids_raw_included[class_id])
            iou: float = class_ious[class_id]
            row.append(iou)
            rows.append(row)
        rows.append(["mean", "", "", miou])
        table.add_rows(rows)
        return table.draw() + "\n"
