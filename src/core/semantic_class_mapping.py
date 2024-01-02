from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from core.identifiers import SemanticClassId
from core.visualization.distinct_colors import distinct_colors_rgb


class SemanticClassMapping:
    """
    An object for storing semantic class IDs, colors, and converting class IDs
    and segmentation maps between various formats.

    A *raw* class ID is an integer representing a segmentation class as it
    appears in the raw labels of a dataset. In using a dataset, we generally (1)
    are only interested in a subset of the classes, (2) want to redefine class
    IDs as a continguous block of integers starting at 0 so that they are in
    1-to-1 correspondence with vector indices. We call the new class IDs *cooked
    class IDs*. The following examples illustrate these concepts.

    Example 1:
    The Cityscapes dataset for semantic image segmentation has the
    34 *raw class IDs*
      [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33].
    However, it is common to only train on the subset of 20 raw IDs
      [0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31,
       32, 33],
    where 0 is the "unlabeled" class (also referred to as the "background" class
    in some projects) that signifies the complement of the set of all other
    classes. To construct cooked class IDs, we assign 0 to 0, 1 to 7, 2 to 8,
    etc. The full list of 20 *cooked class IDs* is thus
      [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19].

    Example 2:
    The COCO-Stuff dataset for semantic image segmentation has the
    183 *raw class IDs*
      [255, 0, 1, 2, 3,..., 181],
    where 255 is the "background" class that signifies the complement of the set
    of all other classes. To construct cooked class IDs, we assign 0 to 255, 1
    to 0, 1 to 2, etc. The full list of 183 *cooked class IDs* is thus
      [0, 1, 2, 3, 4,..., 182]

    Since mostly we work with cooked class IDs, we assume a class ID is cooked
    unless otherwise specified as "raw" in variable/attribute/method names.

    TODO: This class is a mix of code that is applicable to
        (1) general semantic segmentation, and
        (2) semantic segmentation specifically of 2D images.
    Consider separating at some point so that general code can be reused.
    """
    def __init__(
        self,
        class_ids_raw: np.ndarray,  # Usu. dtype=np.int16
        class_ids_raw_included: np.ndarray,  # Usu. dtype=np.int16
        class_names_included: List[str],
        class_colors_rgb_included: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize instance.

        Args:
            class_ids_raw: numpy array of the full set of raw class IDs that
                appear in the raw labels of the dataset.
            class_ids_raw_included: numpy array of the subset of raw class IDs
                that are to be used in training and inference. As a convention,
                include the unlabeled/background raw class ID (usu. 0 or 255) in
                the first position even though, strictly speaking, it's cooked
                definition is different (it's the union of all excluded
                classes).
            class_names_included: List of class names in correspondence with
                `class_ids_raw_included`.
            class_colors_rgb_included: Optional numpy array where rows run over
                class IDs in correspondence with `class_ids_raw_included`, and
                columns run over RGB colors. If not provided, an array of
                default distinct colors is automatically constructed.
        """
        num_classes_included = len(class_ids_raw_included)
        assert len(class_names_included) == num_classes_included
        if class_colors_rgb_included is not None:
            assert len(class_colors_rgb_included) == num_classes_included

        self._class_ids_raw: np.ndarray = class_ids_raw
        self._class_ids_raw.flags.writeable = False
        self._class_ids_raw_included: np.ndarray = class_ids_raw_included
        self._class_ids_raw_included.flags.writeable = False

        # Construct the cooked class IDs.
        self._class_ids: np.ndarray = \
            np.array(range(num_classes_included), dtype=np.uint8)
        self._class_ids.flags.writeable = False

        self._class_names: List[str] = class_names_included
        self._class_names_dict: Dict[int, str] = {
            ix: name for ix, name in enumerate(self._class_names)
        }

        if class_colors_rgb_included is None:
            self._class_colors_rgb = \
                np.zeros((num_classes_included, 3), dtype=np.uint8)
            n = len(distinct_colors_rgb)
            for ix_class in range(len(class_ids_raw_included)):
                self._class_colors_rgb[ix_class] = \
                    distinct_colors_rgb[ix_class % n]
        else:
            self._class_colors_rgb: np.ndarray = class_colors_rgb_included
        self._class_colors_rgb.flags.writeable = False
        self._class_colors_bgr: np.ndarray = np.fliplr(self._class_colors_rgb)
        self._class_colors_bgr.flags.writeable = False

        # Precomputed helpers for mapping between raw and cooked class IDs.
        self._class_id_from_raw: Dict[SemanticClassId, SemanticClassId] = \
            dict(zip(
                class_ids_raw_included,
                range(len(class_ids_raw_included)),
            ))
        self._class_id_to_raw: Dict[SemanticClassId, SemanticClassId] = \
            dict(zip(
                range(len(class_ids_raw_included)),
                class_ids_raw_included,
            ))

    @property
    def class_ids_raw(self) -> np.ndarray:
        """Raw class IDs as they appear in the raw labels of the dataset."""
        return self._class_ids_raw

    @property
    def num_classes_raw(self) -> int:
        """Number of raw integer class IDs."""
        return len(self._class_ids_raw)

    @property
    def class_ids(self) -> np.ndarray:
        """Integer cooked class IDs."""
        return self._class_ids

    @property
    def num_classes(self) -> int:
        """Number of integer cooked class IDs."""
        return len(self._class_ids)

    @property
    def class_ids_raw_included(self) -> np.ndarray:
        """Included raw class IDs in correspondence with `self.class_ids`."""
        return self._class_ids_raw_included

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def class_names_dict(self) -> Dict[SemanticClassId, str]:
        return self._class_names_dict

    @property
    def class_colors_rgb(self) -> np.ndarray:
        return self._class_colors_rgb

    @property
    def class_colors_bgr(self) -> np.ndarray:
        return self._class_colors_bgr

    def segmentation_map_int_from_raw(
        self,
        segmentation_map_int_raw: np.ndarray,
    ) -> np.ndarray:
        """
        Convert raw segmentation map to a version that uses cooked class IDs
        instead of raw class IDs.

        Args:
            segmentation_map_int_raw: The segmentation map as a `np.ndarray`
                with `shape=(height, width)`, `dtype=np.uint8`, and where each
                element is a raw class ID.

        Returns:
            segmentation_map_int: The segmentation map as a `np.ndarray`
                with `shape=(height, width)`, `dtype=np.uint8`, and where each
                element is a cooked class ID.
        """
        segmentation_map_int = \
            np.zeros_like(segmentation_map_int_raw, dtype=np.uint8)
        for class_id_raw in self._class_ids_raw_included[1:]:
            segmentation_map_int[segmentation_map_int_raw == class_id_raw] = \
                self._class_id_from_raw[class_id_raw]
        return segmentation_map_int

    def segmentation_map_int_to_raw(
        self,
        segmentation_map_int: np.ndarray,
    ) -> np.ndarray:
        """
        Convert segmentation map to a version that uses raw class IDs instead of
        cooked class IDs.

        You may need to convert your inferred segmentation maps back to raw
        class IDs for submission to the test servers of open dataset projects.

        Args:
            segmentation_map_int: The segmentation map as a `np.ndarray`
                with `shape=(height, width)`, `dtype=np.uint8`, and where each
                element is a cooked class ID.

        Returns:
            segmentation_map_int_raw: The segmentation map as a `np.ndarray`
                with `shape=(height, width)`, `dtype=np.uint8`, and where each
                element is a raw class ID.
        """
        segmentation_map_int_raw = self._class_ids_raw_included[0] \
            *np.ones_like(segmentation_map_int, dtype=np.uint8)
        for class_id in self._class_ids[1:]:
            segmentation_map_int_raw[
                segmentation_map_int == class_id
            ] = self._class_id_to_raw[class_id]
        return segmentation_map_int_raw

    def segmentation_map_int_from_logits(
        self,
        segmentation_map_logits: torch.Tensor,
    ) -> np.ndarray:
        """
        Convert segmentation map tensor output by model (pre-softmax) to an
        array of cooked class IDs.

        Args:
            segmentation_map_logits: The segmentation map as the pre-softmax
                model outputs `torch.Tensor` with
                `shape=torch.Size([num_classes, height, width])` and
                `dtype=torch.float32`.

        Returns:
            segmentation_map_int: The segmentation map as a `np.ndarray`
                with `shape=(height, width)`, `dtype=np.uint8`, and where each
                element is a cooked class ID.
        """
        if len(segmentation_map_logits.shape) != 3:
            raise NotImplementedError("Currently only rank 3 tensors allowed!")
        return torch.argmax(segmentation_map_logits, dim=0) \
            .detach().cpu().numpy()

    def segmentation_map_int_from_categorical(
        self,
        segmentation_map_categorical: torch.Tensor,
    ) -> np.ndarray:
        """
        Convert segmentation map is a tensor of categorical distributions to an
        array of cooked class IDs.

        Args:
            segmentation_map_categorical: The segmentation map as a
                `torch.Tensor` with
                `shape=torch.Size([num_classes, height, width])` and
                `dtype=torch.float32`, and where each fiber `[:, i, j]`
                represents a categorical distribution over classes
                at pixel (i, j).

        Returns:
            segmentation_map_int: The segmentation map as a `np.ndarray`
                with `shape=(height, width)`, `dtype=np.uint8`, and where each
                element is a cooked class ID.
        """
        if len(segmentation_map_categorical.shape) != 3:
            raise NotImplementedError("Currently only rank 3 tensors allowed!")
        return torch.argmax(segmentation_map_categorical, dim=0) \
            .detach().cpu().numpy()

    def segmentation_map_int_from_rgb(
        self,
        segmentation_map_rgb: np.ndarray,
    ) -> np.ndarray:
        """
        Convert segmentation map from RGB array encoding to an array of cooked
        class IDs.

        Most datasets don't require this, but Pascal VOC does, for example.

        Args:
            segmentation_map_rgb: The segmentation map as an RGB image
                `np.ndarray` with `shape=(height, width, 3)` and
                `dtype=np.uint8`.

        Returns:
            segmentation_map_int: The segmentation map as a `np.ndarray`
                with `shape=(height, width)`, `dtype=np.uint8`, and where each
                element is a cooked class ID.
        """
        # TODO: Finish this if/when a dataset requires it.
        #return segmentation_map_int
        raise NotImplementedError

    def segmentation_map_categorical_from_logits(
        self,
        segmentation_map_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply softmax to convert segmentation map to a tensor of categorical
        distributions from a tensor of logits.

        Args:
            segmentation_map_logits: The segmentation map as the pre-softmax
                model outputs `torch.Tensor` with
                `shape=torch.Size([num_classes, height, width])` and
                `dtype=torch.float32`.

        Returns:
            segmentation_map_categorical: The segmentation map as a
                `torch.Tensor` with
                `shape=torch.Size([num_classes, height, width])` and
                `dtype=torch.float32`, and where each fiber `[:, i, j]`
                represents a categorical distribution over classes
                at pixel (i, j).
        """
        return torch.softmax(segmentation_map_logits, dim=0)

    def segmentation_map_categorical_from_int(
        self,
        segmentation_map_int: np.ndarray,
    ) -> torch.Tensor:
        """
        Convert segmentation map from an array of cooked class IDs to a tensor
        of categorical distributions.

        Args:
            segmentation_map_int: The segmentation map as a `np.ndarray`
                with `shape=(height, width)`, `dtype=np.uint8`, and where each
                element is a cooked class ID.

        Returns:
            segmentation_map_categorical: The segmentation map as a
                `torch.Tensor` with
                `shape=torch.Size([num_classes, height, width])` and
                `dtype=torch.float32`, and where each fiber `[:, i, j]`
                represents a categorical distribution over classes
                at pixel (i, j).
        """
        segmentation_map_categorical: torch.Tensor = F.one_hot(
            torch.tensor(segmentation_map_int, dtype=torch.long),
            num_classes=self.num_classes,
        ).permute(2, 0, 1).float()
        return segmentation_map_categorical

    def segmentation_map_categorical_from_raw(
        self,
        segmentation_map_int_raw: np.ndarray,
    ) -> torch.Tensor:
        """
        Convert segmentation map from an array of raw class IDs to a tensor of
        categorical distributions.

        Args:
            segmentation_map_int_raw: The segmentation map as a `np.ndarray`
                with `shape=(height, width)`, `dtype=np.uint8`, and where each
                element is a raw class ID.

        Returns:
            segmentation_map_categorical: The segmentation map as a
                `torch.Tensor` with
                `shape=torch.Size([num_classes, height, width])` and
                `dtype=torch.float32`, and where each fiber `[:, i, j]`
                represents a categorical distribution over classes
                at pixel (i, j).
        """
        segmentation_map_int: np.ndarray = \
            self.segmentation_map_int_from_raw(segmentation_map_int_raw)
        segmentation_map_categorical: torch.Tensor = \
            self.segmentation_map_categorical_from_int(segmentation_map_int)
        return segmentation_map_categorical

    def segmentation_map_rgb_from_int(
        self,
        segmentation_map_int: np.ndarray,
        ignore_index: Optional[int] = 255,
    ) -> np.ndarray:
        """
        Convert (cooked) int class IDs to RGB colors.

        If `segmentation_map_int` is a `torch.Tensor`, convert to numpy array
        before passing to this function using `segmentation_map_int.numpy()`.

        Args:
            segmentation_map_int: The segmentation map as a `np.ndarray`
                with `shape=(height, width)`, `dtype=np.uint8`, and where each
                element is a cooked class ID.
            ignore_index: any pixels with this label are colored white. The
                intended purpose of this sentinel value is to mark regions that
                are to be ignored by loss functions during model training.

        Returns:
            segmentation_map_rgb: The segmentation map as an RGB image
                `np.ndarray` with `shape=(height, width, 3)` and
                `dtype=np.uint8`.
        """
        red: np.ndarray = np.zeros_like(segmentation_map_int).astype(np.uint8)
        green: np.ndarray = np.zeros_like(segmentation_map_int).astype(np.uint8)
        blue: np.ndarray = np.zeros_like(segmentation_map_int).astype(np.uint8)
        for class_id in self.class_ids:
            ix = segmentation_map_int == class_id
            red[ix] = self.class_colors_rgb[class_id][0]
            green[ix] = self.class_colors_rgb[class_id][1]
            blue[ix] = self.class_colors_rgb[class_id][2]
        if ignore_index is not None:  # Ignored regions are colored white.
            ix = segmentation_map_int == ignore_index
            red[ix] = 255
            green[ix] = 255
            blue[ix] = 255
        segmentation_map_rgb = np.stack([red, green, blue], axis=2)
        return segmentation_map_rgb
