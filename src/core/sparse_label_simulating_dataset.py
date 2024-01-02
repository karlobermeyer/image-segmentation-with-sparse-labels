"""
This module provides `SparseLabelSimulatingDataset`, a semantic image
segmentation dataset type that can simulate sparse (blockwise-dense) labeling.
"""

# Standard
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import gc
import os
from typing import Callable, List, Optional, Tuple, Union

# Numerics
import numpy as np

# Image Processing
#import cv2
import imageio.v3 as imageio

# Machine Learning
import torch
from torch.utils.data import Dataset

import albumentations as albm  # Faster than `torchvision.transforms`.
from albumentations.pytorch import ToTensorV2

# Project-Specific
from core import SemanticClassId
from core.semantic_class_mapping import SemanticClassMapping


def str_to_ndarray(s_str: str, dtype=np.int32) -> np.ndarray:
    """Convert string to numpy array of Unicode char IDs."""
    return np.array([ord(c) for c in s_str], dtype=dtype)


def ndarray_to_str(s_ndarray: np.ndarray) -> str:
    """Convert numpy array of Unicode char IDs to string."""
    return "".join([chr(c) for c in s_ndarray])


def pack_sequences(
    sequences: Union[np.ndarray, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pack numpy arrays of Unicode char IDs into one big numpy array.

    Args:
        sequences: numpy array(s) of Unicode char IDs

    Returns:
        values: a single numpy array of all input arrays packed together.
        offsets: offsets for retrieving individual values.
    """
    values = np.concatenate(sequences, axis=0)
    offsets = np.cumsum([len(s) for s in sequences])
    return values, offsets


def retrieve_str(
    values: np.ndarray,
    offsets: np.ndarray,
    ix: int,
) -> np.ndarray:
    """Retrieve a value from numpy arrays packed by `pack_sequence`."""
    finish = offsets[ix]
    if ix > 0:
        start = offsets[ix - 1]
    elif ix == 0:
        start = 0
    else:
        raise ValueError(ix)
    s_ndarray: np.ndarray = values[start:finish]
    return ndarray_to_str(s_ndarray)


def get_target_unpreprocessed(
    class_mapping: SemanticClassMapping,
    label_density: float,
    ignore_index: int,
    filename: str,
    ix: int,
) -> np.ndarray:
    """
    Get a target as a numpy array with cooked (not raw) semantic class IDs.

    The array is unpreprocessed in the sense that the preprocessing for
    model input has not been applied. However, the target class IDs are
    cooked (not raw) and `ignore_index` has been applied according to
    `label_density`.

    This is implemented as a stand-alone function rather than a method of
    `SparseLabelSimulatingDataset` to speed up the parallelization of counting
    class occurrences over whole datasets.

    Args:
        class_mapping: semantic class mapping.
        label_density: blockwise-dense label density.
        ignore_index: index used to simulate unlabeled pixels.
        filename: target filename.
        ix: index of target in dataset. This determines where the dense block of
            labels appears.

    Returns:
        segmentation_map_int: target as numpy array.
    """
    segmentation_map_int_raw: np.ndarray = imageio.imread(filename)
    segmentation_map_int: np.ndarray = \
        class_mapping.segmentation_map_int_from_raw(
            segmentation_map_int_raw,
        )

    # Mark unobservable portion of segmentation map with ignore index.
    if label_density == 1.0:
        pass
    elif label_density == 0.5:
        width: int = segmentation_map_int.shape[1]
        if ix % 2 == 0:
            segmentation_map_int[:, width//2:] = ignore_index
        else:
            segmentation_map_int[:, :width//2] = ignore_index
    else:
        raise NotImplementedError

    return segmentation_map_int


def class_num_pixels_from_target_file(
    class_mapping: SemanticClassMapping,
    label_density: float,
    ignore_index: int,
    filename: str,
    ix: int,
) -> Counter[SemanticClassId]:
    """
    Count class occurrences in a single target file.

    This is implemented as a stand-alone function rather than a method of
    `SparseLabelSimulatingDataset` to speed up the parallelization of counting
    class occurrences over whole datasets.

    Args:
        class_mapping: semantic class mapping.
        label_density: blockwise-dense label density.
        ignore_index: index used to simulate unlabeled pixels.
        filename: target filename.
        ix: index of target in dataset. This determines where the dense block of
            labels appears.

    Returns:
        counter: class occurrences.
    """
    segmentation_map_int: np.ndarray = get_target_unpreprocessed(
        class_mapping=class_mapping,
        label_density=label_density,
        ignore_index=ignore_index,
        filename=filename,
        ix=ix,
    )
    counter: Counter[SemanticClassId] = Counter(segmentation_map_int.flatten())
    del counter[ignore_index]
    return counter


class SparseLabelSimulatingDataset(Dataset):
    """
    A semantic image segmentation dataset that can simulate sparse
    (blockwise-dense) labeling.

    Label sparsity simulation is accomplished by the getter methods
    automatically and consistently overwriting certain pixels of the target
    segmentation maps with a sentinel value called the `ignore_index` that can
    be recognized by loss functions during model training.

    In order to avoid a well-known problem of excessive memory consumption by
    multi-threaded dataloaders, filenames are stored not as lists of strings,
    but as numpy byte arrays. The helper functions above
    (`str_to_ndarray`,...,`retrieve_str`) make this possible.
    https://docs.aws.amazon.com/codeguru/detector-library/python/pytorch-data-loader-with-multiple-workers/
    https://github.com/pytorch/pytorch/issues/13246

    Args:
        image_filenames: image filenames in correspondence with
            `target_filenames.
        target_filenames: target segmentation map filenames in correspondence
            with `image_filenames`.
        class_mapping: semantic class mapping for converting between raw and
            cooked semantic class IDs.
        preprocess: transformations to apply to images and targets before
            returning from getters.
        label_density: portion of target pixels that are simulated as being
            observable.
        ignore_index: the sentinel value used to mark target pixels as
            unobservable.
        shuffle: whether to shuffle the data upon initialization.
        target_class_num_pixels: optional precomputed list of the class counts
            for each target. If provided, the elements of this list are in
            correspondence with `target_filenames`.
    """
    def __init__(
        self,
        image_filenames: List[str],
        target_filenames: List[str],
        class_mapping: SemanticClassMapping,
        preprocess: Optional[albm.Compose] = None,
        label_density: float = 1.0,
        ignore_index: int = 255,
        shuffle: bool = False,
        target_class_num_pixels: \
            Optional[List[Counter[SemanticClassId]]] = None,
    ) -> None:
        assert len(image_filenames) == len(target_filenames)

        self._len: int = len(image_filenames)

        ixs_shuffled: List[int] = list(range(self._len))
        if shuffle:
            np.random.shuffle(ixs_shuffled)

        image_filename_ndarrays: List[np.ndarray] = [
           str_to_ndarray(s_str=image_filenames[ix], dtype=np.int8)
           for ix in ixs_shuffled
        ]
        values_offsets: Tuple[np.ndarray, np.ndarray] = \
            pack_sequences(image_filename_ndarrays)
        self._image_filename_values: np.ndarray = values_offsets[0]
        self._image_filename_offsets: np.ndarray = values_offsets[1]

        target_filename_ndarrays: List[np.ndarray] = [
            str_to_ndarray(s_str=target_filenames[ix], dtype=np.int8)
            for ix in ixs_shuffled
        ]
        values_offsets: Tuple[np.ndarray, np.ndarray] = \
            pack_sequences(target_filename_ndarrays)
        self._target_filename_values: np.ndarray = values_offsets[0]
        self._target_filename_offsets: np.ndarray = values_offsets[1]

        self._class_mapping: SemanticClassMapping = class_mapping

        if preprocess is None:
            self._preprocess: albm.Compose = albm.Compose([ToTensorV2()])
        else:
            self._preprocess: albm.Compose = preprocess

        assert 0.0 <= label_density <= 1.0
        if label_density not in (0.5, 1.0):
            raise NotImplementedError
        self._label_density: float = label_density
        self._ignore_index: int = ignore_index

        self._target_class_num_pixels_cache: \
            Optional[List[Counter[SemanticClassId]]] = target_class_num_pixels

    @property
    def class_mapping(self) -> SemanticClassMapping:
        return self._class_mapping

    @property
    def label_density(self) -> float:
        return self._label_density

    @property
    def ignore_index(self) -> int:
        return self._ignore_index

    def get_image_filename(self, ix: int) -> str:
        return retrieve_str(
            self._image_filename_values,
            self._image_filename_offsets,
            ix,
        )

    def get_target_filename(self, ix: int) -> str:
        return retrieve_str(
            self._target_filename_values,
            self._target_filename_offsets,
            ix,
        )

    def get_filenames(self, ix: int) -> Tuple[str, str]:
        return self.get_image_filename(ix), self.get_target_filename(ix)

    def construct_image_filename_list(self) -> Tuple[List[str], List[str]]:
        return [ self.get_image_filename(ix) for ix in range(self._len) ]

    def construct_target_filename_list(self) -> Tuple[List[str], List[str]]:
        return [ self.get_target_filename(ix) for ix in range(self._len) ]

    def construct_filename_lists(self) -> Tuple[List[str], List[str]]:
        return self.construct_image_filename_list(), \
            self.construct_target_filename_list()

    def get_image_unpreprocessed(
        self,
        ix: int,
    ) -> np.ndarray:
        """
        Get an image as a numpy array.

        The array is unpreprocessed in the sense that the preprocessing for
        model input has not been applied.
        """
        filename: str = self.get_image_filename(ix)
        image: np.ndarray = imageio.imread(filename)
        return image

    def get_target_unpreprocessed(
        self,
        ix: int,
    ) -> np.ndarray:
        """
        Get a target as a numpy array with cooked (not raw) semantic class IDs.

        The array is unpreprocessed in the sense that the preprocessing for
        model input has not been applied. However, the target class IDs are
        cooked (not raw) and `ignore_index` has been applied according to
        `label_density`.
        """
        filename: str = self.get_target_filename(ix)
        segmentation_map_int: np.ndarray = get_target_unpreprocessed(
            class_mapping=self.class_mapping,
            label_density=self.label_density,
            ignore_index=self.ignore_index,
            filename=filename,
            ix=ix,
        )
        return segmentation_map_int

    def get_unpreprocessed(self, ix: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get an (image, target) pair of numpy arrays with cooked (not raw)
        semantic class IDs.

        The arrays are unpreprocessed in the sense that the preprocessing for
        model input has not been applied. However, the target class IDs are
        cooked (not raw) and `ignore_index` has been applied according to
        `label_density`.
        """
        return self.get_image_unpreprocessed(ix), \
            self.get_target_unpreprocessed(ix)

    def _compute_target_class_num_pixels(self) -> None:
        """
        Compute and cache a list of class occurrences for all targets.

        So that sparse labeling is properly simulated, use this method instead
        of similar EDA functions and classes.

        Keeping the pixel counts of different targets separate in the output is
        useful for finding example images that feature certain classes.

        Returns:
            target_class_num_pixels: list of class occurrences for all targets.
                The list runs over dataset examples. Each element is a `Counter`
                object, where keys are semantic class IDs and values are integer
                numbers of class occurrences in the respective target.
        """
        num_workers_ubnd = max(1, 3*os.cpu_count()//4)
        gc.collect()
        target_filenames: List[str] = self.construct_target_filename_list()
        class_num_pixels_from_target_file_: \
            Callable[[str, int], Counter[SemanticClassId]] = partial(
                class_num_pixels_from_target_file,
                self.class_mapping,
                self.label_density,
                self.ignore_index,
            )
        with ProcessPoolExecutor(max_workers=num_workers_ubnd) as executor:
            target_class_num_pixels: List[Counter[SemanticClassId]] = \
                list(executor.map(
                    class_num_pixels_from_target_file_,
                    target_filenames,
                    range(self._len),
                ))
        self._target_class_num_pixels_cache = target_class_num_pixels

    @property
    def target_class_num_pixels(self) -> List[Counter[SemanticClassId]]:
        """
        Return list of the class counts for each target, computing if not
        already cached.

        Beware this is expensive. It can take a few minutes if your dataset has
        thousands of examples.
        """
        if self._target_class_num_pixels_cache is None:
            self._compute_target_class_num_pixels()
        return self._target_class_num_pixels_cache

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, ix: int) -> Tuple[np.ndarray, np.ndarray]:
        datapair_unpreprocessed: Tuple[np.ndarray, np.ndarray] = \
            self.get_unpreprocessed(ix)
        image: np.ndarray = datapair_unpreprocessed[0]
        segmentation_map_int: np.ndarray = datapair_unpreprocessed[1]

        preprocessed = self._preprocess(image=image, mask=segmentation_map_int)
        preprocessed_image: torch.Tensor = preprocessed["image"]
        preprocessed_mask: torch.Tensor = preprocessed["mask"]

        return preprocessed_image, preprocessed_mask
