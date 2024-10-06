"""
SubCityscapes and Cityscapes semantic image segmentation sparse label simulating
dataset factories.
"""

# Standard
from collections import Counter
import os
from typing import List, Optional

# Numerics
import numpy as np

# Machine Learning
# `Cityscapes` is defined in
# `./...-env/lib/python3.9/site-packages/torchvision/datasets/cityscapes.py`.
# Cf. https://pytorch.org/vision/stable/generated/torchvision.datasets.Cityscapes.html#torchvision.datasets.Cityscapes
from torchvision.datasets import Cityscapes
import albumentations as albm  # Faster than `torchvision.transforms`.

# Project-Specific
from core import (
    SemanticClassId,
    SparseLabelSimulatingDataset,
)
from data.cityscapes.semantic_class_mapping import class_mapping


PROJECT_ROOT: str = os.environ.get("PROJECT_ROOT")
assert PROJECT_ROOT is not None, \
    "PROJECT_ROOT not found! Did you run `setenv.sh` before serving the notebook?"
DATA_ROOT_DIRNAME: str = os.path.join(PROJECT_ROOT, "data/cityscapes")

# These shuffled Cityscapes training indices are used for splitting
# `cityscapes_train` into `subcityscapes_train` and `subcityscapes_val`.
SHUFFLED_CITYSCAPES_TRAIN_IXS: np.ndarray = np.loadtxt(
    fname=os.path.join(
        PROJECT_ROOT,
        "src/data/cityscapes/",
        "shuffled_cityscapes_train_ixs.csv",
    ),
    dtype=np.uint64,
)


def subcityscapes_dataset(
    split: str,  # "train", "val", "trainval", or "test".
    preprocess: Optional[albm.Compose] = None,
    len_scale_factor: float = 1.0,
    label_density: float = 1.0,
    ignore_index: int = 255,
    shuffle: bool = False,
    target_class_num_pixels: \
        Optional[List[Counter[SemanticClassId]]] = None,
) -> SparseLabelSimulatingDataset:
    """
    Construct a SubCityscapes semantic image segmentation sparse label
    simulating dataset.

    Args:
        split: "train", "val", "trainval", or "test".
          "train" => use 2475 examples (71.2 %) of Cityscapes train set.
          "val" => use other 500 examples (14.4 %) of Cityscapes train set.
          "trainval" => use all 2975 examples of Cityscapes train set.
          "test" => use all 500 examples of Cityscapes val set.
        preprocess: preprocessing for model input.
        len_scale_factor: reduce dataset to this times original size.
        label_density: portion of target pixels to mark with the sentinel value
            `ignore_index` to prevent the model from seeing it.
        ignore_index: sentinel value used to mark parts of data examples that
            the model should never see for training.
        shuffle: whether to shuffle the dataset.
        target_class_num_pixels: optional precomputed list of the class counts
            for each target. If provided, the elements of this list are in
            correspondence with the dataset indices.

    Returns:
        A sparse label simulating dataset.
    """
    assert split in ("train", "val", "trainval", "test")

    # Extract filenames from `Cityscapes` instance(s).
    if split in ("train", "val", "trainval"):
        cityscapes_train: Cityscapes = Cityscapes(
            root=DATA_ROOT_DIRNAME,
            split="train",
            mode="fine",
            target_type="semantic",
            transform=None,
            target_transform=None,
            transforms=None,
        )

        assert len(cityscapes_train) == 2975
        if split == "train":
            ixs: np.ndarray = SHUFFLED_CITYSCAPES_TRAIN_IXS[:2475]
        elif split == "val":
            ixs: np.ndarray = SHUFFLED_CITYSCAPES_TRAIN_IXS[-500:]
        else:  # split == "trainval"
            ixs: np.ndarray = np.array(range(2975), dtype=np.uint64)

        image_filenames: List[str] = [
            cityscapes_train.images[ix] for ix in ixs
        ]
        target_filenames: List[str] = [
            cityscapes_train.targets[ix][0] for ix in ixs
        ]

    else:  # split == "test"
        cityscapes_val: Cityscapes = Cityscapes(
            root=DATA_ROOT_DIRNAME,
            split="val",
            mode="fine",
            target_type="semantic",
            transform=None,
            target_transform=None,
            transforms=None,
        )

        assert len(cityscapes_val) == 500
        ixs: np.ndarray = np.array(range(500), dtype=np.uint64)

        image_filenames: List[str] = [
            cityscapes_val.images[ix] for ix in ixs
        ]
        target_filenames: List[str] = [
            cityscapes_val.targets[ix][0] for ix in ixs
        ]

    len_dataset: int = len(image_filenames)
    ixs_shuffled: List[int] = list(range(len_dataset))
    if shuffle:
        np.random.shuffle(ixs_shuffled)

    len_dataset_reduced: int = round(len_scale_factor*len_dataset)
    image_filenames_shuffled: List[str] = [
        image_filenames[ix] for ix in ixs_shuffled[:len_dataset_reduced]
    ]
    target_filenames_shuffled: List[str] = [
        target_filenames[ix] for ix in ixs_shuffled[:len_dataset_reduced]
    ]

    return SparseLabelSimulatingDataset(
        image_filenames=image_filenames_shuffled,
        target_filenames=target_filenames_shuffled,
        class_mapping=class_mapping,
        preprocess=preprocess,
        label_density=label_density,
        ignore_index=ignore_index,
        shuffle=False,
        target_class_num_pixels=target_class_num_pixels,
    )


def cityscapes_dataset(
    split: str,  # "train", "val", "trainval", or "test".
    preprocess: Optional[albm.Compose] = None,
    len_scale_factor: float = 1.0,
    label_density: float = 1.0,
    ignore_index: int = 255,
    shuffle: bool = False,
    target_class_num_pixels: \
        Optional[List[Counter[SemanticClassId]]] = None,
) -> SparseLabelSimulatingDataset:
    """
    Construct a Cityscapes semantic image segmentation sparse label simulating
    dataset.

    Args:
        split: "train", "val", "trainval", or "test".
        preprocess: preprocessing for model input.
        len_scale_factor: reduce dataset to this times original size.
        label_density: portion of target pixels to mark with the sentinel value
            `ignore_index` to prevent the model from seeing it.
        ignore_index: sentinel value used to mark parts of data examples that
            the model should never see for training.
        shuffle: whether to shuffle the dataset.
        target_class_num_pixels: optional precomputed list of the class counts
            for each target. If provided, the elements of this list are in
            correspondence with the dataset indices.

    Returns:
        A sparse label simulating dataset.
    """
    assert split in ("train", "val", "trainval", "test")

    # Extract filenames from `Cityscapes` instance(s).
    if split in ("train", "val", "test"):
        cityscapes: Cityscapes = Cityscapes(
            root=DATA_ROOT_DIRNAME,
            split=split,
            mode="fine",
            target_type="semantic",
            transform=None,
            target_transform=None,
            transforms=None,
        )
        image_filenames: List[str] = [
            cityscapes.images[ix] for ix in range(len(cityscapes))
        ]
        target_filenames: List[str] = [
            cityscapes.targets[ix][0] for ix in range(len(cityscapes))
        ]
    else:  # split == "trainval"
        cityscapes_train: Cityscapes = Cityscapes(
            root=DATA_ROOT_DIRNAME,
            split="train",
            mode="fine",
            target_type="semantic",
            transform=None,
            target_transform=None,
            transforms=None,
        )
        cityscapes_val: Cityscapes = Cityscapes(
            root=DATA_ROOT_DIRNAME,
            split="val",
            mode="fine",
            target_type="semantic",
            transform=None,
            target_transform=None,
            transforms=None,
        )
        image_filenames: List[str] = []
        target_filenames: List[str] = []
        for ix in range(len(cityscapes_train)):
            image_filenames.append(cityscapes_train.images[ix])
            target_filenames.append(cityscapes_train.targets[ix][0])
        for ix in range(len(cityscapes_val)):
            image_filenames.append(cityscapes_val.images[ix])
            target_filenames.append(cityscapes_val.targets[ix][0])

    len_dataset: int = len(image_filenames)
    ixs_shuffled: List[int] = list(range(len_dataset))
    if shuffle:
        np.random.shuffle(ixs_shuffled)

    len_dataset_reduced: int = round(len_scale_factor*len_dataset)
    image_filenames_shuffled: List[str] = [
        image_filenames[ix] for ix in ixs_shuffled[:len_dataset_reduced]
    ]
    target_filenames_shuffled: List[str] = [
        target_filenames[ix] for ix in ixs_shuffled[:len_dataset_reduced]
    ]

    return SparseLabelSimulatingDataset(
        image_filenames=image_filenames_shuffled,
        target_filenames=target_filenames_shuffled,
        class_mapping=class_mapping,
        preprocess=preprocess,
        label_density=label_density,
        ignore_index=ignore_index,
        shuffle=False,
        target_class_num_pixels=target_class_num_pixels,
    )
