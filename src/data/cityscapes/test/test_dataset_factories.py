import os
from typing import Dict

import albumentations as albm

from core import SparseLabelSimulatingDataset
from data.cityscapes.dataset_factories import (
    subcityscapes_dataset,
    cityscapes_dataset,
)
from data.cityscapes.preprocesses import preprocesses_from


REPO_ROOT: str = os.environ.get("REPO_ROOT")
assert REPO_ROOT is not None, \
    "REPO_ROOT not found! Did you run `setenv.sh` before serving the notebook?"
#DATA_ROOT_DIRNAME: str = os.path.join(REPO_ROOT, "data/cityscapes")

IGNORE_INDEX: int = 255

PREPROCESSES: Dict[str, albm.Compose] = preprocesses_from(
    input_height=320,
    input_width=640,
    mean_for_input_normalization=(0.485, 0.456, 0.406),
    std_for_input_normalization=(0.229, 0.224, 0.225),
    do_shift_scale_rotate=True,
    ignore_index=IGNORE_INDEX,
)


def test_subcityscapes_dataset():
    dataset_train: SparseLabelSimulatingDataset = subcityscapes_dataset(
        split="train",
        preprocess=PREPROCESSES["train"],
        len_scale_factor=1.0,
        label_density=1.0,
        ignore_index=IGNORE_INDEX,
    )
    assert len(dataset_train) == 2475

    dataset_val: SparseLabelSimulatingDataset = subcityscapes_dataset(
        split="val",
        preprocess=PREPROCESSES["infer"],
        len_scale_factor=1.0,
        label_density=1.0,
        ignore_index=IGNORE_INDEX,
    )
    assert len(dataset_val) == 500

    dataset_test: SparseLabelSimulatingDataset = subcityscapes_dataset(
        split="test",
        preprocess=PREPROCESSES["infer"],
        len_scale_factor=1.0,
        label_density=1.0,
        ignore_index=IGNORE_INDEX,
    )
    assert len(dataset_test) == 500

    dataset_trainval: SparseLabelSimulatingDataset = subcityscapes_dataset(
        split="trainval",
        preprocess=PREPROCESSES["train"],
        len_scale_factor=1.0,
        label_density=1.0,
        ignore_index=IGNORE_INDEX,
    )
    assert len(dataset_trainval) == 2975

    dataset_train: SparseLabelSimulatingDataset = subcityscapes_dataset(
        split="train",
        preprocess=PREPROCESSES["train"],
        len_scale_factor=0.5,
        label_density=1.0,
        ignore_index=IGNORE_INDEX,
    )
    assert len(dataset_train) == round(0.5*2475)

    dataset_train: SparseLabelSimulatingDataset = subcityscapes_dataset(
        split="train",
        preprocess=PREPROCESSES["train"],
        len_scale_factor=0.5,
        label_density=0.5,
        ignore_index=IGNORE_INDEX,
    )

    # Pick a random target.
    _, target = dataset_train[13]
    ix0: int = target.shape[0]//2
    ix1_l: int = round(0.33*target.shape[1])
    ix1_r: int = round(0.66*target.shape[1])

    assert (target[ix0, ix1_l] == IGNORE_INDEX
        and target[ix0, ix1_r] != IGNORE_INDEX) or \
        (target[ix0, ix1_l] != IGNORE_INDEX
        and target[ix0, ix1_r] == IGNORE_INDEX)

    # Pick another random target.
    _, target = dataset_train[500]
    ix0: int = target.shape[0]//2
    ix1_l: int = round(0.33*target.shape[1])
    ix1_r: int = round(0.66*target.shape[1])

    assert (target[ix0, ix1_l] == IGNORE_INDEX
        and target[ix0, ix1_r] != IGNORE_INDEX) or \
        (target[ix0, ix1_l] != IGNORE_INDEX
        and target[ix0, ix1_r] == IGNORE_INDEX)


def test_cityscapes_dataset():
    dataset_train: SparseLabelSimulatingDataset = cityscapes_dataset(
        split="train",
        preprocess=PREPROCESSES["train"],
        len_scale_factor=1.0,
        label_density=1.0,
        ignore_index=IGNORE_INDEX,
    )
    assert len(dataset_train) == 2975

    dataset_val: SparseLabelSimulatingDataset = cityscapes_dataset(
        split="val",
        preprocess=PREPROCESSES["infer"],
        len_scale_factor=1.0,
        label_density=1.0,
        ignore_index=IGNORE_INDEX,
    )
    assert len(dataset_val) == 500

    dataset_test: SparseLabelSimulatingDataset = cityscapes_dataset(
        split="test",
        preprocess=PREPROCESSES["infer"],
        len_scale_factor=1.0,
        label_density=1.0,
        ignore_index=IGNORE_INDEX,
    )
    assert len(dataset_test) == 1525

    dataset_trainval: SparseLabelSimulatingDataset = cityscapes_dataset(
        split="trainval",
        preprocess=PREPROCESSES["train"],
        len_scale_factor=1.0,
        label_density=1.0,
        ignore_index=IGNORE_INDEX,
    )
    assert len(dataset_trainval) == 2975 + 500

    dataset_train: SparseLabelSimulatingDataset = cityscapes_dataset(
        split="train",
        preprocess=PREPROCESSES["train"],
        len_scale_factor=0.5,
        label_density=1.0,
        ignore_index=IGNORE_INDEX,
    )
    assert len(dataset_train) == round(0.5*2975)

    dataset_train: SparseLabelSimulatingDataset = cityscapes_dataset(
        split="train",
        preprocess=PREPROCESSES["train"],
        len_scale_factor=0.5,
        label_density=0.5,
        ignore_index=IGNORE_INDEX,
    )

    # Pick a random target.
    _, target = dataset_train[13]
    ix0: int = target.shape[0]//2
    ix1_l: int = round(0.33*target.shape[1])
    ix1_r: int = round(0.66*target.shape[1])

    assert (target[ix0, ix1_l] == IGNORE_INDEX
        and target[ix0, ix1_r] != IGNORE_INDEX) or \
        (target[ix0, ix1_l] != IGNORE_INDEX
        and target[ix0, ix1_r] == IGNORE_INDEX)

    # Pick another random target.
    _, target = dataset_train[500]
    ix0: int = target.shape[0]//2
    ix1_l: int = round(0.33*target.shape[1])
    ix1_r: int = round(0.66*target.shape[1])

    assert (target[ix0, ix1_l] == IGNORE_INDEX
        and target[ix0, ix1_r] != IGNORE_INDEX) or \
        (target[ix0, ix1_l] != IGNORE_INDEX
        and target[ix0, ix1_r] == IGNORE_INDEX)
