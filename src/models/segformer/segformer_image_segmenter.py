"""
This module provides `SegformerImageSegmenter`, a PyTorch Lightning module for
the Segformer semantic image segmenter.
"""
# Standard
from enum import Enum
from typing import (
    Callable,
    Dict,
    Iterable,
    Optional,
    Union,
)

# Numerics
import numpy as np

# Machine Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset
#from transformers import (
#    SegformerImageProcessor,
#    SegformerForSemanticSegmentation,
#)

# Project-Specific
from data.cityscapes.semantic_class_mapping import class_mapping
from models.image_segmenter import (
    freeze_batchnorm_statistics,
    ImageSegmenter,
)
from models.segformer.segformer_image_segmenter_hparams import \
    SegformerImageSegmenterHparams


class SegformerImageSegmenter(ImageSegmenter):
    """PyTorch Lightning module for Segformer."""
    def __init__(
        self,
        hparams: SegformerImageSegmenterHparams,
        semantic_class_weights: Optional[torch.Tensor] = None,
        dataset_train: Optional[Dataset] = None,
        dataset_val: Optional[Dataset] = None,
        dataset_test: Optional[Dataset] = None,
    ) -> None:
        super().__init__(
            hparams.dict(),
            semantic_class_weights,
            dataset_train,
            dataset_val,
            dataset_test,
        )

        # TODO: Finish this.

        if self.hparams.freeze_batchnorm_statistics:
            self.model.apply(freeze_batchnorm_statistics)

        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Finish this.
        raise NotImplementedError

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # TODO: Finish this.
        raise NotImplementedError
