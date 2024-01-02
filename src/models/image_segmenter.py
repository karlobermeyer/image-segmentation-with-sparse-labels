"""
This module provides `ImageSegmenter`, a base PyTorch Lightning module for
semantic image segmenters.
"""
# Standard
from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Iterable, Optional

# Numerics
import numpy as np

# Image Processing
import cv2

# Machine Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import albumentations as albm

# Project-Specific
from core import (
    #SemanticClassId,
    SemanticClassMapping,
)
from data.cityscapes.semantic_class_mapping import class_mapping
from data.cityscapes import preprocesses_from


PIN_MEMORY: bool = True


def freeze_batchnorm_statistics(module: nn.Module) -> None:
    """
    Freeze the batchnorm statistics.

    This does not freeze the affine transformation parameters of batchnorm
    layers. If you want to do that, do it separately.
    """
    if isinstance(module, nn.BatchNorm2d):
        module.eval()


class ImageSegmenter(pl.LightningModule, ABC):
    """Base PyTorch Lightning module for semantic image segmenters."""
    def __init__(
        self,
        hparams: Dict[str, Any],
        semantic_class_weights: Optional[torch.Tensor] = None,
        dataset_train: Optional[Dataset] = None,
        dataset_val: Optional[Dataset] = None,
        dataset_test: Optional[Dataset] = None,
    ) -> None:
        super().__init__()
        self.class_mapping: SemanticClassMapping = class_mapping

        # This saves the hyperparameters to `self.hparams`.
        self.save_hyperparameters(hparams)

        #self.num_workers = 0  # Slow, but sometimes useful for debugging.
        #self.num_workers = max(1, os.cpu_count()//2)
        self.num_workers = max(1, 3*os.cpu_count()//4)
        #self.num_workers = max(1, os.cpu_count() - 1)

        # `CrossEntropyLoss` does not require one-hot encoding of targets. It
        # accepts integer labels directly.
        # https://pytorch.org/docs/stable/generated/nn.CrossEntropyLoss.html
        self.criterion = nn.CrossEntropyLoss(
            weight=semantic_class_weights,
            ignore_index=self.hparams.ignore_index,
            reduction="mean",
            label_smoothing=self.hparams.label_smoothing,
        )

        self.preprocesses: Dict[str, albm.Compose] = preprocesses_from(
            input_height=self.hparams.input_height,
            input_width=self.hparams.input_width,
            mean_for_input_normalization= \
                self.hparams.mean_for_input_normalization,
            std_for_input_normalization= \
                self.hparams.std_for_input_normalization,
            do_shift_scale_rotate=True,
            ignore_index=self.hparams.ignore_index,
        )
        self.dataset_train: Optional[Dataset] = dataset_train
        self.dataset_val: Optional[Dataset] = dataset_val
        self.dataset_test: Optional[Dataset] = dataset_test

    @property
    def num_classes(self) -> int:
        return self.class_mapping.num_classes

    @property
    def learning_rate_final_layer(self) -> float:
        return self.hparams.learning_rate_final_layer

    @property
    def learning_rate_nonfinal_layers(self) -> float:
        return self.hparams.learning_rate_nonfinal_layers

    @property
    def batch_size(self) -> int:
        return self.hparams.minibatch_size

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Redefine this in any subclass that will not always use equal learning
        rates for all trainable layers.
        """
        assert self.hparams.layers_to_train == "final" or \
            self.hparams.learning_rate_final_layer \
            == self.hparams.learning_rate_nonfinal_layers

        trainable_params: Iterable[nn.Parameter] = \
            [ param for param in self.model.parameters()
            if param.requires_grad ]

        # `AdamW` includes momentum and weight decay by default.
        # https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            params=trainable_params,
            lr=self.learning_rate_final_layer,
        )

        return optimizer

    def train_dataloader(self) -> DataLoader:
        assert self.dataset_train is not None
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=PIN_MEMORY,
            drop_last=False,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.dataset_val is None:
            return None
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=PIN_MEMORY,
            drop_last=False,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.dataset_test is None:
            return None
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=PIN_MEMORY,
            drop_last=False,
        )

    def _batch_loss(self, batch: torch.Tensor) -> torch.Tensor:
        images: torch.Tensor = batch[0]
        segmentation_map_int_targets: torch.Tensor = batch[1]
        segmentation_map_logits: torch.Tensor = self(images)
        loss: torch.Tensor = self.criterion(
            segmentation_map_logits,
            segmentation_map_int_targets.long(),
        )
        return loss

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        loss: torch.Tensor = self._batch_loss(batch)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss}

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        loss: torch.Tensor = self._batch_loss(batch)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return {"val_loss": loss}

    def test_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        loss: torch.Tensor = self._batch_loss(batch)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return {"test_loss": loss}

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the segmentation map of a single raw RGB image.

        Args:
            image: Raw RGB image.

        Returns:
            segmentation_map_int: array of semantic class IDs with the same
                height and width as the raw input image.
        """
        device: str = \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        height_raw: int = image.shape[0]
        width_raw: int = image.shape[1]
        image_tensor: torch.Tensor = \
            self.preprocesses["infer"](image=image)["image"].to(device)
        minibatch: torch.Tensor = image_tensor.unsqueeze(0)
        self.to(device)
        self.eval()
        with torch.inference_mode():
            segmentation_map_logits: torch.Tensor = self.forward(minibatch)[0]
        segmentation_map_int: np.ndarray = \
            self.class_mapping.segmentation_map_int_from_logits(
                segmentation_map_logits,
            )
        segmentation_map_int: np.ndarray = cv2.resize(
            segmentation_map_int,
            (width_raw, height_raw),
            interpolation=cv2.INTER_NEAREST,
        )
        return segmentation_map_int
