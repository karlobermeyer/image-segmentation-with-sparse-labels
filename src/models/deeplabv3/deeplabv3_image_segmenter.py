"""
This module provides `DeepLabV3ImageSegmenter`, a PyTorch Lightning module for
the DeepLabV3 semantic image segmenter.
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

# Machine Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.models.segmentation import deeplabv3

# Project-Specific
from data.cityscapes.semantic_class_mapping import class_mapping
from models.image_segmenter import (
    freeze_batchnorm_statistics,
    ImageSegmenter,
)
from models.deeplabv3.deeplabv3_image_segmenter_hparams import \
    DeepLabV3ImageSegmenterHparams


class DeepLabV3ImageSegmenter(ImageSegmenter):
    """PyTorch Lightning module for DeepLabV3."""
    def __init__(
        self,
        hparams: DeepLabV3ImageSegmenterHparams,
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

        if self.hparams.backbone_name == "mobilenet_v3_large":
            model_from: Callable = deeplabv3.deeplabv3_mobilenet_v3_large
            weights: Enum = \
                deeplabv3.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            self.model: nn.Module = model_from(weights=weights)
            self.model.aux_classifier = None  # Remove auxiliary classifier.
        elif self.hparams.backbone_name == "resnet50":
            model_from: Callable = deeplabv3.deeplabv3_resnet50
            weights: Enum = deeplabv3.DeepLabV3_ResNet50_Weights.DEFAULT
            self.model: nn.Module = model_from(weights=weights)
            self.model.aux_classifier = None  # Remove auxiliary classifier.
        elif self.hparams.backbone_name == "resnet101":
            model_from: Callable = deeplabv3.deeplabv3_resnet101
            weights: Enum = deeplabv3.DeepLabV3_ResNet101_Weights.DEFAULT
            self.model: nn.Module = model_from(weights=weights)
            self.model.aux_classifier = None  # Remove auxiliary classifier.
        else:
            raise NotImplementedError

        # Replace model's final classification layer. By default, the `Conv2d`
        # parameters are initialized using the "Kaiming uniform initializer".
        self.model.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=class_mapping.num_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

        if self.hparams.layers_to_train == "all":
            for param in self.model.parameters():
                param.requires_grad = True  # Unfreeze all.
        elif self.hparams.layers_to_train == "head":
            # If the datasets were more different, we might want to replace
            # the model's entire head like this.
            # self.model.classifier = deeplabv3.DeepLabHead(
            #     in_channels=960,  # How to determine automatically?
            #     num_classes=class_mapping.num_classes,
            # )
            for param in self.model.parameters():
                param.requires_grad = False  # Freeze all.
            for param in self.model.classifier.parameters():
                param.requires_grad = True  # Unfreeze classifier.
        elif self.hparams.layers_to_train == "final":
            for param in self.model.parameters():
                param.requires_grad = False  # Freeze all.
            for param in self.model.classifier[4].parameters():
                param.requires_grad = True   # Unfreeze last layer.
        else:
            raise NotImplementedError

        if self.hparams.freeze_batchnorm_statistics:
            self.model.apply(freeze_batchnorm_statistics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.hparams.layers_to_train == "final" or \
                self.hparams.learning_rate_final_layer \
                == self.hparams.learning_rate_nonfinal_layers:
            optimizer: torch.optim.Optimizer = super().configure_optimizers()
        else:
            params_final_layer: Dict[
                str,
                Union[float, Iterable[nn.Parameter]],
            ] = {
                "lr": self.learning_rate_final_layer,  # float
                "params": self.model.classifier[4].parameters(),
            }
            params_nonfinal_layers: Dict[
                str,
                Union[float, Iterable[nn.Parameter]],
            ] = {
                "lr": self.learning_rate_nonfinal_layers,
                "params": [
                    param for name, param in self.model.named_parameters()
                    if param.requires_grad and ("classifier.4" not in name)
                ],
            }
            optimizer: torch.optim.Optimizer = torch.optim.AdamW([
                params_final_layer,
                params_nonfinal_layers,
            ])
        return optimizer
