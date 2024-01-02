from typing import Any, Dict
import yaml

from pydantic import field_validator

from models.image_segmenter_hparams import ImageSegmenterHparams


class DeepLabV3ImageSegmenterHparams(ImageSegmenterHparams):
    """
    Parameters for a PyTorch Lightning image segmenter module based on
    DeepLabV3.
    """
    model_name: str = "deeplabv3"

    # With `deeplabv3` from `torchvision.models.segmentation`,
    # "mobilenet_v3_large" => `deeplabv3.deeplabv3_mobilenet_v3_large`,
    # "resnet50" => `deeplabv3.deeplabv3_resnet50`,
    # "resnet101" => `deeplabv3.deeplabv3_resnet101`.
    backbone_name: str = "resnet101"

    @field_validator("model_name")
    def val_model_name(cls, v: str) -> str:
        if v != "deeplabv3":
            raise ValueError("Invalid model name!")
        return v

    @field_validator("backbone_name")
    def val_backbone_name(cls, v: str) -> str:
        if v not in ("mobilenet_v3_large", "resnet50", "resnet101"):
            raise ValueError("Invalid backbone name!")
        return v


def deeplabv3_image_segmenter_hparams_from_yaml(
    filename: str,
) -> DeepLabV3ImageSegmenterHparams:
    """DeepLabV3 image segmenter hyperparameters factory."""
    with open(filename, "r") as fin:
        hparams_dict: Dict[str, Any] = yaml.full_load(fin)
    return DeepLabV3ImageSegmenterHparams(**hparams_dict)
