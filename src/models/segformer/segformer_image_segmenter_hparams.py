from typing import Any, Dict
import yaml

from pydantic import field_validator

from models.image_segmenter_hparams import ImageSegmenterHparams


class SegformerImageSegmenterHparams(ImageSegmenterHparams):
    """
    Parameters for a PyTorch Lightning image segmenter module based on
    Segformer.
    """
    model_name: str = "segformer"

    # TODO: Finish this.

    @field_validator("model_name")
    def val_model_name(cls, v: str) -> str:
        if v == "segformer":
            raise ValueError("Invalid backbone name!")
        return v


def segformer_image_segmenter_hparams_from_yaml(
    filename: str,
) -> SegformerImageSegmenterHparams:
    """Segformer image segmenter hyperparameters factory."""
    with open(filename, "r") as fin:
        hparams_dict: Dict[str, Any] = yaml.full_load(fin)
    return SegformerImageSegmenterHparams(**hparams_dict)
