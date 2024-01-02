from typing import Any, Dict, Optional, Tuple
import yaml

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)


class ImageSegmenterHparams(BaseModel):
    """Parameters for a PyTorch Lightning image segmenter module."""

    # Whether to do a quick smoke test with just a small number of epochs and
    # minibatches.
    smoke_test: bool = False

    # Name of image segmentation model, e.g., "deeplabv3" or "segformer".
    model_name: str = "deeplabv3"

    # Name of scenario. A *scenario* is a unique set of the hyperparameters. A
    # *training run* is uniquely specified by a scenario together with UTC
    # creation datetime. Recommended scenario name format:
    # "TRAINING_DATASET_NAME--ROOT_SCENARIO_NAME", e.g.,
    # "subcityscapes--000".
    scenario_name: str = "scenario_name"

    # Master random number seed. Used in `pl.seed_everything(seed)` at the
    # beginning of each run.
    seed: int = 13

    # Name of dataset to train on, e.g., "subcityscapes_train",
    # "subcityscapes_trainval", "cityscapes_train", or "cityscapes_trainval".
    dataset_train_name: str = "subcityscapes_train"

    # Name of dataset to use for validation, e.g., "subcityscapes_val" or
    # "cityscapes_val".
    dataset_val_name: Optional[str] = "subcityscapes_val"

    # Portion of the available training datatuples to use in constructing the
    # training dataset. For example, 1.0 => use all datatuples, 0.5 => use half
    # of available datatuples (chosen randomly). This is for simulating
    # having less data than we actually do.
    len_dataset_train_scale_factor: float = 1.0

    # Portion of the available validation datatuples to use in constructing the
    # validation dataset. For example, 1.0 => use all datatuples, 0.5 => use
    # half of available datatuples (chosen randomly). This is for simulating
    # having less data than we actually do.
    len_dataset_val_scale_factor: float = 1.0

    # What portion of each image to label.
    label_density: float = 1.0

    # Sentinel value used to mark target pixels as unobservable.
    ignore_index: int = 255

    # Height and width of images after preprocessing, i.e., as they are to be
    # fed to the underlying `torch.nn.Module`.
    input_height: int = 320  # px, Default: 320
    input_width: int = 640  # px, Default: 640

    # Mean and standard deviation used for normalizing input images during
    # preprocessing.
    mean_for_input_normalization: Tuple[float, float, float] = \
        (0.485, 0.456, 0.406)
    std_for_input_normalization: Tuple[float, float, float] = \
        (0.229, 0.224, 0.225)

    # Training Scope:
    # "all" => all layers,
    # "final" => final layer only, or
    # "head" => classifier head layers only.
    layers_to_train: str = "all"

    # Number of image-target pairs per minibatch.
    minibatch_size: int = 32  # Usu. 16, 32, or 64.

    # Learning rate for final layer only.
    learning_rate_final_layer: float = 0.001

    # Learning rate for trainable non-final layers. Use `0.0` when
    # `layers_to_train == "final"`.
    learning_rate_nonfinal_layers: float = 0.0001

    # Whether to freeze batchnorm statistics (but not batchnorm's affine
    # transformation parameters).
    freeze_batchnorm_statistics: bool = False

    # Whether to weight classes in the loss function. True => classes are
    # automatically weighted based on their frequency in training and validation
    # sets combined.
    weight_classes: bool = False

    # Upper bound enforced on the ratio of the largest semantic class frequency
    # to the smallest frequency. This measure of smoothness was inspired by the
    # condition number of a normal matrix, which is the ratio of largest to
    # smallest eigenvalue magnitudes.
    class_frequency_max_to_min_ratio_ubnd: Optional[float] = None

    # Label smoothing, most commonly 0.1, usu. in [0.0, 0.2].
    label_smoothing: float = 0.0

    # Whether to use early stopping. This requires a validation set.
    stop_early: bool = True

    # Maximum number of training epochs. Because epoch IDs start at 0, the
    # maximum epoch ID will be `num_epochs_ubnd - 1`.
    num_epochs_ubnd: int = 500

    # <end of field declarations>

    model_config: ConfigDict = ConfigDict(
        protected_namespaces = (),
    )

    @field_validator("model_name")
    def val_model_name(cls, v: str) -> str:
        if v not in ("deeplabv3", "segformer"):
            raise ValueError("Invalid model name!")
        return v

    @field_validator("dataset_train_name")
    def val_dataset_train_name(cls, v: str) -> str:
        if v not in (
            "subcityscapes_train",
            "subcityscapes_trainval",
            "cityscapes_train",
            "cityscapes_trainval",
        ):
            raise ValueError("Invalid training dataset!")
        return v

    @field_validator("dataset_val_name")
    def val_dataset_val_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in (
            "subcityscapes_val",
            "cityscapes_val",
        ):
            raise ValueError("Invalid validation dataset!")
        return v

    @field_validator("len_dataset_train_scale_factor")
    def val_len_dataset_train_scale_factor(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("Invalid dataset scale factor!")
        return v

    @field_validator("len_dataset_val_scale_factor")
    def val_len_dataset_val_scale_factor(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("Invalid dataset scale factor!")
        return v

    @field_validator("label_density")
    def val_label_density(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("Invalid label density!")
        return v

    @field_validator("input_height")
    def val_input_height(cls, v: int) -> int:
        if not 0 < v:
            raise ValueError("Invalid input height!")
        return v

    @field_validator("input_width")
    def val_input_width(cls, v: int) -> int:
        if not 0 < v:
            raise ValueError("Invalid input width!")
        return v

    @field_validator("layers_to_train")
    def val_layers_to_train(cls, v: str) -> str:
        if v not in ("all", "final", "head"):
            raise ValueError("Invalid specification of layers to train!")
        return v

    @field_validator("minibatch_size")
    def val_minibatch_size(cls, v: int) -> int:
        if not 0 < v:
            raise ValueError("Minibatch size must be greater than zero!")
        return v

    @field_validator("learning_rate_final_layer")
    def val_learning_rate_final_layer(cls, v: float) -> float:
        if not 0.0 < v:
            raise ValueError(
                "Learning rate of final layer must be positive!"
            )
        return v

    @field_validator("learning_rate_nonfinal_layers")
    def val_learning_rate_nonfinal_layers(cls, v: float) -> float:
        if not 0.0 <= v:
            raise ValueError(
                "Learning rate of non-final layers must be non-negative!",
            )
        return v

    @field_validator("label_smoothing")
    def val_label_smoothing(cls, v: float) -> float:
        if not 0.0 <= v:
            raise ValueError("Label smoothing must be non-negative!")
        return v

    @field_validator("class_frequency_max_to_min_ratio_ubnd")
    def val_class_frequency_max_to_min_ratio_ubnd(
        cls,
        v: Optional[float],
    ) -> Optional[float]:
        if v is not None and not 1.0 <= v:
            raise ValueError(
                "Class frequency max to min ratio must be "
                "greater or equal to 1.0!"
            )
        return v

    @field_validator("num_epochs_ubnd")
    def val_num_epochs_ubnd(cls, v: int) -> int:
        if not 0 < v:
            raise ValueError("Number of epochs must be positive!")
        return v

    @model_validator(mode="after")
    def val_early_stopping_requires_val_data(self)-> Any:
        if self.stop_early and self.dataset_val_name is None:
            raise ValueError("Early stopping requires validation data!")
        return self

    @model_validator(mode="after")
    def val_train_final_only_requires_zero_nonfinal_lr(self)-> Any:
        if self.layers_to_train == "final" \
                and self.learning_rate_nonfinal_layers != 0.0:
            raise ValueError(
                "Learning rate of non-final layers must be zero "
                "when only training final layer!"
            )
        return self

    def to_yaml(self, filename: str) -> None:
        with open(filename, "w") as fout:
            yaml.dump(self.model_dump(), fout)


def image_segmenter_hparams_from_yaml(
    filename: str,
) -> ImageSegmenterHparams:
    """Image segmenter hyperparameters factory."""
    with open(filename, "r") as fin:
        hparams_dict: Dict[str, Any] = yaml.full_load(fin)
    return ImageSegmenterHparams(**hparams_dict)
