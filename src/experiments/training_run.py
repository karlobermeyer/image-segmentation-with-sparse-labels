# Standard
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import yaml

# Numerics
import numpy as np

# Machine Learning
import torch
import pandas as pd
import pytorch_lightning as pl

# Project-Specific
from core import (
    #SemanticClassId,
    SemanticClassMapping,
    SparseLabelSimulatingDataset,
)
from core.visualization import (
    plot_grid_of_images,
    construct_segmentation_side_by_sides,
)
from data.cityscapes import class_mapping
from models import (
    ImageSegmenter,
    DeepLabV3ImageSegmenterHparams,
    DeepLabV3ImageSegmenter,
    SegformerImageSegmenterHparams,
    SegformerImageSegmenter,
)
from experiments.training_runs_access import (
    model_name_from_run_dirname,
    load_run_checkpoint,
    load_run_training_time_series_metrics,
)
from experiments.training_accessories import (
    plot_losses_vs_epoch,
)


class TrainingRun:
    """
    Container and interface for training run results.

    A *training run* is induced by a model name, scenario name, UTC creation
    time, and random number seed.
    """
    def __init__(self, run_dirname: str) -> None:
        assert os.path.exists(run_dirname), "Missing run directory!"
        self._run_dirname: str = run_dirname

        model_name: str = model_name_from_run_dirname(run_dirname)
        assert model_name in ("deeplabv3", "segformer"), "Unknown model!"

        self._class_mapping: SemanticClassMapping = class_mapping

        self._training_time_series_metrics: pd.DataFrame = \
            load_run_training_time_series_metrics(run_dirname)
        checkpoint: Dict[str, Any] = load_run_checkpoint(run_dirname)
        # Checkpoint keys available:
        # "epoch"
        # "global_step"
        # "pytorch-lightning_version"
        # "state_dict"
        # "loops"
        # "callbacks"
        # "optimizer_states"
        # "lr_schedulers"
        # "MixedPrecisionPlugin"
        # "hparams_name"  <-- Ignore this.
        # "hyper_parameters"

        self._accepted_epoch_id: int = checkpoint["epoch"]  # IDs start at 0.
        self._num_epochs_accepted: int = checkpoint["epoch"] + 1
        self._num_steps_accepted: int = checkpoint["global_step"]
        self._num_epochs_trained: int = \
            self._training_time_series_metrics["epoch"].max() + 1
        hparams_dict: Dict[str, Any] = checkpoint["hyper_parameters"]
        assert model_name == hparams_dict["model_name"], \
            "Model name from run directory does not match hyperparameters!"
        if "criterion.weight" in checkpoint["state_dict"]:
            semantic_class_weights: Optional[torch.Tensor] = \
                checkpoint["state_dict"]["criterion.weight"]
        else:
            semantic_class_weights: Optional[torch.Tensor] = None
        if model_name:
            self._segmenter: ImageSegmenter = DeepLabV3ImageSegmenter(
                hparams=DeepLabV3ImageSegmenterHparams(**hparams_dict),
                semantic_class_weights=semantic_class_weights,
                dataset_train=None,
                dataset_val=None,
                dataset_test=None,
            )
        elif model_name == "segformer":
            self._segmenter: ImageSegmenter = SegformerImageSegmenter(
                hparams=SegformerImageSegmenterHparams(**hparams_dict),
                semantic_class_weights=semantic_class_weights,
                dataset_train=None,
                dataset_val=None,
                dataset_test=None,
            )
        else:
            raise NotImplementedError
        self._segmenter.load_state_dict(checkpoint["state_dict"])

        training_time_filename: str = os.path.join(
            run_dirname,
            "training_time_hrs.txt",
        )
        with open(training_time_filename, "r") as fin:
            self._training_time_hrs: float = float(fin.read())

    @property
    def run_dirname(self) -> str:
        return self._run_dirname

    @property
    def class_mapping(self) -> SemanticClassMapping:
        return self._class_mapping

    @property
    def hparams(self) -> pl.utilities.parsing.AttributeDict:
        return self.segmenter.hparams

    @property
    def model_name(self) -> str:
        return self.hparams.model_name

    @property
    def scenario_name(self) -> str:
        return self.hparams.scenario_name

    @property
    def accepted_epoch_id(self) -> int:
        """ID of the epoch the accepted checkpoint was taken from."""
        return self._accepted_epoch_id

    @property
    def num_epochs_accepted(self) -> int:
        """Number of epochs the accepted checkpoint was trained for."""
        return self._num_epochs_accepted

    @property
    def num_steps_accepted(self) -> int:
        """
        Number of training steps (minibatches) the accepted checkpoint was
        trained for.
        """
        return self._num_steps_accepted

    @property
    def num_epochs_trained(self) -> int:
        """
        Total number of epochs trained (including past the accepted early
        stopping point).
        """
        return self._num_epochs_trained

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def training_time_hrs(self) -> float:
        return self._training_time_hrs

    @property
    def training_time_series_metrics(self) -> pd.DataFrame:
        return self._training_time_series_metrics

    @property
    def segmenter(self) -> ImageSegmenter:
        return self._segmenter

    def plot_losses_vs_epoch(
        self,
        yscale: str = "linear",
        include_title: bool = True,
    ) -> None:
        """
        Plot training and validation losses (vertical) as a function of training
        epoch (horizontal).

        If there is no validation loss, only training loss is plotted.

        Args:
            yscale: What scale to use on vertical axis ("linear" or "log").
            include_title: whether to include a plot title.
        """
        plot_losses_vs_epoch(
            metrics_df=self.training_time_series_metrics,
            accepted_epoch_id=self.accepted_epoch_id,
            yscale=yscale,
            include_title=include_title,
        )

    def plot_examples(
        self,
        dataset: SparseLabelSimulatingDataset,
        example_ixs: Sequence[int],
        include_targets: bool = True,
        include_column_headings: bool = True,
    ) -> None:
        """
        Show side-by-side images, model output, and targets.

        Args:
            dataset: The dataset to take examples from.
            example_ixs: indices in `dataset` of the examples to use.
            include_targets: Whether to include targets (may not be available
                for test sets).
            include_column_headings: Whether to include column headings on plot.
        """
        image_lists: List[List[np.ndarray]] = []
        for ix in example_ixs:
            image_list: List[np.ndarray] = []
            datapair_unpreprocessed: Tuple[np.ndarray, np.ndarray] = \
                dataset.get_unpreprocessed(ix)
            image_unpreprocessed: np.ndarray = datapair_unpreprocessed[0]
            image_list.append(image_unpreprocessed)

            segmentation_map_int: np.ndarray = self.segmenter.segment(
                image_unpreprocessed,
            )
            segmentation_map_rgb: np.ndarray = \
                self.class_mapping.segmentation_map_rgb_from_int(
                    segmentation_map_int,
                )
            image_list.append(segmentation_map_rgb)

            if include_targets:
                target_unpreprocessed: np.ndarray = datapair_unpreprocessed[1]
                segmentation_map_rgb_target: np.ndarray = \
                    self.class_mapping.segmentation_map_rgb_from_int(
                        target_unpreprocessed,
                    )
                image_list.append(segmentation_map_rgb_target)

            image_lists.append(image_list)

        images_for_plotting: List[np.ndarray] = \
            construct_segmentation_side_by_sides(image_lists)

        num_rows = len(example_ixs)
        num_cols = 1
        fig, axs = plot_grid_of_images(
            images_for_plotting,
            num_cols=num_cols,
            num_rows=num_rows,
            vertical_spacing=0.1,
            horizontal_spacing=0.03,
            show_image_boundaries=True,
        )

        if include_column_headings:
            num_headings: int = len(image_lists[0])
            dx: float = 1.0/num_headings
            title_fontsize: float = 8.0
            axs[0, 0].text(
                0.5*dx, 1.10, "Input",
                transform=axs[0, 0].transAxes, ha="center", va="center",
                fontsize=title_fontsize,
            )
            # Kept `ix_run` factored out here so that it is easier to see the
            # correctness as a special case of plotting multiple runs together.
            ix_run: int = 0
            axs[0, 0].text(
                0.5*dx + (1.0 + ix_run)*dx, 1.10, "Output",
                transform=axs[0, 0].transAxes, ha="center", va="center",
                fontsize=title_fontsize,
            )
            if include_targets:
                axs[0, 0].text(
                    1.0 - 0.5*dx, 1.10, "Target",
                    transform=axs[0, 0].transAxes, ha="center", va="center",
                    fontsize=title_fontsize,
                )

    def load_summary_metrics(
        self,
        dataset_eval_name: str,
    ) -> Dict[str, float]:
        filename: str = os.path.join(
            self.run_dirname,
            "final_model_evaluation",
            dataset_eval_name,
            "summary.yaml",
        )
        assert os.path.exists(filename), \
            "Summary metrics file does not exist for that evaluation dataset!"
        with open(filename, "r") as fin:
            summary_metrics: Dict[str, float] = yaml.safe_load(fin)
        for value in summary_metrics.values():
            assert isinstance(value, float)
        return summary_metrics
