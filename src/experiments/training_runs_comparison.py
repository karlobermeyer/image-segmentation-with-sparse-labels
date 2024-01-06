"""
Tools for comparing multiple training runs.
"""

# Standard
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Numerics
import numpy as np

# Machine Learning
import pandas as pd

# Other 3rd-Party
import texttable  # https://pypi.org/project/texttable/

# Project-Specific
from core import (
    #SemanticClassId,
    #SemanticClassMapping,
    SparseLabelSimulatingDataset,
)
from core.visualization import (
    plot_grid_of_images,
    construct_segmentation_side_by_sides,
)
from experiments.training_run import TrainingRun


def plot_model_outputs_side_by_side(
    runs: List[TrainingRun],
    dataset: SparseLabelSimulatingDataset,
    example_ixs: Sequence[int],
    include_targets: bool = True,
    run_names: Optional[List[str]] = None,
) -> None:
    """
    Show side-by-side images, model outputs, and targets.

    We recommend that `runs` be sorted so that performance improves from left to
    right. That way there is a natural mapping between performance and physical
    proximity to the target.

    Args:
        runs: the runs to be compared.
        dataset: The dataset to take examples from.
        example_ixs: indices in `dataset` of the examples to use.
        include_targets: Whether to include targets (may not be available
            for test sets).
        run_names: optional strings to be used as run column headings.
    """
    if run_names is not None:
        assert len(runs) == len(run_names)

    image_lists: List[List[np.ndarray]] = []
    for ix in example_ixs:
        image_list: List[np.ndarray] = []
        datapair_unpreprocessed: Tuple[np.ndarray, np.ndarray] = \
            dataset.get_unpreprocessed(ix)
        image_unpreprocessed: np.ndarray = datapair_unpreprocessed[0]
        image_list.append(image_unpreprocessed)

        for run in runs:
            segmentation_map_int: np.ndarray = run.segmenter.segment(
                image_unpreprocessed,
            )
            segmentation_map_rgb: np.ndarray = \
                dataset.class_mapping.segmentation_map_rgb_from_int(
                    segmentation_map_int,
                )
            image_list.append(segmentation_map_rgb)

        if include_targets:
            target_unpreprocessed: np.ndarray = datapair_unpreprocessed[1]
            segmentation_map_rgb_target: np.ndarray = \
                dataset.class_mapping.segmentation_map_rgb_from_int(
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

    if run_names is not None:
        num_headings: int = len(image_lists[0])
        dx: float = 1.0/num_headings
        y_title: float = 1.15
        title_fontsize: float = 15.0
        axs[0, 0].text(
            0.5*dx, y_title, "Input",
            transform=axs[0, 0].transAxes, ha="center", va="center",
            fontsize=title_fontsize,
        )
        for ix, run_name in enumerate(run_names):
            axs[0, 0].text(
                0.5*dx + (1.0 + ix)*dx, y_title, run_name,
                transform=axs[0, 0].transAxes, ha="center", va="center",
                fontsize=title_fontsize,
            )
        if include_targets:
            axs[0, 0].text(
                1.0 - 0.5*dx, y_title, "Target",
                transform=axs[0, 0].transAxes, ha="center", va="center",
                fontsize=title_fontsize,
            )


def models_performance_comparison_df(
    runs: List[TrainingRun],
    dataset_eval_name: str,
    hparam_names: List[str],
    summary_metric_names: List[str],
    run_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Construct and return a pandas dataframe of relevant hyperparameters and
    summary metrics.

    Each row is a run. Columns are hyperparameters and summary metrics. Consider
    sorting `runs` so that rows appear in ascending or descending order with
    respect to the primary metric.

    Assumes evaluation results have already been generated for all runs.

    Args:
        runs: list of training runs.
        dataset_eval_name: name of evaluation dataset to use metric from.
        hparams_names: names of hyperparameters to include.
        summary_metric_names: names of summary metrics to include.
        run_names: optional strings to help identify runs, used as first column
            if provided.

    Returns:
        pandas dataframe
    """
    columns: List[str] = []
    if run_names is not None:
        columns.append("run_name")
    for hparam_name in hparam_names:
        columns.append(hparam_name)
    for metric_name in summary_metric_names:
        columns.append(metric_name)

    rows: List[List[Any]] = []
    for ix, run in enumerate(runs):
        row: List[Any] = []
        if run_names is not None:
            row.append(run_names[ix])
        for hparam_name in hparam_names:
            row.append(run.hparams[hparam_name])
        summary_metrics: Dict[str, float] = \
            run.load_summary_metrics(dataset_eval_name)
        for metric_name in summary_metric_names:
            row.append(summary_metrics[metric_name])
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def models_performance_comparison_texttable_str(
    runs: List[TrainingRun],
    dataset_eval_name: str,
    hparam_names: List[str],
    summary_metric_names: List[str],
    run_names: Optional[List[str]] = None,
) -> str:
    """
    Construct and return a texttable string of relevant hyperparameters and
    summary metrics.

    Each row is a run. Columns are hyperparameters and summary metrics. Consider
    sorting `runs` so that rows appear in ascending or descending order with
    respect to the primary metric.

    Assumes evaluation results have already been generated for all runs.

    Args:
        runs: list of training runs.
        dataset_eval_name: name of evaluation dataset to use metric from.
        hparams_names: names of hyperparameters to include.
        summary_metric_names: names of summary metrics to include.
        run_names: optional strings to help identify runs, used as first column
            if provided.

    Returns:
        texttable string
    """
    table = texttable.Texttable()

    header_row: List[str] = []
    cols_align: List[str] = []
    cols_valign: List[str] = []
    cols_dtype: List[str] = []
    if run_names is not None:
        header_row.append("run_name")
        cols_align.append("r")
        cols_valign.append("c")
        cols_dtype.append("a")
    for hparam_name in hparam_names:
        header_row.append(hparam_name)
        cols_align.append("c")
        cols_valign.append("c")
        dtype: type = type(runs[0].hparams[hparam_name])
        if dtype is str:
            cols_dtype.append("a")
        elif dtype is bool:
            cols_dtype.append("t")
        elif dtype is int:
            cols_dtype.append("i")
        elif dtype is float:
            cols_dtype.append("f")
        else:
            raise NotImplementedError
    for metric_name in summary_metric_names:
        header_row.append(metric_name)
        cols_align.append("c")
        cols_valign.append("c")
        cols_dtype.append("f")
    table.set_cols_align(cols_align)  # l, c, r
    table.set_cols_valign(cols_valign)  # t, m, b
    table.set_cols_dtype(cols_dtype)  # t, f, e, i, a

    rows: List[List[Any]] = [ header_row ]
    for ix, run in enumerate(runs):
        row: List[Any] = []
        if run_names is not None:
            row.append(run_names[ix])
        for hparam_name in hparam_names:
            row.append(run.hparams[hparam_name])
        summary_metrics: Dict[str, float] = \
            run.load_summary_metrics(dataset_eval_name)
        for metric_name in summary_metric_names:
            row.append(summary_metrics[metric_name])
        rows.append(row)
    table.add_rows(rows)

    return table.draw() + "\n"
