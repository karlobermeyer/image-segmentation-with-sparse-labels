"""
Helpers for accessing run directories and their contents.
"""
# Standard
from datetime import datetime
import os
from typing import Any, Dict, List, Optional, Tuple

# Machine Learning
import pandas as pd
import torch

# Project-Specific
from core import utc_datetime_from_str_prefix


def model_name_from_run_dirname(run_dirname: str) -> str:
    return run_dirname.rstrip("/").split("/")[-3]


def get_latest_run_dirname(scenario_dirname: str) -> str:
    datetime_dirname_pairs: List[Tuple[datetime, str]] = []
    for run_dirname_leaf in os.listdir(scenario_dirname):
        run_dirname: str = os.path.join(
            scenario_dirname,
            run_dirname_leaf,
        )
        if not os.path.isdir(run_dirname):
            continue
        datetime_dirname_pairs.append((
            utc_datetime_from_str_prefix(run_dirname_leaf),
            run_dirname,
        ))
    datetime_dirname_pairs.sort()
    return datetime_dirname_pairs[-1][1]


def get_run_checkpoint_filename(run_dirname: str) -> str:
    checkpoint_filename_leaf: Optional[str] = None
    for filename in os.listdir(run_dirname):
        if filename.endswith(".ckpt"):
            checkpoint_filename_leaf = filename
            break
    assert checkpoint_filename_leaf is not None, "Checkpoint file missing!"
    return os.path.join(
        run_dirname,
        checkpoint_filename_leaf,
    )


def load_run_checkpoint(run_dirname: str) -> Dict[str, Any]:
    checkpoint_filename: str = get_run_checkpoint_filename(run_dirname)
    return torch.load(checkpoint_filename)


def load_run_training_time_series_metrics(run_dirname: str) -> pd.DataFrame:
    training_metrics: pd.DataFrame = pd.read_csv(
        os.path.join(run_dirname, "training_time_series", "metrics_raw.csv")
    )
    training_metrics: pd.DataFrame = \
        training_metrics.groupby(["epoch", "step"], as_index=False).first()
    return training_metrics
