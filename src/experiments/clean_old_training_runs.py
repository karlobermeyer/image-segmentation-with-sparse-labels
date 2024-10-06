#!/usr/bin/env python3
"""
Delete old training run directories, retaining only the latest under each
scenario.

Within each scenario directory
`{PROJECT_ROOT}/runs/<model_name>/<scenario_name>/`,
there are multiple run subdirectories named according to their UTC datetime of
creation. This script deletes all but the latest of those subdirectories.
"""

import os
import shutil

from experiments.training_runs_access import get_latest_run_dirname


PROJECT_ROOT: str = os.environ.get("PROJECT_ROOT")
assert PROJECT_ROOT is not None, \
    "PROJECT_ROOT not found! Did you run `setenv.sh`?"


def clean_old_training_runs() -> None:
    """
    Delete old run directories, retaining only the latest under each scenario.
    """
    runs_dirname: str = os.path.join(PROJECT_ROOT, "runs")

    for model_dirname_leaf in os.listdir(runs_dirname):
        model_dirname: str = os.path.join(runs_dirname, model_dirname_leaf)
        if not os.path.isdir(model_dirname):
            continue

        for scenario_dirname_leaf in os.listdir(model_dirname):
            scenario_dirname: str = os.path.join(
                model_dirname,
                scenario_dirname_leaf,
            )
            if not os.path.isdir(scenario_dirname):
                continue

            latest_run_dirname: str = \
                get_latest_run_dirname(scenario_dirname)
            for run_dirname_leaf in os.listdir(scenario_dirname):
                run_dirname: str = os.path.join(
                    scenario_dirname,
                    run_dirname_leaf,
                )
                if not os.path.isdir(run_dirname):
                    continue
                if run_dirname != latest_run_dirname:
                    shutil.rmtree(run_dirname)


if __name__ == "__main__":
    clean_old_training_runs()
