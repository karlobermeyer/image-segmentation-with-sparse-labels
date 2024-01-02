# Standard
import os
import sys
from typing import Any, Dict

# Visualization
import matplotlib.pyplot as plt

# Machine Learning
import pandas as pd
import pytorch_lightning as pl

# Project-Specific
from experiments.training_runs_access import (
    load_run_checkpoint,
    load_run_training_time_series_metrics,
)


class StdStreamLogger:
    """
    For CC'ing stdout and/or stderr to log file(s).

    Usage:
    ```
    sys.stdout = StdStreamLogger(".../stdout.log")
    sys.stderr = StdStreamLogger(".../stderr.log")
    <do stuff>
    sys.stdout.log.close()
    sys.stderr.log.close()
    sys.stdout = sys.stdout.terminal
    sys.stderr = sys.stderr.terminal
    ```

    Args:
      filename: full path of file to write to.
      mode: "w" => overwrite, "a" => append.
    """
    def __init__(self, filename: str, mode: str = "w") -> None:
        self.terminal = sys.stdout
        self.log = open(filename, mode, buffering=1)

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def plot_losses_vs_epoch(
    metrics_df: pd.DataFrame,
    accepted_epoch_id: int,
    yscale: str = "linear",
    include_title: bool = True,
) -> None:
    """
    Plot training and validation losses (vertical) as a function of training
    epoch (horizontal).

    If there is no validation loss, only training loss is plotted.

    Args:
        metrics_df: time series metrics dataframe.
        accepted_epoch_id: ID of epoch accepted (1st epoch ID is 0).
        yscale: What scale to use on vertical axis ("linear" or "log").
        include_title: whether to include a plot title.
    """
    assert yscale in ("linear", "log")
    plt.plot(
        metrics_df["epoch"],
        metrics_df["train_loss"],
        label="Training Loss",
        color="tab:blue",
        marker="o",
        zorder=10,
    )
    if "val_loss" in metrics_df:
        plt.plot(
            metrics_df["epoch"],
            metrics_df["val_loss"],
            label="Validation Loss",
            color="tab:orange",
            marker="o",
            zorder=5,
        )

    if yscale == "log":
        ax = plt.gca()
        ax.set_yscale("log")

    x_min: float = min(
        metrics_df["epoch"].min(),
        metrics_df["epoch"].min(),
    )
    x_max: float = max(
        metrics_df["epoch"].max(),
        metrics_df["epoch"].max(),
    )
    if "val_loss" in metrics_df:
        y_min: float = min(
            metrics_df["val_loss"].min(),
            metrics_df["train_loss"].min(),
        )
        y_max: float = max(
            metrics_df["val_loss"].max(),
            metrics_df["train_loss"].max(),
        )
    else:
        y_min: float = metrics_df["train_loss"].min()
        y_max: float = metrics_df["train_loss"].max()
    dx: float = x_max - x_min
    dy: float = y_max - y_min
    margin_factor: float = 0.08
    x_plot_min: float = x_min - margin_factor*dx
    x_plot_max: float = x_max + margin_factor*dx
    y_plot_min: float = y_min - margin_factor*dy
    y_plot_max: float = y_max + margin_factor*dy

    if "val_loss" in metrics_df:
        plt.plot(
            [accepted_epoch_id, accepted_epoch_id],
            [0.0, metrics_df["val_loss"][accepted_epoch_id]],
            label="Accepted Checkpoint",
            color="green",
            marker="",
            linestyle="--",
            linewidth=3.0,
            zorder=0,
        )

    if include_title:
        if "val_loss" in metrics_df:
            plt.title("Losses vs Epoch ID", fontsize=16, fontweight="bold")
        else:
            plt.title(
                "Training Loss vs Epoch ID",
                fontsize=16,
                fontweight="bold",
            )
    if "val_loss" in metrics_df:
        plt.legend()

    plt.xlabel("Epoch ID", fontsize=14, fontweight="bold")
    plt.ylabel("Loss", fontsize=14, fontweight="bold")
    #ax = plt.gca()
    #ax.margins(0.1)
    plt.xlim((x_plot_min, x_plot_max))
    plt.ylim((y_plot_min, y_plot_max))
    plt.grid(True, linestyle=":")


class TrainingRunPostprocessing(pl.Callback):
    def __init__(self, run_dirname: str) -> None:
        self.run_dirname = run_dirname

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        print("\nRun postprocessing:")

        # Fix checkpoint filename.
        for filename in os.listdir(self.run_dirname):
            if filename.endswith(".ckpt") and "=" in filename:
                print("  Adjusting checkpoint filename.")
                # Remove "=" symbol from checkpoint file.
                os.rename(
                    os.path.join(self.run_dirname, filename),
                    os.path.join(self.run_dirname, filename.replace("=", "")),
                )
            if filename == "metrics.csv":
                print("  Moving training time series metrics file.")
                 # Rename metrics file saved by Lightning CSVLogger.
                os.rename(
                     os.path.join(self.run_dirname, filename),
                     os.path.join(
                        self.run_dirname,
                        "training_time_series/",
                        "metrics_raw.csv",
                    ),
                )

        print("  Saving alternative formats of training time series metrics.")
        checkpoint: Dict[str, Any] = load_run_checkpoint(self.run_dirname)
        subdirname: str = \
            os.path.join(self.run_dirname, "training_time_series/")
        filename: str = os.path.join(subdirname, "metrics_raw.csv")
        metrics_df: pd.DataFrame = \
            load_run_training_time_series_metrics(self.run_dirname)
        accepted_epoch_id: int = checkpoint["epoch"]  # IDs start at 0.
        #num_epochs_accepted: int = checkpoint["epoch"] + 1
        #num_steps_accepted: int = checkpoint["global_step"]
        #num_epochs_trained: int = metrics_df["epoch"].max() + 1
        #
        # Generate cooked metrics files.
        metrics_df.to_csv(
            os.path.join(subdirname, "metrics.csv"),
            index=False,
        )
        # This is much more readable than `metrics_df.to_csv(...)`.
        with open(os.path.join(subdirname, "metrics.txt"), "w") as fout:
            fout.write(str(metrics_df) + "\n")

        print("  Generating plots of losses vs epoch, linear vertical scale.")
        plt.figure(figsize=(6, 6))
        plot_losses_vs_epoch(
            metrics_df=metrics_df,
            accepted_epoch_id=accepted_epoch_id,
            yscale="linear",
            include_title=False,
        )
        plt.savefig(
            os.path.join(subdirname, "loss_vs_epoch.png"),
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close()
        plt.figure(figsize=(6, 6))
        plot_losses_vs_epoch(
            metrics_df=metrics_df,
            accepted_epoch_id=accepted_epoch_id,
            yscale="linear",
            include_title=True,
        )
        plt.savefig(
            os.path.join(subdirname, "loss_vs_epoch--with_title.png"),
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close()

        print("  Generating plots of losses vs epoch, log vertical scale.")
        plt.figure(figsize=(6, 6))
        plot_losses_vs_epoch(
            metrics_df=metrics_df,
            accepted_epoch_id=accepted_epoch_id,
            yscale="log",
            include_title=False,
        )
        plt.savefig(
            os.path.join(subdirname, "log_loss_vs_epoch.png"),
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close()
        plt.figure(figsize=(6, 6))
        plot_losses_vs_epoch(
            metrics_df=metrics_df,
            accepted_epoch_id=accepted_epoch_id,
            yscale="log",
            include_title=True,
        )
        plt.savefig(
            os.path.join(subdirname, "log_loss_vs_epoch--with_title.png"),
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close()
        print("")
