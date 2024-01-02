#!/usr/bin/env python3
"""
Evaluate an image segmenter training run.

Evaluation is performed on a single validation or test subset of SubCityscapes
or Cityscapes. Run this script multiple times to evaluate on multiple datasets,
or call the `evaluate_training_run` function from another script.

The Cityscapes test dataset is a special case. Its labels are not public, so
metrics cannot be computed locally for that dataset only. Instead, a zip
file of model outputs is created for upload to the Cityscapes test server.

Command-Line Usage:
```
# View documentation.
$ ./evaluate_training_run.py -h

# Example minimal pattern (fewest arguments provided):
$ ./evaluate_training_run.py \
    --model MODEL \
    --scenario SCENARIO \
    --dataset_eval DATASET
# Among all runs of the specified scenario, this finds and evaluates the one
# with the latest training datetime. Model outputs are not saved by default.
# Here is a concrete instance of this pattern:
$ ./evaluate_training_run.py \
    --model deeplabv3 \
    --scenario smoke_test \
    --dataset_eval subcityscapes_test

# Example maximal pattern (all possible arguments provided):
$ ./evaluate_training_run.py \
    --model MODEL \
    --scenario SCENARIO \
    --datetime YYYY-MM-DDTHH_MM_SSZ \
    --dataset_eval DATASET \
    --do_cityscapes_eval
```

Results subdirectories are stored in subdirectory `final_model_evaluation/` of
the run directory. After running this script, the contents of the run directory
should look roughly like this, with some variation depending on options:
REPO_ROOT/runs/DATASET/MODEL/SCENARIO/YYYY-MM-DDTHH_MM_SSZ/
├── ...    ┐
├── ...    │ Preexisting items from training run
├── ...    ┘
│
└──final_model_evaluation/
    ├── subcityscapes_val/   <-- If run with dataset_eval="subcityscapes_val"
    │   ├── ious_background_excluded.csv     ┐
    │   ├── ious_background_excluded.txt     │
    │   ├── ious.csv                         │ Class-level metrics tables
    │   ├── ious.txt                         │
    │   ├── pixel_accuracies.csv             │
    │   ├── pixel_accuracies.txt             ┘
    │   ├── summary.yaml   <------- Summary metrics
    │   ├── examples--04.png                  ┐
    │   ├── examples--04--with_headings.png   │ Example images
    │   ├── examples--06.png                  │
    │   └── examples--06--with_headings.png   ┘
    │
    ├── subcityscapes_test/   <-- If run with dataset_eval="subcityscapes_test"
    │   ├── ious_background_excluded.csv     ┐
    │   ├── ious_background_excluded.txt     │
    │   ├── ious.csv                         │ Class-level metrics tables
    │   ├── ious.txt                         │
    │   ├── pixel_accuracies.csv             │
    │   ├── pixel_accuracies.txt             ┘
    │   ├── summary.yaml   <------- Summary metrics
    │   ├── examples--04.png                  ┐
    │   ├── examples--04--with_headings.png   │ Example images
    │   ├── examples--06.png                  │
    │   └── examples--06--with_headings.png   ┘
    │
    ├── cityscapes_val/  <-- If run with dataset_eval="cityscapes_val"
    │   ├── ious_background_excluded.csv     ┐
    │   ├── ious_background_excluded.txt     │
    │   ├── ious.csv                         │ Class-level metrics tables
    │   ├── ious.txt                         │
    │   ├── pixel_accuracies.csv             │
    │   ├── pixel_accuracies.txt             ┘
    │   ├── summary.yaml   <------- Summary metrics
    │   ├── examples--04.png                  ┐
    │   ├── examples--04--with_headings.png   │ Example images
    │   ├── examples--06.png                  │
    │   ├── examples--06--with_headings.png   ┘
    │   └── cityscapes_eval
    │       ├── model_outputs--raw_class_ids
    │       │   └── *.png
    │       └── cityscapes_eval.txt   <-- Results of Cityscapes eval script
    │
    └── cityscapes_test/   <-- If run with dataset_eval="cityscapes_test"
        ├── examples--04.png                  ┐
        ├── examples--04--with_headings.png   │ Example images
        ├── examples--06.png                  │
        ├── examples--06--with_headings.png   ┘
        └── cityscapes_eval
            ├── model_outputs--raw_class_ids
            │   └── *.png
            └── model_outputs--raw_class_ids.zip   <-- For upload to test server

_References_
https://www.cityscapes-dataset.com/benchmarks/
https://www.cityscapes-dataset.com/create-submission/
https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py
"""

# Standard
from argparse import ArgumentParser
import gc
import os
import shutil
import subprocess
import time
from typing import Dict, List, Set
import zipfile

# Numerics
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from tqdm import tqdm

# Image Processing
import imageio.v3 as imageio

# Machine Learning
import albumentations as albm  # Faster than `torchvision.transforms`.
from cityscapesscripts.evaluation import evalPixelLevelSemanticLabeling

# Project-Specific
from core import (
    #SemanticClassId,
    SparseLabelSimulatingDataset,
)
from core.metrics import ImagesIou, ImagesPixelAccuracy
from data.cityscapes import (
    subcityscapes_dataset,
    cityscapes_dataset,
    class_mapping,
    preprocesses_from,
    shortened_target_filename_leaf,
)
from models import (
    #ImageSegmenter,
    DeepLabV3ImageSegmenterHparams,
    deeplabv3_image_segmenter_hparams_from_yaml,
    #DeepLabV3ImageSegmenter,
    SegformerImageSegmenterHparams,
    segformer_image_segmenter_hparams_from_yaml,
    #SegformerImageSegmenter,
)
from experiments.training_runs_access import (
    model_name_from_run_dirname,
    get_latest_run_dirname,
)
from experiments.training_run import TrainingRun


REPO_ROOT: str = os.environ.get("REPO_ROOT")
assert REPO_ROOT is not None, \
    "REPO_ROOT not found! Did you run `setenv.sh`?"

CITYSCAPES_DATA_ROOT: str = os.path.join(REPO_ROOT, "data/cityscapes")


def abbreviate_runs_path(path: str) -> str:
    """Abbreviate runs path for inclusion in printed statements."""
    return ".../runs/" + path.split("/runs/")[1]


def evaluate_training_run(
    run_dirname: str,
    dataset_eval_name: str,
    do_cityscapes_eval: bool = False,
) -> Dict[str, float]:
    """
    Evaluate an image segmenter training run.

    Args:
        run_dirname: run directory name.
        dataset_eval_name: name of dataset to evaluate model against.
        do_cityscapes_eval: whether to do extra computations to evaluate the
            model using the Cityscapes evaluation script.

    Returns:
        summary: dictionary of summary metrics.
    """
    assert os.path.exists(run_dirname), "Missing run directory!"

    model_name: str = model_name_from_run_dirname(run_dirname)
    assert model_name in ("deeplabv3", "segformer"), "Unknown model!"

    assert dataset_eval_name in (
        "subcityscapes_val",
        "subcityscapes_test",
        "cityscapes_val",
        "cityscapes_test",
    ), "Unrecognized dataset!"

    if do_cityscapes_eval:
        assert dataset_eval_name in \
            ("cityscapes_val", "cityscapes_test"), \
            "Invalid dataset for using Cityscapes evaluation script!"

    eval_superdirname: str = os.path.join(run_dirname, "final_model_evaluation")
    eval_dirname: str = os.path.join(eval_superdirname, dataset_eval_name)
    if os.path.exists(eval_dirname):
        shutil.rmtree(eval_dirname)
    os.makedirs(eval_dirname, exist_ok=False)

    hparams_filename: str = os.path.join(run_dirname, "hparams.yaml")
    if model_name == "deeplabv3":
        hparams: DeepLabV3ImageSegmenterHparams = \
            deeplabv3_image_segmenter_hparams_from_yaml(hparams_filename)
    elif model_name == "segformer":
        hparams: SegformerImageSegmenterHparams = \
            segformer_image_segmenter_hparams_from_yaml(hparams_filename)
    else:
        raise NotImplementedError
    assert model_name == hparams.model_name, \
        "Model name in run directory name should match " \
        "model specified in hyperparameters!"

    print("Evaluating run in")
    print(f"{abbreviate_runs_path(run_dirname)}/\n")

    # Load evaluation dataset.
    preprocesses: Dict[str, albm.Compose] = preprocesses_from(
        input_height=hparams.input_height,
        input_width=hparams.input_width,
        mean_for_input_normalization=hparams.mean_for_input_normalization,
        std_for_input_normalization=hparams.std_for_input_normalization,
        do_shift_scale_rotate=True,
        ignore_index=hparams.ignore_index,
    )
    if "subcityscapes" in dataset_eval_name:
        dataset_eval: SparseLabelSimulatingDataset = subcityscapes_dataset(
            split=dataset_eval_name.split("_")[1],
            preprocess=preprocesses["infer"],
            len_scale_factor=1.0,
            label_density=1.0,
            ignore_index=hparams.ignore_index,
            shuffle=False,
        )
    else:
        dataset_eval: SparseLabelSimulatingDataset = cityscapes_dataset(
            split=dataset_eval_name.split("_")[1],
            preprocess=preprocesses["infer"],
            len_scale_factor=1.0,
            label_density=1.0,
            ignore_index=hparams.ignore_index,
            shuffle=False,
        )

    run: TrainingRun = TrainingRun(run_dirname)

    # Create and save a variety of side-by-side images showing original image
    # together with segmentation map(s). The first few were hand-picked for
    # subcityscapes_test aka cityscapes_val.
    ixs: List[int] = [15, 170, 181, 304, 316, 367]
    num_examples: int = len(dataset_eval)
    ixs_other_set: Set[int] = set(range(num_examples))
    ixs_other_set.difference_update(ixs)
    ixs_other: List[int] = list(ixs_other_set)
    np.random.shuffle(ixs_other)
    ixs.extend(ixs_other)
    #
    print(
        "Creating and saving side-by-side images with 4 examples."
    )
    run.plot_examples(
        dataset=dataset_eval,
        example_ixs=ixs[:4],
        include_targets=True,
        include_column_headings=False,
    )
    plt.savefig(
        os.path.join(
            eval_dirname,
            "examples_val--04.png",
        ),
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=500,
    )
    plt.close()
    #
    run.plot_examples(
        dataset=dataset_eval,
        example_ixs=ixs[:4],
        include_targets=True,
        include_column_headings=True,
    )
    plt.savefig(
        os.path.join(
            eval_dirname,
            "examples_val--04--with_headings.png",
        ),
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=500,
    )
    plt.close()

    print("Creating and saving side-by-side images with 6 examples.")
    run.plot_examples(
        dataset=dataset_eval,
        example_ixs=ixs[:6],
        include_targets=True,
        include_column_headings=False,
    )
    plt.savefig(
        os.path.join(
            eval_dirname,
            "examples_val--06.png",
        ),
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=500,
    )
    plt.close()
    #
    run.plot_examples(
        dataset=dataset_eval,
        example_ixs=ixs[:6],
        include_targets=True,
        include_column_headings=True,
    )
    plt.savefig(
        os.path.join(
            eval_dirname,
            "examples_val--06--with_headings.png",
        ),
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=500,
    )
    plt.close

    print("\nComputing metrics.")

    if do_cityscapes_eval:
        # Create directories for Cityscapes evaluation script inputs and
        # outputs.
        cityscapes_eval_dirname: str = \
            os.path.join(eval_dirname, "cityscapes_eval")
        if os.path.exists(cityscapes_eval_dirname):
            shutil.rmtree(cityscapes_eval_dirname)
        os.makedirs(cityscapes_eval_dirname, exist_ok=False)
        model_outputs_dirname_leaf: str = "model_outputs--raw_class_ids"
        model_outputs_dirname: str = os.path.join(
            cityscapes_eval_dirname,
            model_outputs_dirname_leaf,
        )
        if os.path.exists(model_outputs_dirname):
            shutil.rmtree(model_outputs_dirname)
        os.makedirs(model_outputs_dirname, exist_ok=False)
        print(
            f"For Cityscapes evaluation script, saving model outputs to"
            f"\n{abbreviate_runs_path(model_outputs_dirname)}"
        )

    gc.collect()

    images_iou: ImagesIou = ImagesIou(class_mapping=class_mapping)
    images_iou_background_excluded: ImagesIou = ImagesIou(
        class_mapping=class_mapping,
        class_ids_excluded=set((0,)),
    )
    images_pixel_accuracy: ImagesPixelAccuracy = \
        ImagesPixelAccuracy(class_mapping=class_mapping)
    summary: Dict[str, float] = {
        "miou_background_excluded": float("nan"),
        "miou": float("nan"),
        "pixel_accuracy": float("nan"),
        "mean_pixel_accuracy": float("nan"),
    }

    #num_examples: int = 10  # For testing only.
    num_examples: int = len(dataset_eval)
    for ix in tqdm(range(num_examples)):
        image, target = dataset_eval.get_unpreprocessed(ix)
        #model_output: np.ndarray = target.copy()  # For testing only.
        model_output: np.ndarray = run.segmenter.segment(image)
        _, target_filename = dataset_eval.get_filenames(ix)

        if do_cityscapes_eval:
            # Save version of model output with raw class IDs for Cityscapes
            # evaluation script.
            model_output_filename: str = os.path.join(
                model_outputs_dirname,
                shortened_target_filename_leaf(target_filename),
            )
            model_output_raw: np.ndarray = \
                class_mapping.segmentation_map_int_to_raw(model_output)
            imageio.imwrite(model_output_filename, model_output_raw)

        # Update statistics.
        images_iou.update(model_output, target)
        images_iou_background_excluded.update(model_output, target)
        images_pixel_accuracy.update(model_output, target)

    # Write metrics to files.
    if dataset_eval_name != "cityscapes_test":
        print("Writing metrics to")
        #
        metrics_filename_txt: str = \
            os.path.join(eval_dirname, "ious.txt")
        metrics_filename_csv: str = \
            os.path.join(eval_dirname, "ious.csv")
        print(
            f"{abbreviate_runs_path(metrics_filename_txt)}"
            f"\n{abbreviate_runs_path(metrics_filename_csv)}"
        )
        images_iou.write_txt(metrics_filename_txt)
        images_iou.write_csv(metrics_filename_csv)
        #
        metrics_filename_txt: str = \
            os.path.join(eval_dirname, "ious_background_excluded.txt")
        metrics_filename_csv: str = \
            os.path.join(eval_dirname, "ious_background_excluded.csv")
        print(
            f"{abbreviate_runs_path(metrics_filename_txt)}"
            f"\n{abbreviate_runs_path(metrics_filename_csv)}"
        )
        images_iou_background_excluded.write_txt(metrics_filename_txt)
        images_iou_background_excluded.write_csv(metrics_filename_csv)
        #
        metrics_filename_txt: str = \
            os.path.join(eval_dirname, "pixel_accuracies.txt")
        metrics_filename_csv: str = \
            os.path.join(eval_dirname, "pixel_accuracies.csv")
        print(
            f"{abbreviate_runs_path(metrics_filename_txt)}"
            f"\n{abbreviate_runs_path(metrics_filename_csv)}"
        )
        images_pixel_accuracy.write_txt(metrics_filename_txt)
        images_pixel_accuracy.write_csv(metrics_filename_csv)
        #
        metrics_filename_yaml: str = \
            os.path.join(eval_dirname, "summary.yaml")
        print(f"{abbreviate_runs_path(metrics_filename_yaml)}")
        summary["miou_background_excluded"] = images_iou.miou()
        summary["miou"] = images_iou_background_excluded.miou()
        summary["pixel_accuracy"] = images_pixel_accuracy.pixel_accuracy()
        summary["mean_pixel_accuracy"] = \
            images_pixel_accuracy.mean_pixel_accuracy()
        with open(metrics_filename_yaml, "w") as fout:
            # The Cityscapes project shows float metrics with 6 significant
            # figures on the leader board, so we match that here.
            fout.write(
                "miou_background_excluded: "
                f"{summary['miou_background_excluded']:.6f}\n"
            )
            fout.write(f"miou: {summary['miou']:.6f}\n")
            fout.write(f"pixel_accuracy: {summary['pixel_accuracy']:.6f}\n")
            fout.write(
                f"mean_pixel_accuracy: {summary['mean_pixel_accuracy']:.6f}\n"
            )

    if do_cityscapes_eval and dataset_eval_name == "cityscapes_val":
        print("\nRunning Cityscapes evaluation script.")
        # Run Cityscapes evaluation script on the validation dataset of 500
        # images. Usu. takes ~70 s.
        CITYSCAPES_DATASET: str = os.environ.get("CITYSCAPES_DATASET")
        assert CITYSCAPES_DATASET == os.path.join(REPO_ROOT, "data/cityscapes/")
        CITYSCAPES_RESULTS: str = model_outputs_dirname
        os.environ["CITYSCAPES_RESULTS"] = CITYSCAPES_RESULTS
        assert CITYSCAPES_RESULTS == os.environ.get("CITYSCAPES_RESULTS")
        cityscapes_eval_script_filename: str = \
            evalPixelLevelSemanticLabeling.__file__
        command: List[str] = ["python", cityscapes_eval_script_filename]
        cityscapes_eval_results_filename: str = \
            os.path.join(cityscapes_eval_dirname, "cityscapes_eval.txt")
        with open(cityscapes_eval_results_filename, "w") as fout:
            try:
                subprocess.run(
                    command,
                    check=True,
                    stdout=fout,
                    stderr=subprocess.PIPE,
                )
                print("Successfully ran Cityscapes evaluation script.")
            except subprocess.CalledProcessError as e:
                print("Error running Cityscapes evaluation script. See logs.")
                fout.write(f"Error: {e}\n")

    if do_cityscapes_eval and dataset_eval_name == "cityscapes_test":
        os.chdir(cityscapes_eval_dirname)
        zip_filename_leaf: str = model_outputs_dirname_leaf + ".zip"
        zip_filename: str = os.path.join(
            cityscapes_eval_dirname,
            zip_filename_leaf,
        )
        print(
            f"\nCompressing model outputs into zip file\n"
            f"{abbreviate_runs_path(zip_filename)}"
        )
        with zipfile.ZipFile(zip_filename, "w") as zip_fout:
            for _, _, filename_leaves in os.walk(model_outputs_dirname):
                for filename_leaf in filename_leaves:
                    #print(filename_leaf)
                    zip_fout.write(os.path.join(
                        model_outputs_dirname_leaf,
                        filename_leaf,
                    ))
        print(
            "Upload this zip file to the Cityscapes "
            "test server for further evaluation."
        )

    return summary


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        description="Evaluate an image segmenter training run. Results are " \
            "stored in a subdirectory `final_model_evaluation/DATASET_EVAL/` " \
            "of the run directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name.",
        required=True,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario name.",
        required=True,
    )
    parser.add_argument(
        "--datetime",
        type=str,
        help="UTC datetime of run in the format YYYY-MM-DDTHH_MM_SSZ. " \
            "Not provided => latest available datetime is used.",
        required=False,
    )
    parser.add_argument(
        "--dataset_eval",
        type=str,
        help="Name of dataset to evaluate the model against: " \
            "subcityscapes_val, subcityscapes_test, cityscapes_val, or " \
            "cityscapes_test.",
        required=True,
    )
    parser.add_argument(
        "--do_cityscapes_eval",
        action="store_true",
        help="Including this no-argument flag causes model outputs to be "
        "saved and extra computations to be performed for the Cityscapes "
        "evaluation script. This only works when evaluating against the "
        "`cityscapes_val` and `cityscapes_test` datasets. For `cityscapes_val`,"
        " model outputs are saved and the evaluation script is run locally. "
        "For `cityscapes_test`, model outputs are saved and a zip file is "
        "created for upload to the Cityscapes test server.",
        required=False,
        default=False,
    )
    args = parser.parse_args()

    scenario_dirname: str = \
        os.path.join(REPO_ROOT, "runs", args.model, args.scenario)
    assert os.path.exists(scenario_dirname), "Missing scenario directory!"

    if args.datetime is None:
        run_dirname: str = get_latest_run_dirname(scenario_dirname)
    else:
        run_dirname: str = os.path.join(scenario_dirname, args.datetime)
    assert os.path.exists(run_dirname), "Missing run directory!"

    start_time: float = time.time()

    summary: Dict[str, float] = evaluate_training_run(
        run_dirname=run_dirname,
        dataset_eval_name=args.dataset_eval,
        do_cityscapes_eval=args.do_cityscapes_eval,
    )

    if args.dataset_eval != "cityscapes_test":
        print("\n_Summary Metrics_")
        print(
            "miou_background_excluded: "
            f"{summary['miou_background_excluded']:.5f}"
        )
        print(f"miou: {summary['miou']:.5f}")
        print(f"pixel_accuracy: {summary['pixel_accuracy']:.5f}")
        print(
            f"mean_pixel_accuracy: {summary['mean_pixel_accuracy']:.5f}"
        )

    elapsed_time: float = time.time() - start_time
    print(f"\nElapsed time = {elapsed_time/3600.0:.3f} h")
