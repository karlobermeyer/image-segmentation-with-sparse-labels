# image-segmentation-with-sparse-labels [WIP]

In this project we use PyTorch+Lightning to train and evaluate deep neural
networks for semantic image segmentation on the Cityscapes dataset [[1](#1)]
with an aim at studying the effects of sparse labels. Specifically, we seek
evidence about whether better segmentation performance can be obtained by
training on `2*n` half-labeled images rather than `n` fully-labeled images for
some `n`.

![Cityscapes Dense and Sparse Labels Examples](
  images_for_readme/cityscapes_dense_vs_sparse_labels_examples.png
)

In a very simplistic and cursory experiment, we started with a DeepLabV3 model
[[5](#5)] pretrained on COCO-Stuff data [[2](#2)-[3](#3)] with Pascal VOC
semantic classes [[4](#4)]. We fine-tuned only the final layer on a small subset
of the Cityscapes data using constant learning rate and standard cross-entropy
loss. Not surprisingly, the segmentation performance was far below SOTA (State
Of The Art), but the model from the training run with `2*n` half-labeled images
did perform slightly better than the model from the training run with `n`
images. Metrics tables and visualizations can be viewed in
[this notebook](notebooks/03--compare_experiment_runs.ipynb). In the next set of
experiments, when we train all layers, we expect that the performance gain from
using sparse labels will be much greater because the final network layer
represents more localized structure, but the information shared across image
halves is more global. *Further work is in progress to test the sparse labeling
effect for improved training schemes where the models achieve near-SOTA
performance.*


## Table of Contents

- [Files and Directories](#files-and-directories)
- [Using](#using)
  * [Initial Setup](#initial-setup)
  * [After Initial Setup](#after-initial-setup)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Running Jupyter Notebooks](#running-jupyter-notebooks)
  * [Code Testing](#code-testing)
  * [Teardown](#teardown)
  * [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)


## Files and Directories

```
./
├── README.md   <-- You are here.
├── CITATION.bib
├── LICENSE
├── pyproject.toml   <-- Ruff linter configuration.
├── requirements.txt   <-- Dependencies.
├── setenv.sh   <-- Script automatically creates and loads virtual env.
│
├── data
│   ├── get_cityscapes_data.sh   <-- Script downloads Cityscapes data.
│   └── cityscapes   <-- Landing directory for downloaded data.
│       └── ...
│
├── src   <-- Source code.
│   ├── core   <-- Basic segmentation and analysis tools.
│   │   ├── sparse_label_simulating_dataset.py
│   │   └── ...
│   ├── data   <-- Data access helpers.
│   │   ├── cityscapes   <-- The dataset we train and evaluate with.
│   │   │   └── ...
│   │   └── pascal_voc   <-- DeepLabV3 pretrained semantic classes.
│   │       └── ...
│   ├── models   <-- Lightning modules that wrap PyTorch models.
│   │   ├── deeplabv3   <-- The model we're currently experimenting with.
│   │   │   └── ...
│   │   ├── segformer   <-- Not fully implemented yet.
│   │   │   └── ...
│   │   ├── image_segmenter.py       <-- Subclass `ImageSegmenter` to try
│   │   ├── image_segmenter_hparams.py   different models.
│   │   └── ...
│   └── experiments   <-- Training and evaluation scripts.
│       ├── deeplabv3
│       │   └── ...   <-- Hyperparameters for different DeepLabV3 training runs.
│       ├── train_image_segmenter.py   <--- Training script.
│       ├── evaluate_training_run.py   <--- Evaluation script.
│       └── ...
│
├── runs  <-- Landing directory for training run results.
│   └── ...
│
└── notebooks
    ├── 00--eda_raw_cityscapes.ipynb
    ├── 01--inspect_cityscapes_cooked_class_frequencies.ipynb
    ├── 02--inspect_pretrained_deeplabv3.ipynb
    └── 03--compare_experiment_runs.ipynb
```


## Using

### Initial Setup

0. Clone this
  [repository](https://github.com/karlobermeyer/image-segmentation-with-sparse-labels)
  into a directory of your choice, denoted henceforth by `${PROJECT_ROOT}`.

1. Obtain a username and password for the
  [Cityscapes](https://www.cityscapes-dataset.com/login/) project.

2. Insert your Cityscapes credentials into the file `${PROJECT_ROOT}/.env`.
  These are to remain secret, so do not commit this change to the repository.

3. Get Cityscapes data [[1](#1)].
```
$ cd ${PROJECT_ROOT}/data
$ ./get_cityscapes_data.sh
```

4. Use `setenv.sh` to set environment variables, modify some search function
  definitions, and create and activate the Python virtual environment
  `image-segmentation-with-sparse-labels-env`. This virtual environment includes
  all the project dependencies listed in `${PROJECT_ROOT}/requirements.txt`
```
$ cd ${PROJECT_ROOT}
$ source ./setenv.sh
```

Your environment should now be ready to run any of the code and serve any of the
Jupyter notebooks.

### After Initial Setup

After you have completed [Initial Setup](#initial-setup), assuming data
persistence, you can open the project in a fresh terminal and set up the
environment by simply running `setenv.sh` again.
```
$ cd ${PROJECT_ROOT}
$ source ./setenv.sh
```
This will run much faster after the first time because it merely activates the
virtual environment rather than rebuilding from scratch.

### Training

To execute a training run, choose a model, create a hyperparameters yaml file
for it, and run `src/experiments/train_image_segmenter.py` on it. For example, a
short smoke test can be run as follows.
```
$ cd ${PROJECT_ROOT}/src/experiments
$ ./train_image_segmenter.py \
    --model deeplabv3 \
    --hparams deeplabv3/hparams--smoke_test.yaml
```

The 4 cursory final-layer experiment runs, which are much longer, can be
executed as follows.
```
$ cd ${PROJECT_ROOT}/src/experiments
$ ./train_image_segmenter.py \
    --model deeplabv3 \
    --hparams deeplabv3/hparams--final_layer_000.yaml
$ ./train_image_segmenter.py \
    --model deeplabv3 \
    --hparams deeplabv3/hparams--final_layer_001.yaml
$ ./train_image_segmenter.py \
    --model deeplabv3 \
    --hparams deeplabv3/hparams--final_layer_002.yaml
$ ./train_image_segmenter.py \
    --model deeplabv3 \
    --hparams deeplabv3/hparams--final_layer_003.yaml
```

See the header of `src/experiments/train_image_segmenter.py` for further
instructions.

### Evaluation

Once a training run is complete, you can evaluate the model against various
datasets using `src/experiments/evaluate_training_run.py`. For example, generate
evaluation results for the short smoke test as follows.
```
$ cd ${PROJECT_ROOT}/src/experiments
$ ./evaluate_training_run.py \
    --model deeplabv3 \
    --scenario smoke_test \
    --dataset_eval subcityscapes_test
```

Generate evaluation results for the 4 cursory final-layer experiment runs as
follows.
```
$ cd ${PROJECT_ROOT}/src/experiments
$ ./evaluate_training_run.py \
    --model deeplabv3 \
    --scenario final_layer_000 \
    --dataset_eval subcityscapes_test
$ ./evaluate_training_run.py \
    --model deeplabv3 \
    --scenario final_layer_001 \
    --dataset_eval subcityscapes_test
$ ./evaluate_training_run.py \
    --model deeplabv3 \
    --scenario final_layer_002 \
    --dataset_eval subcityscapes_test
$ ./evaluate_training_run.py \
    --model deeplabv3 \
    --scenario final_layer_003 \
    --dataset_eval subcityscapes_test
```

See the header of `src/experiments/evaluate_training_run.py` for further
instructions.

The Cityscapes test labels are not publicly available, so if you want to
evaluate model outputs against the Cityscapes test set, you must be send them to
the Cityscapes test server. Follow the Cityscapes test server
[submission instructions](https://www.cityscapes-dataset.com/create-submission/).


### Running Jupyter Notebooks

Execute the following to run the Jupyter notebooks in `notebooks/`.

```
$ cd ${PROJECT_ROOT}
$ source ./setenv.sh  # If you haven't already.

# Serve Jupyter to port 8888.
$ cd ${PROJECT_ROOT}/notebooks
$ nohup jupyter notebook --no-browser --port 8888 & disown
```

The notebook can be viewed in your web browser locally at
`http://localhost:8888/`, or on another machine if you catch the forwarded port.
```
# Catch forwarded port locally.
$ nohup ssh -NL 8888:localhost:8888 username@host
```

### Code Testing

Pytests are present only for the trickiest and most critical parts of the code.
Ruff is available for linting. Some "unused import" errors deliberately remain
for debugging convenience, and where the dependency is required in common
temporary modifications. Those may be removed eventually.

```
$ cd ${PROJECT_ROOT}/src
$ pytest  # Run pytests.
$ ruff .  # Lint code.
```

```
$ cd ${PROJECT_ROOT}/notebooks
$ nbqa ruff .  # Lint notebooks.
```

### Teardown

```
# Stop serving Jupyter to port XXXX (usu. 8888).
$ jupyter notebook stop XXXX

# Deactivate the Python virtual environment.
$ deactivate
```

### Troubleshooting

```
# Confirm PyTorch is installed with GPU support.
$ python -c "import torch; print(torch.cuda.is_available())"
$ python -c "import torch; print(torch.version.cuda)"
```

If somehow you have trouble with your virtual environment, you can try deleting
and rebuilding it.
```
$ cd ${PROJECT_ROOT}
$ rm -rf image-segmentation-with-sparse-labels-env  # Remove virtual env.
$ source ./setenv.sh  # Rebuild virtual environment.
```

We have chosen to use virtual environments instead of conda for speed and
portability. If you are concurrently running conda on your system, it should be
that
* Packages installed in a virtual environment using pip will not be visible to
  conda, and
* Packages installed in the conda environment will not be visible to the virtual
  environment.


## References

<a id="1">[1]</a> 
["The Cityscapes Dataset for Semantic Urban Scene
Understanding"](https://www.cityscapes-dataset.com/) by M. Cordts et al, 2016.

<a id="2">[2]</a>
["Microsoft COCO: Common Objects in Context"](https://arxiv.org/abs/1405.0312)
by T.-Y. Lin et al, 2015.

<a id="3">[3]</a>
["COCO-Stuff: Thing and Stuff Classes in
Context"](https://arxiv.org/abs/1612.03716) by H. Caesar et al, 2016.

<a id="4">[4]</a>
["The Pascal Visual Object Classes Challenge: A
Retrospective"](http://host.robots.ox.ac.uk/pascal/VOC/) by M. Everingham et al,
2015.

<a id="5">[5]</a>
["Rethinking Atrous Convolution for Semantic Image
Segmentation"](https://arxiv.org/abs/1706.05587) by L.-C. Chen et al, 2017.


## License

The code for this project is [Apache2](LICENSE.txt) licensed. The open datasets
used by this project are subject to their own respective licenses.
