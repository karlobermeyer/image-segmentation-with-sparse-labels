#!/usr/bin/env python3
"""
Compute and pickle a `TargetClassNumPixelsCache` instance which contains lists
of target class pixel counts for the most commonly used datasets.

Run this script like this.
```
# Takes ~20 min.
$ ./precompute_target_class_num_pixels.py
```

Load the cache from the pickle file for use in analysis and training scripts as
follows.
```
REPO_ROOT: str = os.environ.get("REPO_ROOT")
assert REPO_ROOT is not None, \
    "REPO_ROOT not found! Did you run `setenv.sh` before serving the notebook?"
num_pixels_cache_filename: str = os.path.join(
    REPO_ROOT,
    "src/data/cityscapes/",
    "target_class_num_pixels_cache.pkl",
)
with open(num_pixels_cache_filename, "rb") as fin:
    target_class_num_pixels_cache: TargetClassNumPixelsCache = pickle.load(fin)
assert target_class_num_pixels.has(
    dataset_name,
    len_scale_factor,
    label_density,
)
target_class_num_pixels: List[Counter[SemanticClassId]] = \
    target_class_num_pixels_cache.get(
        dataset_name,  # e.g., "subcityscapes_train"
        len_scale_factor,  # e.g., 0.5 or 1.0
        label_density,  # e.g., 0.5 or 1.0
    )
```

The pickle has been generated once and committed to the repository by the
original authors, so you shouldn't have to run this script again unless you want
to change or add more datasets.
"""

# Standard
import os
import pickle
import time

# Scientific Computing and Visualization
import numpy as np

# Project-Specific
from data.cityscapes.target_class_num_pixels_cache import \
    TargetClassNumPixelsCache


REPO_ROOT: str = os.environ.get("REPO_ROOT")
assert REPO_ROOT is not None, \
    "REPO_ROOT not found! Did you run `setenv.sh` before serving the notebook?"


def main():
    np.random.seed(13)
    start_time = time.time()

    cache: TargetClassNumPixelsCache = TargetClassNumPixelsCache(
        dataset_names = [
            "subcityscapes_train",
            "subcityscapes_val",
            "subcityscapes_trainval",  # == cityscapes_train
            "subcityscapes_test",  # == cityscapes_val
            "cityscapes_train",
            "cityscapes_val",
            "cityscapes_trainval",
        ],
        len_scale_factors=(1.0, 0.5),
        label_densities=(1.0, 0.5),
    )

    filename: str = os.path.join(
        REPO_ROOT,
        "src/data/cityscapes/",
        "target_class_num_pixels_cache.pkl",
    )
    with open(filename, "wb") as fout:
        pickle.dump(cache, fout)

    print(f"Elapsed time = {time.time() - start_time:.3f} s")


if __name__ == "__main__":
    main()
