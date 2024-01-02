#!/usr/bin/env python3
"""
Generate and serialize a shuffling of Cityscapes training set indices.

This script serializes the indices into
`REPO_ROOT/src/data/cityscapes/shuffled_cityscapes_train_ixs.csv`.

The file has been generated once and committed to the repository by the original
authors, so you shouldn't have to run this script again unless you want to
change the random seed. The purpose of committing the file is to ensure
consistency in construction of SubCityscapes training and validation datasets.
"""

# Standard
import os
import time

# Scientific Computing and Visualization
import numpy as np


REPO_ROOT: str = os.environ.get("REPO_ROOT")
assert REPO_ROOT is not None, \
    "REPO_ROOT not found! Did you run `setenv.sh` before serving the notebook?"


def main():
    np.random.seed(13)
    start_time = time.time()

    ixs: np.ndarray = np.array(range(2975), dtype=np.uint64)
    np.random.shuffle(ixs)

    filename: str = os.path.join(
        REPO_ROOT,
        "src/data/cityscapes/",
        "shuffled_cityscapes_train_ixs.csv",
    )

    np.savetxt(fname=filename, X=ixs, fmt="%d")

    ixs_recovered: np.ndarray = np.loadtxt(fname=filename, dtype=np.uint64)
    assert np.all(ixs == ixs_recovered)

    print(f"Elapsed time = {time.time() - start_time:.3f} s")


if __name__ == "__main__":
    main()
