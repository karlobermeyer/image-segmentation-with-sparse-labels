"""
This module provides functions for computing cooked semantic class weights to be
used by loss functions during semantic image segmentation model training.
"""
# Standard
from collections import Counter
from typing import List

# Numerics
import numpy as np

# Project-Specific
from core.identifiers import SemanticClassId
from core.semantic_class_mapping import SemanticClassMapping


def compute_semantic_class_num_pixels(
    class_mapping: SemanticClassMapping,
    target_class_num_pixels: List[Counter[SemanticClassId]],
) -> Counter[SemanticClassId]:
    """
    Compute and return joint semantic class occurrences for all targets.

    Args:
        class_mapping: semantic class mapping.
        target_class_num_pixels: list of semantic class occurrences for all
            targets, e.g., as returned by
            `SparseLabelSimulatingDataset.target_class_num_pixels`. The
            list runs over dataset examples. Each element is a `Counter` object,
            where keys are semantic class IDs and values are integer numbers of
            class occurrences in the respective target.

    Returns:
        class_num_pixels: a `Counter` object, where keys are cooked semantic
            class IDs and values are integer numbers of class occurrences.
    """
    class_num_pixels: Counter[SemanticClassId] = Counter(
        { class_id: 0 for class_id in class_mapping.class_ids }
    )
    for item in target_class_num_pixels:
        class_num_pixels.update(item)
    return class_num_pixels


def compute_semantic_class_frequencies(
    class_num_pixels: Counter[SemanticClassId],
) -> np.ndarray:
    """Compute and return cooked semantic class frequencies."""
    frequencies: np.ndarray = np.zeros(len(class_num_pixels))
    for class_id, num_pixels in class_num_pixels.items():
        frequencies[class_id] = num_pixels
    frequencies /= sum(frequencies)
    return frequencies


def smooth_semantic_class_frequencies(
    class_frequencies: np.ndarray,
    frequency_max_to_min_ratio_ubnd: float = 10.0,
) -> np.ndarray:
    """
    Reduce disparity between semantic class frequencies by uniformly adding
    probability mass and renormalizing.

    Args:
        class_frequencies: semantic class frequencies. Indices correspond with
            cooked semantic class IDs.
        frequency_max_to_min_ratio_ubnd: upper bound enforced on the ratio of
            the largest frequency to the smallest frequency in the output. This
            measure of smoothness was inspired by the condition number of a
            normal matrix, which is the ratio of largest to smallest eigenvalue
            magnitudes.

    Returns:
        class_frequencies_out: new array of smoother frequencies.
    """
    ratio_ubnd: float = frequency_max_to_min_ratio_ubnd  # Abbreviation.
    assert 1.0 <= ratio_ubnd
    assert abs(1.0 - sum(class_frequencies)) < 0.0001, \
        "Input frequencies must be normalized!"
    f_min: float = class_frequencies.min()
    f_max: float = class_frequencies.max()
    if 0.0 < f_min and f_max/f_min <= ratio_ubnd:
        return class_frequencies.copy()  # No changes necessary.
    # (f_max + x)/(f_min + x) = ratio_ubnd =>
    c: float = (ratio_ubnd*f_min - f_max)/(1.0 - ratio_ubnd)
    class_frequencies_out: np.ndarray = class_frequencies.copy() + c
    class_frequencies_out /= class_frequencies_out.sum()
    return class_frequencies_out


def compute_semantic_class_weights(class_frequencies: np.ndarray) -> np.ndarray:
    """
    Compute and return semantic class weights to help train on unbalanced
    datasets.

    Optionally apply `smooth_semantic_class_frequencies` before passing class
    frequencies to this function.

    Pass `torch.Tensor(class_weights)` to PyTorch functions.
    """
    class_weights: np.ndarray = 1.0/class_frequencies
    class_weights /= sum(class_weights)
    return class_weights
