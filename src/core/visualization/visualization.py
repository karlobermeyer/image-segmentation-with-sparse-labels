# Standard
from typing import Any, List, Sequence, Tuple

# Scientific and Visualization
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import imageio.v3 as imageio
import cv2

# ML
#import torch


def scale_plot_size(factor: float = 1.0):
    default_dpi = matplotlib.rcParamsDefault['figure.dpi']
    matplotlib.rcParams['figure.dpi'] = default_dpi*factor


def show_image(image):
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])


def overlay_segmentation_map_rgb(
    image: np.ndarray,
    segmentation_map_rgb: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.8,
) -> np.ndarray:
    if True:
        # This may be slightly better.
        image_with_overlay: np.ndarray = \
            (image*(1.0 - alpha) + segmentation_map_rgb*alpha).astype(np.uint8)
    else:
        image_with_overlay = cv2.addWeighted(
            src1=image,
            alpha=1.0,  # Transparency for original image.
            src2=segmentation_map_rgb,
            beta=beta,  # Transparency for segmentation map.
            gamma=0.0,  # Scalar added to each sum.
        )
    return image_with_overlay


def plot_grid_of_images(
    images: List[np.ndarray],
    num_cols: int,
    num_rows: int,
    vertical_spacing: float = 0.1,
    horizontal_spacing: float = 0.03,
    show_image_boundaries: bool = True,
) -> Tuple[Any, Any]:
    """
    Plot a grid of images using matplotlib subplots.

    Args:
        images: List of images represented as numpy arrays.
        num_cols: Number of grid columns.
        num_rows: Number of grid rows.
        horizontal_spacing: Horizontal spacing between images.
        vertical_spacing: Vertical spacing between images.
        show_image_boundaries: whether to show image boundaries as black lines.

    Returns:
        None
    """
    plt.clf()
    num_images = len(images)
    num_subplots = num_cols*num_rows

    max_image_width = 0.0
    max_image_height = 0.0
    for image in images:
        max_image_height = max(max_image_height, image.shape[0])
        max_image_width = max(max_image_width, image.shape[1])
    width_for_images = (num_cols*max_image_width)/1000.0
    height_for_images = (num_rows*max_image_height)/1000.0
    fig_width = width_for_images + (num_cols - 1)*horizontal_spacing
    fig_height = height_for_images + (num_rows - 1)*vertical_spacing
    # figsize := ((width, height))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    # Flatten axes if there's only one row or one column.
    if num_rows == 1:
        axs = axs.reshape(1, -1)
    if num_cols == 1:
        axs = axs.reshape(-1, 1)

    # Turn off axes clutter.
    for ax in axs.flat:
        if not show_image_boundaries:
            ax.axis('off')
            continue
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot images.
    for i in range(min(num_images, num_subplots)):
        ax = axs[i // num_cols, i % num_cols]
        ax.imshow(images[i])
        ax.margins(0.0, 0.0)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=horizontal_spacing, hspace=vertical_spacing)
    plt.margins(0.0, 0.0)
    #plt.tight_layout()

    return fig, axs


def construct_segmentation_side_by_sides(
    image_sequences: List[Sequence[np.ndarray]],
    interface_halfwidth: int = 2,
) -> List[np.ndarray]:
    """
    From a list of image sequences, construct a list of images, where each image
    is the horizontal concatenation of all the images in the respective
    sequence.

    This is useful, e.g., for showing semantic segmentation
    [ input | output | target ] triples.

    Args:
        image_sequences: list of RGB image sequences. All images within each
            sequence must have the same shape.
        interface_halfwidth: halfwidth (px) of black line to draw at interface
            between images. 0 => no line.

    Returns:
        List of images.
    """
    side_by_sides: List[np.ndarray] = []
    for image_sequence in image_sequences:
        for image in image_sequence[1:]:
            assert image_sequence[0].shape == image.shape
        height = image_sequence[0].shape[0]
        width = image_sequence[0].shape[1]
        num_images: int = len(image_sequence)

        combined_image = np.zeros((height, num_images*width, 3), dtype=np.uint8)
        for ix in range(num_images):
            combined_image[:, (ix*width):((ix + 1)*width), :] = \
                image_sequence[ix]

        if 0 < interface_halfwidth:  # Mark image interfaces.
            for ix in range(num_images - 1):
                center: int = (ix + 1)*width  # Interface line center.
                combined_image[
                    :, (center - interface_halfwidth) \
                        :(center + interface_halfwidth), :
                ] = 0

        side_by_sides.append(combined_image)

    return side_by_sides


def plot_semantic_class_frequencies(
    class_names: List[str],
    class_frequencies: np.ndarray,
    bar_label_angle: float = 70,
    title: str = "Semantic Class Frequencies",
) -> None:
    #plt.figure(figsize = (13, 5))
    plt.xlabel("Class ID", fontweight="bold", fontsize=12)
    plt.ylabel("Frequency", fontweight="bold", fontsize=12)
    plt.title(title, fontweight="bold", fontsize=16)
    class_strs: List[str] = [
        f"{class_id}: {class_name}" for class_id, class_name in \
        enumerate(class_names)
    ]
    plt.bar(class_strs, class_frequencies, color="orange", width=0.4)
    plt.xticks(rotation=bar_label_angle)


def plot_semantic_class_weights(
    class_names: List[str],
    class_weights: np.ndarray,
    bar_label_angle: float = 70,
    title: str = "Semantic Class Weights",
) -> None:
    #plt.figure(figsize = (13, 5))
    plt.xlabel("Class ID", fontweight="bold", fontsize=12)
    plt.ylabel("Weight", fontweight="bold", fontsize=12)
    plt.title(title, fontweight="bold", fontsize=16)
    class_strs: List[str] = [
        f"{class_id}: {class_name}" for class_id, class_name in \
        enumerate(class_names)
    ]
    plt.bar(class_strs, class_weights, color="purple", width=0.4)
    plt.xticks(rotation=bar_label_angle)
