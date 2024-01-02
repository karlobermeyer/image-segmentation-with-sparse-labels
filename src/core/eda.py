"""
EDA (Exploratory Data Analysis) utilities for semantic image segmentation.
"""
# Standard
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple

# Numerics
import numpy as np

# Image Processing
import imageio.v3 as imageio

# Visualization
#import matplotlib.pyplot as plt
#from tqdm import tqdm

# Other 3rd-Party
import texttable  # https://pypi.org/project/texttable/

# Project-Specific
from core.identifiers import SemanticClassId
from core.semantic_class_mapping import SemanticClassMapping


def pdir(
    object: Any,
    include_magic: bool = True,
    include_private: bool = True,
) -> str:
    """Inspect object attribute names."""
    names: List[str] = dir(object)
    names_reduced: List[str] = []
    for name in names:
        is_magic: bool = name.startswith("__") and name.endswith("__")
        is_private: bool = name.startswith("_") and not is_magic
        if include_magic and is_magic:
            names_reduced.append(name)
            continue
        elif include_private and is_private:
            names_reduced.append(name)
            continue
        elif not is_magic and not is_private:
            names_reduced.append(name)
    pprint(names_reduced)


def inspect_dataset_list_attribute(
    dataset: List[Any],
    attribute_name: str,
) -> None:
    value = getattr(dataset, attribute_name)
    print("type(value):", type(value))
    print("type(value[0]):", type(value[0]))
    print("len(value):", len(value))
    # print("value:", value)
    print("value:")
    for item in value[:3]:
      print(item)
    print(".\n.\n.\n")


def print_image_metadata(image: np.ndarray) -> None:
    print(f"type: {type(image)}")
    print(f"dtype: {image.dtype}")
    print(f"shape: {image.shape}")


def image_shape_from_file(image_filename: str) -> Tuple[int, int]:
    image: np.ndarray = imageio.imread(image_filename)
    return image.shape  # (height, width).


class ImagesShapeStats:
    def __init__(self, image_filenames: List[str]) -> None:
        num_workers_ubnd = max(1, 3*os.cpu_count()//4)
        with ProcessPoolExecutor(max_workers=num_workers_ubnd) as executor:
            image_shapes: List[Tuple[int, int]] = \
                list(executor.map(image_shape_from_file, image_filenames))
        assert len(image_shapes) == len(image_filenames)
        self.heights: np.ndarray = \
            np.array([shape[0] for shape in image_shapes])
        self.widths: np.ndarray = \
            np.array([shape[1] for shape in image_shapes])

        # As conventional, aspect ratio is width:height.
        self.aspect_ratios = self.widths/self.heights

        self.mean_height = np.mean(self.heights)
        self.median_height = np.median(self.heights)
        self.min_height = np.min(self.heights)
        self.max_height = np.max(self.heights)

        self.mean_width = np.mean(self.widths)
        self.median_width = np.median(self.widths)
        self.min_width = np.min(self.widths)
        self.max_width = np.max(self.widths)

        self.mean_aspect_ratio = np.mean(self.aspect_ratios)
        self.median_aspect_ratio = np.median(self.aspect_ratios)
        self.min_aspect_ratio = np.min(self.aspect_ratios)
        self.max_aspect_ratio = np.max(self.aspect_ratios)

    def print_summary(self) -> None:
        table = texttable.Texttable()
        table.set_cols_align(["r", "c", "c", "c"])
        table.add_rows([
            ["", "height", "width", "aspect ratio (width:height)"],
            ["mean", self.mean_height, self.mean_width, self.mean_aspect_ratio],
            ["median", self.median_height, self.median_width, self.median_aspect_ratio],
            ["min", self.min_height, self.min_width, self.min_aspect_ratio],
            ["max", self.max_height, self.max_width, self.max_aspect_ratio],
        ])
        print(table.draw())


class SegmentationMapPixelStats:
    def __init__(self, segmentation_map_int: np.ndarray) -> None:
        height = segmentation_map_int.shape[0]
        width = segmentation_map_int.shape[1]
        self.num_pixels: int = height*width
        self.class_num_pixels: Counter[SemanticClassId] = \
            Counter(segmentation_map_int.flatten())
        self.class_ids: List[int] = sorted(self.class_num_pixels.keys())
        self.num_classes: int = len(self.class_ids)

    def print_summary(
        self,
        class_mapping: Optional[SemanticClassMapping] = None,
    ) -> None:
        table = texttable.Texttable()
        if class_mapping is None:
            table.set_cols_align(["c", "c", "c"])  # l, c, r
            table.set_cols_valign(["c", "c", "c"])  # t, m, b
            #table.set_cols_dtype(["i", "i", "i", "f", "f"])  # t, f, e, i, a
            table.header([
                "class ID",
                "# pixels",
                "portion\nof pixels",
            ])
            for class_id in self.class_ids:
                num_class_pixels: int = self.class_num_pixels[class_id]
                table.add_row([
                    f"{class_id}",
                    f"{num_class_pixels}",
                    f"{num_class_pixels/self.num_pixels}",
                ])
        else:
            table.set_cols_align(["c", "c", "c", "c"])  # l, c, r
            table.set_cols_valign(["c", "c", "c", "c"])  # t, m, b
            #table.set_cols_dtype(["i", "i", "i", "f", "f"])  # t, f, e, i, a
            table.header([
                "class ID",
                "class name",
                "# pixels",
                "portion\nof pixels",
            ])
            for class_id in self.class_ids:
                num_class_pixels: int = self.class_num_pixels[class_id]
                table.add_row([
                    f"{class_id}",
                    f"{class_mapping.class_names[class_id]}",
                    f"{num_class_pixels}",
                    f"{num_class_pixels/self.num_pixels}",
                ])
        print(table.draw())


def segmentation_map_pixel_stats_from_file(
    segmentation_map_filename: str,
    segmentation_map_int_from_raw: \
        Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> SegmentationMapPixelStats:
    segmentation_map_int_raw: np.ndarray = \
        imageio.imread(segmentation_map_filename)
    if segmentation_map_int_from_raw is None:
        return SegmentationMapPixelStats(segmentation_map_int_raw)
    segmentation_map_int: np.ndarray = \
        segmentation_map_int_from_raw(segmentation_map_int_raw)
    return SegmentationMapPixelStats(segmentation_map_int)


class SemanticClassStats:
    def __init__(self, class_id: SemanticClassId) -> None:
        self.class_id: SemanticClassId = class_id
        self.num_pixels: int = 0
        self.num_images: int = 0
        self.portion_of_pixels: float = 0.0
        self.portion_of_images: float = 0.0


class SegmentationMapsJointPixelStats:
    def __init__(
        self,
        segmentation_map_filenames: List[str],
        segmentation_map_int_from_raw: \
            Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        num_workers_ubnd = max(1, 3*os.cpu_count()//4)
        segmentation_map_pixel_stats_from_file_: \
            Callable[[str], SegmentationMapPixelStats] = partial(
                segmentation_map_pixel_stats_from_file,
                segmentation_map_int_from_raw=segmentation_map_int_from_raw,
            )
        with ProcessPoolExecutor(max_workers=num_workers_ubnd) as executor:
            all_pixel_stats: List[SegmentationMapPixelStats] = \
                list(executor.map(
                    segmentation_map_pixel_stats_from_file_,
                    segmentation_map_filenames,
                ))
        assert len(all_pixel_stats) == len(segmentation_map_filenames)
        self.num_images: int = len(segmentation_map_filenames)
        self.num_pixels: int = 0
        self.class_stats: Dict[SemanticClassId, SemanticClassStats] = {}
        for pixel_stats in all_pixel_stats:
            self.num_pixels += pixel_stats.num_pixels
            for class_id, num_pixels in pixel_stats.class_num_pixels.items():
                if class_id not in self.class_stats:
                    self.class_stats[class_id] = SemanticClassStats(class_id)
                self.class_stats[class_id].num_pixels += num_pixels
                self.class_stats[class_id].num_images += 1
        for class_stats in self.class_stats.values():
            class_stats.portion_of_pixels = \
                class_stats.num_pixels/self.num_pixels
            class_stats.portion_of_images = \
                class_stats.num_images/self.num_images
        self.class_ids: List[int] = sorted(self.class_stats.keys())
        self.num_classes: int = len(self.class_ids)

    def get_class_frequencies(self) -> np.ndarray:
        frequencies: np.ndarray = np.zeros(self.num_classes, dtype=np.float64)
        for ix, class_id in enumerate(self.class_ids):
            frequencies[ix] = self.class_stats[class_id].portion_of_pixels
        return frequencies

    def print_summary(
        self,
        class_id_names: Optional[Dict[SemanticClassId, str]] = None,
    ) -> None:
        table = texttable.Texttable()
        if class_id_names is None:
            table.set_cols_align(["c", "c", "c", "c", "c"])  # l, c, r
            table.set_cols_valign(["c", "c", "c", "c", "c"])  # t, m, b
            #table.set_cols_dtype(["i", "i", "i", "f", "f"])  # t, f, e, i, a
            table.header([
                "class ID",
                "# pixels",
                "# images",
                "portion\nof pixels",
                "portion\nof images"
            ])
            for class_id in self.class_ids:
                table.add_row([
                    f"{class_id}",
                    f"{self.class_stats[class_id].num_pixels}",
                    f"{self.class_stats[class_id].num_images}",
                    f"{self.class_stats[class_id].portion_of_pixels}",
                    f"{self.class_stats[class_id].portion_of_images}",
                ])
        else:
            table.set_cols_align(["c", "c", "c", "c", "c", "c"])  # l, c, r
            table.set_cols_valign(["c", "c", "c", "c", "c", "c"])  # t, m, b
            #table.set_cols_dtype(["i", "i", "i", "f", "f"])  # t, f, e, i, a
            table.header([
                "class ID",
                "class name",
                "# pixels",
                "# images",
                "portion\nof pixels",
                "portion\nof images"
            ])
            for class_id in self.class_ids:
                table.add_row([
                    f"{class_id}",
                    f"{class_id_names[class_id]}",
                    f"{self.class_stats[class_id].num_pixels}",
                    f"{self.class_stats[class_id].num_images}",
                    f"{self.class_stats[class_id].portion_of_pixels}",
                    f"{self.class_stats[class_id].portion_of_images}",
                ])
        print(table.draw())
