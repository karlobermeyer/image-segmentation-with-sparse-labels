"""
Class mapping for semantic segmentation with the Pascal VOC dataset.
"""
# Standard
from typing import List, Tuple

# Scientific
import numpy as np

# Project-Specific
from core import (
    SemanticClassId,
    SemanticClassMapping,
)


class_ids_raw: np.ndarray = \
    np.linspace(start=0, stop=20, num=21, dtype=np.uint8)
class_ids_raw.flags.writeable = False
class_ids_raw_included: np.ndarray = class_ids_raw.copy()
class_ids_raw_included.flags.writeable = False

class_names_included: List[str] = [
    "background",  # 0
    "aeroplane",  # 1
    "bicycle",  # 2
    "bird",  # 3
    "boat",  # 4
    "bottle",  # 5
    "bus",  # 6
    "car",  # 7
    "cat",  # 8
    "chair",  # 9
    "cow",  # 10
    "dining_table",  # 11
    "dog",  # 12
    "horse",  # 13
    "motorbike",  # 14
    "person",  # 15
    "potted_plant",  # 16
    "sheep",  # 17
    "sofa",  # 18
    "train",  # 19
    "tv/monitor",  # 20
]

class_colors_rgb_included: List[Tuple[int, int, int]] = [
    (0, 0, 0),  # background
    (128, 0, 0),  # aeroplane
    (0, 128, 0),  # bicycle
    (128, 128, 0),  # bird
    (0, 0, 128),  # boat
    (128, 0, 128),  # bottle
    (0, 128, 128),  # bus
    (128, 128, 128),  # car
    (64, 0, 0),  # cat
    (192, 0, 0),  # chair
    (64, 128, 0),  # cow
    (192, 128, 0),  # dining_table
    (64, 0, 128),  # dog
    (192, 0, 128),  # horse
    (64, 128, 128),  # motorbike
    (192, 128, 128),  # person
    (0, 64, 0),  # potted_plant
    (128, 64, 0),  # sheep
    (0, 192, 0),  # sofa
    (128, 192, 0),  # train
    (0, 64, 128),  # tv/monitor
]
class_colors_rgb_included: np.ndarray = \
    np.array(class_colors_rgb_included, dtype=np.uint8)
class_colors_rgb_included.flags.writeable = False

class_mapping = SemanticClassMapping(
    class_ids_raw=class_ids_raw,
    class_ids_raw_included=class_ids_raw_included,
    class_names_included=class_names_included,
    class_colors_rgb_included=class_colors_rgb_included,
)
