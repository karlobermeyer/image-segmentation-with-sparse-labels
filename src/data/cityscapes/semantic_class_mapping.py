"""
Class mapping for semantic segmentation with the Cityscapes dataset.

The raw Cityscapes data (see `labels.py` in [1]) has 34 classes with
*raw class IDs*
  [0, 1, 2, ..., 33].
The evaluator `cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling`,
which is also used by the Cityscapes test server, only uses the 19 classes with
raw IDs
  [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31,
   32, 33].
Therefore, this module assumes you want only to train and infer on this subset
of 19 classes. Using the cooked class ID 0 for the "unlabeled" class, 1 for 7, 2
for 8, etc., we obtain the full list of 20 *cooked class IDs*
  [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
   17, 18, 19].

_References_
[1] https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
[2] https://pypi.org/project/cityscapesScripts/
[3] https://stackoverflow.com/a/64242989
"""
# Standard
from typing import List, Tuple

# Scientific
import numpy as np

# Project-Specific
from core import (
    #SemanticClassId,
    SemanticClassMapping,
)


class_ids_raw: np.ndarray = \
    np.linspace(start=0, stop=33, num=34, dtype=np.int16)
class_ids_raw.flags.writeable = False
class_ids_raw_included: np.ndarray = np.array(
    (0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
     23, 24, 25, 26, 27, 28, 31, 32, 33),
    dtype=np.int16,
)
class_ids_raw_included.flags.writeable = False

class_names_included: List[str] = [
    "unlabeled",  # 0
    "road",  # 1
    "sidewalk",  # 2
    "building",  #3
    "wall",  # 4
    "fence",  # 5
    "pole",  # 6
    "traffic_light",  # 7
    "traffic_sign",  # 8
    "vegetation",  # 9
    "terrain",  # 10
    "sky",  # 11
    "person",  # 12
    "rider",  # 13
    "car",  # 14
    "truck",  # 15
    "bus",  # 16
    "train",  # 17
    "motorcycle",  # 18
    "bicycle",  # 19
]

class_colors_rgb_raw: List[Tuple[int, int, int]] = [
   (0, 0, 0),  # unlabeled
   (0, 0, 0),  # ego_vehicle
   (0, 0, 0),  # rectification_border
   (0, 0, 0),  # out_of_roi
   (0, 0, 0),  # static
   (111, 74, 0),  # dynamic
   (81, 0, 81),  # ground
   (128, 64, 128),  # road
   (244, 35, 232),  # sidewalk
   (250, 170, 160),  # parking
   (230, 150, 140),  # rail_track
   (70, 70, 70),  # building
   (102, 102, 156),  # wall
   (190, 153, 153),  # fence
   (180, 165, 180),  # guard_rail
   (150, 100, 100),  # bridge
   (150, 120, 90),  # tunnel
   (153, 153, 153),  # pole
   (153, 153, 153),  # polegroup
   (250, 170, 30),  # traffic_light
   (220, 220, 0),  # traffic_sign
   (107, 142, 35),  # vegetation
   (152, 251, 152),  # terrain
   (70, 130, 180),  # sky
   (220, 20, 60),  # person
   (255, 0, 0),  # rider
   (0, 0, 142),  # car
   (0, 0, 70),  # truck
   (0, 60, 100),  # bus
   (0, 0, 90),  # caravan
   (0, 0, 110),  # trailer
   (0, 80, 100),  # train
   (0, 0, 230),  # motorcycle
   (119, 11, 32),  # bicycle
   (0, 0, 142),  # license_plate
]
class_colors_rgb_included: List[Tuple[int, int, int]] = [(0, 0, 0)]
for class_id_raw in class_ids_raw_included[1:]:
    class_colors_rgb_included.append(class_colors_rgb_raw[class_id_raw])
class_colors_rgb_included: np.ndarray = np.array(
    class_colors_rgb_included,
    dtype=np.uint8,
)
class_colors_rgb_included.flags.writeable = False

class_mapping = SemanticClassMapping(
    class_ids_raw=class_ids_raw,
    class_ids_raw_included=class_ids_raw_included,
    class_names_included=class_names_included,
    class_colors_rgb_included=class_colors_rgb_included,
)
