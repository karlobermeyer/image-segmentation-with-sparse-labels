from typing import Dict, Tuple

import cv2

# https://albumentations.ai/docs/api_reference/full_reference/
import albumentations as albm  # Faster than `torchvision.transforms`.
from albumentations.pytorch import ToTensorV2


def preprocesses_from(
    input_height: int = 320,
    input_width: int = 640,
    mean_for_input_normalization: Tuple[int, int, int] = (0.485, 0.456, 0.406),
    std_for_input_normalization: Tuple[int, int, int] = (0.229, 0.224, 0.225),
    do_shift_scale_rotate: bool = True,
    ignore_index: int = 255,
) -> Dict[str, albm.Compose]:
    preprocesses: Dict[str, albm.Compose] = {}

    if do_shift_scale_rotate:
        preprocesses["train"] = albm.Compose([
            # `Resize` warps via bilinear interpolation.
            albm.Resize(height=input_height, width=input_width),
            albm.GaussNoise(p=0.2),
            albm.RandomBrightnessContrast(p=0.5),
            albm.HorizontalFlip(p=0.5),
            albm.ShiftScaleRotate(
                shift_limit=0.1,  # Usu. 0.05 - 0.1, not more than 0.3.
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,  # Image padding value.
                mask_value=ignore_index,  # Target padding value.
                p=0.5,
            ),
            albm.Normalize(
                mean=mean_for_input_normalization,
                std=std_for_input_normalization,
            ),
            ToTensorV2(),
        ])
    else:
        preprocesses["train"] = albm.Compose([
            # `Resize` warps via bilinear interpolation.
            albm.Resize(height=input_height, width=input_width),
            albm.GaussNoise(p=0.2),
            albm.RandomBrightnessContrast(p=0.5),
            albm.HorizontalFlip(p=0.5),
            albm.Normalize(
                mean=mean_for_input_normalization,
                std=std_for_input_normalization,
            ),
            ToTensorV2(),
        ])

    preprocesses["infer"] = albm.Compose([
        # `Resize` warps via bilinear interpolation.
        albm.Resize(height=input_height, width=input_width),
        albm.Normalize(
            mean=mean_for_input_normalization,
            std=std_for_input_normalization,
        ),
        ToTensorV2(),
    ])

    return preprocesses
