"""
Helper functions for manipulating Cityscapes image and target filenames.
"""


def shortened_image_filename_leaf(filename: str) -> str:
    """
    Construct image filename leaf with boilerplate removed.

    _Example_
    Input: ".../cityscapes/leftImg8bit/train/dusseldorf/dusseldorf_000030_000019_leftImg8bit.png"
    Output: "dusseldorf_000030_000019.png"
    """
    return filename.split("/")[-1][:-16] + ".png"


def shortened_target_filename_leaf(filename: str) -> str:
    """
    Construct target filename leaf with boilerplate removed.

    _Example_
    Input: ".../cityscapes/gtFine/train/dusseldorf/dusseldorf_000030_000019_gtFine_labelIds.png"
    Output: "dusseldorf_000030_000019.png"
    """
    return filename.split("/")[-1][:-20] + ".png"
