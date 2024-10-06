import os
import pytest

from models.image_segmenter_hparams import (
    ImageSegmenterHparams,
    image_segmenter_hparams_from_yaml,
)


PROJECT_ROOT: str = os.environ.get("PROJECT_ROOT")
assert PROJECT_ROOT is not None, \
    "PROJECT_ROOT not found! Did you run `setenv.sh`?"


def test_image_segmenter_hparams():
    hparams_filename: str = os.path.join(
        PROJECT_ROOT,
        "src/models/test/image_segmenter_hparams_example.yaml",
    )
    try:
        hparams: ImageSegmenterHparams = \
            image_segmenter_hparams_from_yaml(hparams_filename)
        assert hparams is not None
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")
