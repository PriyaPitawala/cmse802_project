"""
Unit tests for image_loader.py using pytest.

Tests:
------
- Loading a valid image returns a NumPy array with 3 channels
- Invalid path raises FileNotFoundError

Requires: OpenCV, pytest

Author: Priyangika Pitawala
Date: April 2025
"""

import numpy as np
import pytest
import cv2

from data_loading.image_loader import load_image


@pytest.fixture(scope="module")
def valid_image_path(tmp_path_factory):
    """
    Creates a temporary 50x50 white image for testing.
    Returns the file path.
    """
    test_dir = tmp_path_factory.mktemp("test_images")
    img_path = test_dir / "sample_image.jpg"
    dummy_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(img_path), dummy_img)
    return str(img_path)


@pytest.fixture
def invalid_image_path():
    """Returns a guaranteed invalid image path."""
    return "non_existent_directory/invalid_image.jpg"


def test_load_valid_image(valid_image_path):
    """Test that a valid image loads correctly as a BGR NumPy array."""
    image = load_image(valid_image_path)
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8


def test_load_invalid_image_raises(invalid_image_path):
    """Test that loading from an invalid path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_image(invalid_image_path)
