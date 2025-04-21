"""
Unit tests for preprocess_malcross.py using pytest.

This module verifies the image preprocessing pipeline for identifying Maltese crosses
in polarized optical microscopy (POM) images. It checks the behavior of the
preprocessing function under different thresholding and contrast enhancement settings.

Tested Functions:
-----------------
- preprocess_image: Converts BGR image to grayscale, enhances contrast, applies blur,
  and generates a binary mask using adaptive or global thresholding.

Fixtures:
---------
- synthetic_maltese_cross_image: A dummy image with a cross pattern designed to simulate
  a Maltese cross-like nucleation center.

Run with:
---------
    pytest tests/test_preprocess_malcross.py

#Author: Priyangika Pitawala
#Date: April 2025
"""

import numpy as np
import cv2
import pytest
from preprocessing import preprocess_malcross


@pytest.fixture
def synthetic_maltese_cross_image():
    """Creates a dummy image with a bright cross-like region in the center."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.line(img, (50, 10), (50, 90), (255, 255, 255), 3)
    cv2.line(img, (10, 50), (90, 50), (255, 255, 255), 3)
    return img


def test_preprocess_default_params(synthetic_maltese_cross_image):
    """Test default behavior of preprocess_image with adaptive thresholding
    and contrast enhancement."""
    mask = preprocess_malcross.preprocess_image(synthetic_maltese_cross_image)
    assert isinstance(mask, np.ndarray)
    assert mask.shape[:2] == synthetic_maltese_cross_image.shape[:2]
    assert mask.dtype == np.uint8
    assert np.any(mask == 255)


def test_preprocess_global_thresholding(synthetic_maltese_cross_image):
    """Test global thresholding mode with contrast enhancement off."""
    mask = preprocess_malcross.preprocess_image(
        synthetic_maltese_cross_image,
        adaptive=False,
        enhance_contrast=False,
        threshold_value=100,
    )
    assert mask.shape == synthetic_maltese_cross_image.shape[:2]
    assert np.unique(mask).tolist() in [[0], [0, 255], [255]]


def test_preprocess_different_blur(synthetic_maltese_cross_image):
    """Test that using a different blur kernel does not crash and still
    outputs a valid binary mask."""
    mask = preprocess_malcross.preprocess_image(
        synthetic_maltese_cross_image,
        blur_kernel=(9, 9),
    )
    assert mask.shape == synthetic_maltese_cross_image.shape[:2]
    assert np.any(mask == 255)
