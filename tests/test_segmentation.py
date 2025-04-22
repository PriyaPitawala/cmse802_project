"""
test_watershed_segmentation.py

Unit tests for the `apply_watershed` function in the watershed_segmentation module.

This test suite verifies:
- That the watershed algorithm executes correctly and returns a valid segmented image.
- That watershed boundaries (marked in green) are present in the output.
- That the function raises appropriate errors when given invalid input.

This module is intended to be run with pytest.

#Author: Priyangika Pitawala
#Date: April 2025
"""

import os
import numpy as np
import cv2
import pytest
import sys

from segmentation.watershed_segmentation import apply_watershed
from preprocessing import preprocessing_old

# Use non-interactive matplotlib backend
import matplotlib
matplotlib.use("Agg")

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


@pytest.fixture
def sample_image():
    """Fixture to load a test image or skip if not available."""
    path = "tests/data/test_pom_image.tif"
    if not os.path.exists(path):
        pytest.skip("Test image not found.")
    image = cv2.imread(path)
    if image is None:
        pytest.skip("Failed to load test image.")
    return image


def test_apply_watershed_runs(sample_image):
    """Test that apply_watershed returns a correctly shaped uint8 image."""
    binary = preprocessing_old.preprocess_image(sample_image)
    markers = preprocessing_old.compute_markers(binary)
    result_img, result_markers = apply_watershed(sample_image.copy(), markers)

    assert isinstance(result_img, np.ndarray)
    assert result_img.shape == sample_image.shape
    assert result_img.dtype == np.uint8


def test_apply_watershed_adds_boundaries(sample_image):
    """Test that green watershed boundaries are added to the result image."""
    binary = preprocessing_old.preprocess_image(sample_image)
    markers = preprocessing_old.compute_markers(binary)
    result_img, _ = apply_watershed(sample_image.copy(), markers)

    # Check for green boundaries in the image
    green_mask = np.all(result_img == [0, 255, 0], axis=-1)
    assert np.sum(green_mask) > 0, "Expected green boundary pixels"


@pytest.mark.parametrize("image, markers", [
    (None, np.zeros((10, 10), dtype=np.int32)),
    (np.zeros((10, 10, 3), dtype=np.uint8), None),
    (np.array(5), np.array(5)),
])
def test_apply_watershed_with_invalid_inputs(image, markers):
    """Test that invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        apply_watershed(image, markers)
