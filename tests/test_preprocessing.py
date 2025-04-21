"""
Unit tests for preprocessing.py and preprocessing_old.py using pytest.

This test module validates the behavior of both legacy and current image preprocessing
pipelines used to segment semicrystalline polymer regions in POM images.

Tested Modules:
---------------
- preprocessing/preprocessing_old.py
- preprocessing/preprocessing.py

Test Coverage:
--------------
- Verifies binary output from both `preprocess_image()` functions
- Ensures watershed markers from `compute_markers()` are valid and nontrivial

Fixtures:
---------
- Uses a test image placed at: tests/data/test_pom_image.tif
  (Skip tests gracefully if image is not found)

Run with:
---------
    pytest tests/test_preprocessing.py

Dependencies:
-------------
- OpenCV
- NumPy
- Matplotlib (non-interactive mode)
- SciPy, scikit-image (via preprocessing module)

#Author: Priyangika Pitawala
#Date: April 2025
"""

import pytest
import numpy as np
import cv2
import os
import sys
import matplotlib

# Import both preprocessing modules
from preprocessing import preprocessing as debug_preprocessing
from preprocessing import preprocessing_old

# Use a non-interactive backend for testing
matplotlib.use("Agg")

# Add src/ to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


@pytest.fixture(scope="module")
def sample_image():
    path = "tests/data/test_pom_image.tif"
    if not os.path.exists(path):
        pytest.skip("Test image not found. Place it at tests/data/test_pom_image.tif")
    image = cv2.imread(path)
    if image is None:
        pytest.fail("Failed to load test image.")
    return image


def test_preprocess_image_old_returns_binary(sample_image):
    binary = preprocessing_old.preprocess_image(sample_image)
    unique_vals = np.unique(binary)
    assert set(unique_vals).issubset(
        {0, 255}
    ), f"Old version output not binary: {unique_vals}"
    assert binary.shape == sample_image.shape[:2]


def test_compute_markers_old_produces_labels(sample_image):
    binary = preprocessing_old.preprocess_image(sample_image)
    markers = preprocessing_old.compute_markers(binary)
    num_labels = len(np.unique(markers))
    assert num_labels > 1, "Old version should produce multiple marker regions"


def test_preprocess_image_debug_returns_binary(sample_image):
    binary = debug_preprocessing.preprocess_image(sample_image)
    unique_vals = np.unique(binary)
    assert set(unique_vals).issubset(
        {0, 255}
    ), f"Debug version output not binary: {unique_vals}"
    assert binary.shape == sample_image.shape[:2]


def test_compute_markers_debug_produces_labels(sample_image):
    binary = debug_preprocessing.preprocess_image(sample_image)
    markers = debug_preprocessing.compute_markers(binary)
    num_labels = len(np.unique(markers))
    assert num_labels > 1, "Debug version should produce multiple marker regions"
