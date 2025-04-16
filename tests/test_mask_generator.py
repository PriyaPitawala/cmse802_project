"""
test_mask_generator.py

Unit tests for preprocessing grayscale microscopy images and generating binary masks
for foreground (spherulites), background (Maltese crosses), and removed regions.

Main Functionalities Tested:
----------------------------
- Grayscale conversion and CLAHE contrast enhancement
- Foreground and background binary mask generation
- Removal of small objects based on physical size (in microns)
- Overlay of removed regions on grayscale image

Dependencies:
-------------
- Requires calibration constants from `scale_bar_config.py`

Author: Priyangika Pitawala
Date: April 2025
"""

import numpy as np
import cv2
import pytest

from preprocessing import mask_generator


@pytest.fixture
def synthetic_image():
    """
    Create a synthetic BGR image with bright and dark features.

    Returns
    -------
    np.ndarray
        Test image with distinct regions for foreground and background simulation.
    """
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (40, 40), (255, 255, 255), -1)  # Bright region
    cv2.rectangle(img, (60, 60), (90, 90), (50, 50, 50), -1)  # Dark region
    return img


def test_preprocess_image_with_contrast(synthetic_image):
    """
    Test grayscale conversion and CLAHE enhancement.

    Asserts
    -------
    - Output shape and dtype match input expectations.
    - CLAHE modifies the grayscale image.
    """
    gray, original = mask_generator.preprocess_image(
        synthetic_image, enhance_contrast=True
    )
    assert gray.shape == original.shape
    assert gray.dtype == np.uint8
    assert not np.array_equal(gray, original), "CLAHE should alter pixel values"


def test_preprocess_image_without_contrast(synthetic_image):
    """
    Test grayscale conversion without applying CLAHE.

    Asserts
    -------
    - Output is identical to standard grayscale conversion.
    """
    gray, original = mask_generator.preprocess_image(
        synthetic_image, enhance_contrast=False
    )
    assert np.array_equal(
        gray, original
    ), "Image should remain unchanged if CLAHE is disabled"


def test_compute_masks_output_shapes():
    """
    Test generation of foreground, background, and removed masks.

    Asserts
    -------
    - All masks match the input shape and type.
    """
    gray_image = np.zeros((50, 50), dtype=np.uint8)
    gray_image[5:20, 5:20] = 200  # Foreground
    gray_image[30:45, 30:45] = 40  # Background

    bg_mask, fg_mask, removed = mask_generator.compute_foreground_background_masks(
        gray_image, background_thresh=100, min_object_length_um=1.0
    )

    for mask in [bg_mask, fg_mask, removed]:
        assert mask.shape == gray_image.shape
        assert mask.dtype == np.uint8


def test_remove_small_objects_by_size_removes_expected():
    """
    Test filtering of small objects based on physical size.

    Asserts
    -------
    - Large object is retained and small object is removed.
    - Combined mask of filtered and removed regions matches the original.
    """
    binary = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(binary, (10, 10), (30, 30), 1, -1)  # Large
    cv2.rectangle(binary, (60, 60), (61, 61), 1, -1)  # Small

    filtered, removed = mask_generator.remove_small_objects_by_size(
        binary, min_object_length_um=2.0, return_removed_mask=True
    )

    assert np.count_nonzero(filtered) > 0
    assert np.count_nonzero(removed) > 0
    assert np.array_equal(cv2.bitwise_or(filtered, removed), binary)


def test_overlay_removed_regions_dimensions():
    """
    Test visual overlay of removed region contours on grayscale image.

    Asserts
    -------
    - Output is a 3-channel color image.
    - Removed regions are color-coded as specified.
    """
    gray = np.zeros((50, 50), dtype=np.uint8)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[10:20, 10:20] = 1

    overlay = mask_generator.overlay_removed_regions(gray, mask, color=(0, 255, 0))
    assert overlay.shape == (50, 50, 3)
    assert overlay.dtype == np.uint8
    assert np.any(
        np.all(overlay == [0, 255, 0], axis=-1)
    ), "Overlay should contain specified color"
