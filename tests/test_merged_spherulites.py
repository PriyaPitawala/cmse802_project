"""
Unit tests for split_merged_spherulites.py using pytest.

This module validates that the watershed-based splitting function correctly
identifies and splits large, merged regions based on gradient and distance transform.

Test Scope:
-----------
1. Handles small non-merged regions without altering them.
2. Splits large, artificially merged regions.
3. Preserves original label format and returns valid output.
4. Leaves background untouched.

Run with:
---------
    pytest tests/test_split_merged_spherulites.py
"""

import numpy as np
import cv2
import os
import sys

from segmentation import split_merged_spherulites

# Add src/ to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def test_preserve_small_regions():
    """Small regions should not be split."""
    gray = np.full((100, 100), 150, dtype=np.uint8)
    mask = np.zeros_like(gray, dtype=np.int32)
    cv2.circle(mask, (30, 30), 5, 1, -1)
    cv2.circle(mask, (70, 70), 5, 2, -1)

    result = split_merged_spherulites.split_merged_spherulites(
        gray, mask, area_threshold=1000
    )
    assert set(np.unique(result)) >= {0, 1, 2}


def test_split_large_merged_region():
    """Watershed should split large, artificially merged regions
    into multiple labels."""
    # Large canvas to avoid overlap artifacts
    size = 200
    gray = np.zeros((size, size), dtype=np.uint8)

    # Two bright circular blobs with a slight valley between
    cv2.circle(gray, (60, 100), 30, 220, -1)  # left
    cv2.circle(gray, (140, 100), 30, 200, -1)  # right

    # Single label combining both
    mask = np.zeros_like(gray, dtype=np.int32)
    cv2.circle(mask, (60, 100), 30, 1, -1)
    cv2.circle(mask, (140, 100), 30, 1, -1)

    # Run split
    result = split_merged_spherulites.split_merged_spherulites(
        gray,
        mask,
        area_threshold=500,  # definitely large enough to split
        dist_thresh_factor=0.5,  # default okay
    )

    labels = np.unique(result)
    regions = labels[labels > 0]
    assert len(regions) > 1, f"Expected >1 region, got: {regions}"


def test_background_is_zero():
    """Ensure background stays labeled as 0."""
    gray = np.full((50, 50), 100, dtype=np.uint8)
    mask = np.zeros_like(gray, dtype=np.int32)
    cv2.rectangle(mask, (10, 10), (40, 40), 1, -1)

    result = split_merged_spherulites.split_merged_spherulites(gray, mask)
    assert 0 in np.unique(result)


def test_output_type_and_shape():
    """Check that output has correct shape and type."""
    gray = np.full((60, 60), 120, dtype=np.uint8)
    mask = np.zeros_like(gray, dtype=np.int32)
    cv2.circle(mask, (30, 30), 20, 1, -1)

    result = split_merged_spherulites.split_merged_spherulites(gray, mask)
    assert result.shape == mask.shape
    assert result.dtype == np.int32
