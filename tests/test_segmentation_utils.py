"""
Unit tests for segmentation_utils.py using pytest.

Tests the watershed marker generation based on a cleaned foreground mask,
ensuring correct labeling behavior and marker filtering.

Test Scope:
-----------
1. compute_markers:
    - Produces integer labels (np.int32)
    - Returns nonzero markers only for sufficiently large foreground regions
    - Sets unknown regions to 0
    - Background starts from label 1

Run with:
---------
    pytest tests/test_segmentation_utils.py

#Author: Priyangika Pitawala
#Date: April 2025
"""

import numpy as np
import cv2
import os
import sys

from segmentation import segmentation_utils

# Add src/ to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def test_markers_output_type_and_shape():
    """Markers should match input shape and be int32"""
    mask = np.zeros((10, 10), dtype=np.uint8)
    cv2.rectangle(mask, (2, 2), (4, 4), 255, -1)

    markers = segmentation_utils.compute_markers(mask)
    assert markers.shape == mask.shape
    assert markers.dtype == np.int32


def test_markers_labels_are_sequential():
    """Only one labeled marker should appear if one foreground object"""
    mask = np.zeros((20, 20), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (14, 14), 255, -1)

    markers = segmentation_utils.compute_markers(mask)
    unique = np.unique(markers)
    assert set(unique) >= {0, 1, 2}  # 0=unknown, 1=background, 2=foreground marker


def test_small_object_is_filtered_out():
    """Small objects below min_foreground_area should not produce markers"""
    mask = np.zeros((10, 10), dtype=np.uint8)
    cv2.circle(mask, (5, 5), 1, 255, -1)  # Tiny circle

    markers = segmentation_utils.compute_markers(mask, min_foreground_area=20)
    assert np.all(markers <= 1)  # Only background (1), unknown (0)


def test_multiple_objects_create_multiple_markers():
    """Multiple large foreground objects should be labeled uniquely"""
    mask = np.zeros((30, 30), dtype=np.uint8)
    cv2.rectangle(mask, (2, 2), (8, 8), 255, -1)
    cv2.rectangle(mask, (20, 20), (26, 26), 255, -1)

    markers = segmentation_utils.compute_markers(mask, min_foreground_area=10)
    unique_labels = np.unique(markers)
    # We expect: 0 (unknown), 1 (background), 2 (obj1), 3 (obj2)
    assert set(unique_labels) >= {0, 1, 2, 3}
