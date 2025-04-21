"""
Unit tests for edge_segmentation.py using pytest.

Tests the edge-free segmentation of spherulites using connected component labeling
on a binary foreground mask. This method does not rely on edge detection.

Test Scope:
-----------
1. segment_by_edges:
    - Correctly labels separate regions in binary mask
    - Handles empty masks (no foreground)
    - Handles masks with one solid region
    - Ignores the grayscale input

Run with:
---------
    pytest tests/test_edge_segmentation.py

#Author: Priyangika Pitawala
#Date: April 2025
"""

import numpy as np
import cv2
import os
import sys

from segmentation import edge_segmentation

# Add src/ to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def test_multiple_regions_labeled_correctly():
    foreground_mask = np.zeros((10, 10), dtype=np.uint8)
    cv2.rectangle(foreground_mask, (1, 1), (3, 3), 255, -1)
    cv2.rectangle(foreground_mask, (6, 6), (8, 8), 255, -1)
    gray_image = np.zeros_like(foreground_mask)

    labels = edge_segmentation.segment_by_edges(gray_image, foreground_mask)

    # Expect 2 labeled regions (not including background)
    unique = np.unique(labels)
    assert set(unique) == {0, 1, 2}


def test_empty_mask_returns_all_zero():
    foreground_mask = np.zeros((5, 5), dtype=np.uint8)
    gray_image = np.zeros_like(foreground_mask)

    labels = edge_segmentation.segment_by_edges(gray_image, foreground_mask)
    assert np.all(labels == 0)


def test_single_connected_region():
    foreground_mask = np.ones((4, 4), dtype=np.uint8) * 255
    gray_image = np.zeros_like(foreground_mask)

    labels = edge_segmentation.segment_by_edges(gray_image, foreground_mask)
    unique = np.unique(labels)
    assert set(unique) == {1}  # No background in input


def test_grayscale_input_does_not_affect_result():
    foreground_mask = np.zeros((5, 5), dtype=np.uint8)
    foreground_mask[2:4, 2:4] = 255

    random_gray = np.random.randint(0, 256, (5, 5), dtype=np.uint8)
    constant_gray = np.ones((5, 5), dtype=np.uint8) * 128

    labels_random = edge_segmentation.segment_by_edges(random_gray, foreground_mask)
    labels_constant = edge_segmentation.segment_by_edges(constant_gray, foreground_mask)

    # Since gray_image is unused, both should yield the same result
    assert np.array_equal(labels_random, labels_constant)
