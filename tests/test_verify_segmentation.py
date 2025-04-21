"""
Test module for verify_segmentation.py

This module contains pytest-based unit tests for validating the functionality of the
verify_segmentation module, which overlays various types of segmentation boundaries
on microscopy image masks.

Test coverage includes:
- overlay_boundaries_on_mask: Ensures watershed (-1) boundaries
are correctly overlaid in color
- overlay_combined_boundaries: Confirms correct combination of
watershed and edge-based overlays
- overlay_labels_as_boundaries: Validates proper boundary extraction
from labeled masks and rendering on grayscale images

Each test uses synthetic inputs (small NumPy arrays) to provide controlled conditions
and reproducible results.

#Author: Priyangika Pitawala
#Date: April 2025
"""
import numpy as np
import pytest
from segmentation.verify_segmentation import (
    overlay_boundaries_on_mask,
    overlay_combined_boundaries,
    overlay_labels_as_boundaries,
)


@pytest.fixture
def simple_foreground_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    return mask


@pytest.fixture
def simple_markers():
    markers = np.ones((10, 10), dtype=np.int32)
    markers[4:6, 4:6] = -1  # simulate boundary
    return markers


@pytest.fixture
def simple_edge_labels():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:5, 2:5] = 1
    labels[5:8, 5:8] = 2
    return labels


def test_overlay_boundaries_on_mask(simple_foreground_mask, simple_markers):
    result = overlay_boundaries_on_mask(simple_foreground_mask, simple_markers)
    assert result.shape == (10, 10, 3)
    assert np.any((result == [0, 255, 0]).all(axis=-1))  # green boundaries


def test_overlay_combined_boundaries(
    simple_foreground_mask, simple_markers, simple_edge_labels
):
    result = overlay_combined_boundaries(
        simple_foreground_mask, simple_markers, simple_edge_labels
    )
    assert result.shape == (10, 10, 3)
    has_green = np.any((result == [0, 255, 0]).all(axis=-1))  # watershed
    has_blue = np.any((result == [255, 0, 0]).all(axis=-1))  # edge
    assert has_green or has_blue


def test_overlay_labels_as_boundaries():
    base = np.full((10, 10), 120, dtype=np.uint8)
    label_mask = np.zeros((10, 10), dtype=np.int32)
    label_mask[3:7, 3:7] = 1
    result = overlay_labels_as_boundaries(base, label_mask)
    assert result.shape == (10, 10, 3)
    assert np.any((result == [0, 0, 255]).all(axis=-1))  # red boundaries
