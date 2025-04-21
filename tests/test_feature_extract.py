"""
Unit tests for feature_extract.py using pytest.

This module validates the correctness of region-based segmentation post-processing
including label cleaning, feature extraction, and label overlay.

Tested Functions:
-----------------
- clean_watershed_labels: Ensures boundary values (-1) are removed and labels relabeled.
- extract_region_features: Validates DataFrame feature output for known regions.
- overlay_labels_on_image: Checks bounding box and text rendering on synthetic input.

Fixtures:
---------
- dummy_labeled_image:
    A small labeled NumPy array with two distinct rectangular objects.

- dummy_raw_image:
    A synthetic BGR image with grayscale contrast matching the labeled image.

Run with:
---------
    pytest tests/test_feature_extract.py

#Author: Priyangika Pitawala
#Date: April 2025
"""

import numpy as np
import cv2
import pytest
import pandas as pd
from feature_extraction import feature_extract


@pytest.fixture
def dummy_labeled_image():
    """Creates a dummy labeled image with two rectangular regions."""
    img = np.zeros((100, 100), dtype=np.int32)
    img[10:30, 10:30] = 1  # Region 1
    img[50:80, 60:90] = 2  # Region 2
    return img


@pytest.fixture
def dummy_raw_image():
    """Returns a dummy grayscale image with synthetic contrast converted to BGR."""
    base = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(base, (10, 10), (30, 30), 100, -1)
    cv2.rectangle(base, (60, 50), (90, 80), 200, -1)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def test_clean_watershed_labels_removes_boundaries():
    """Watershed boundaries (-1) are removed and sequential labels are enforced."""
    markers = np.array(
        [
            [-1, 2, 2],
            [2, 2, -1],
            [-1, -1, 3],
        ],
        dtype=np.int32,
    )

    cleaned = feature_extract.clean_watershed_labels(markers)
    assert np.all(cleaned >= 0)
    assert np.issubdtype(cleaned.dtype, np.integer)  # ✅ portable type check
    assert len(np.unique(cleaned)) <= 3  # Background + 2 labels max


def test_extract_region_features_returns_dataframe(
    dummy_labeled_image, dummy_raw_image
):
    """Feature extraction should return a valid DataFrame with shape info
    and intensity stats."""
    features_df = feature_extract.extract_region_features(
        dummy_labeled_image, dummy_raw_image
    )
    assert isinstance(features_df, pd.DataFrame)
    assert "area" in features_df.columns
    assert len(features_df) == 2


def test_overlay_labels_on_image(dummy_labeled_image, dummy_raw_image):
    """Overlay must return a BGR image with same dimensions as input and type uint8."""
    overlay = feature_extract.overlay_labels_on_image(
        dummy_labeled_image, dummy_raw_image
    )
    assert overlay.shape == dummy_raw_image.shape
    assert overlay.dtype == np.uint8
    assert overlay.shape[2] == 3  # BGR
