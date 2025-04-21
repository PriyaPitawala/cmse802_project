"""
Unit tests for image_with_scale_bar.py and real-scale-bar overlay validation.

Tests:
------
1. Synthetic images:
    - Validate conversion of grayscale, float, and int images to displayable BGR format
    - Ensure display_image runs without exception
    (uses non-interactive matplotlib backend).

2. Real image analysis:
    - Detect white background label and black scale bar from annotated image
    - Compare detected scale bar length to expected pixel value (137 px)

Notes:
------
- Requires OpenCV and matplotlib
- Place a real annotated image in: tests/data/image_with_real_scale_bar.tif

Author: Priyangika Pitawala
Date: April 2025
"""

import os
import numpy as np
import pytest
import cv2
import matplotlib

from visualization.image_with_scale_bar import convert_to_displayable, display_image

matplotlib.use("Agg")  # Set backend after all imports


# --------------------------
# Synthetic Image Conversion
# --------------------------


@pytest.fixture
def synthetic_images():
    """Returns a dict of grayscale, float, int, and BGR test images."""
    return {
        "gray": np.random.randint(0, 256, (100, 100), dtype=np.uint8),
        "float": np.random.rand(100, 100).astype(np.float32),
        "int": np.random.randint(0, 256, (100, 100), dtype=np.int32),
        "bgr": np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
    }


def test_convert_gray_to_bgr(synthetic_images):
    """Grayscale uint8 image should convert to 3-channel uint8 BGR."""
    img = synthetic_images["gray"]
    result = convert_to_displayable(img)
    assert result.dtype == np.uint8
    assert result.ndim == 3 and result.shape[2] == 3


def test_convert_float_to_bgr(synthetic_images):
    """Float32 image should be normalized to 8-bit and converted to BGR."""
    img = synthetic_images["float"]
    result = convert_to_displayable(img)
    assert result.dtype == np.uint8
    assert result.ndim == 3 and result.shape[2] == 3


def test_convert_int_to_bgr(synthetic_images):
    """int32 image should be cast to uint8 and converted to BGR."""
    img = synthetic_images["int"]
    result = convert_to_displayable(img)
    assert result.dtype == np.uint8
    assert result.ndim == 3 and result.shape[2] == 3


def test_convert_bgr_unchanged(synthetic_images):
    """BGR input image should remain unchanged."""
    img = synthetic_images["bgr"]
    result = convert_to_displayable(img)
    assert np.array_equal(result, img)


def test_display_image_runs(synthetic_images):
    """Ensure display_image() executes without raising exceptions."""
    try:
        display_image(synthetic_images["bgr"], title="Test Image")
    except Exception as e:
        pytest.fail(f"display_image raised an exception: {e}")


# -------------------------------
# Real Image Scale Bar Validation
# -------------------------------


@pytest.mark.skipif(
    not os.path.exists("tests/data/image_with_real_scale_bar.tif"),
    reason="Real annotated scale bar image not found.",
)
def test_detect_real_scale_bar(tmp_path):
    """
    Validates scale bar length by detecting black bar inside white label region.
    Expects the scale bar to be 137 ± 5 pixels long.
    """
    img_path = "tests/data/image_with_real_scale_bar.tif"
    output_debug = tmp_path / "debug_scale_bar_detection.png"
    expected_length = 137
    tolerance = 5
    region_fraction = 0.25  # Bottom-right 25% region

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    assert img is not None, "Failed to load real image."

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = img_gray.shape
    y_start = int(h * (1 - region_fraction))
    x_start = int(w * (1 - region_fraction))
    cropped = img_gray[y_start:, x_start:]

    cropped_color = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    # Detect white background
    _, white_mask = cv2.threshold(cropped, 220, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    assert contours, "No white background detected."

    white_bg = max(contours, key=cv2.contourArea)
    x_bg, y_bg, w_bg, h_bg = cv2.boundingRect(white_bg)
    white_bg_region = cropped[y_bg : y_bg + h_bg, x_bg : x_bg + w_bg]

    # Detect black bar inside white region
    _, black_thresh = cv2.threshold(white_bg_region, 50, 255, cv2.THRESH_BINARY_INV)
    bar_contours, _ = cv2.findContours(
        black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for cnt in bar_contours:
        x, y, w_bar, h_bar = cv2.boundingRect(cnt)
        aspect_ratio = max(w_bar, h_bar) / (min(w_bar, h_bar) + 1e-5)
        if aspect_ratio > 1.5:
            length = max(w_bar, h_bar)
            candidates.append(length)

    assert candidates, "No black bar found in white background region."

    actual_length = max(candidates)
    assert abs(actual_length - expected_length) <= tolerance, (
        f"Detected scale bar length ({actual_length}px) "
        f"differs from expected ({expected_length}px ± {tolerance})"
    )

    # Save debug overlay
    cv2.rectangle(
        cropped_color,
        (x_bg, y_bg),
        (x_bg + w_bg, y_bg + h_bg),
        (255, 255, 0),
        2,
    )
    cv2.imwrite(str(output_debug), cropped_color)
    print(f"[DEBUG] Output saved to: {output_debug}")
