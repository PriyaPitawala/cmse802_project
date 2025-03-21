# Module for unit testing the image_with_scale_bar.py.

# This module contains functions for unit testing all the functions in the age_with_scale_bar.py.

# Author: Priyangika Pitawala
# Date: March 2025


import unittest
import numpy as np
import cv2
import os
import sys
import matplotlib

# Use a non-interactive backend for testing display functions
matplotlib.use('Agg')

# Add src/ to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the functions to test
from visualization.image_with_scale_bar import convert_to_displayable, display_image

class TestImageScaleBar(unittest.TestCase):

    def setUp(self):
        # Create dummy test images
        self.gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        self.float_image = np.random.rand(100, 100).astype(np.float32)
        self.int_image = np.random.randint(0, 256, (100, 100), dtype=np.int32)
        self.bgr_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    def test_convert_gray_to_bgr_uint8(self):
        result = convert_to_displayable(self.gray_image)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape[2], 3)

    def test_convert_float_to_bgr_uint8(self):
        result = convert_to_displayable(self.float_image)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape[2], 3)

    def test_convert_int_to_bgr_uint8(self):
        result = convert_to_displayable(self.int_image)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape[2], 3)

    def test_convert_bgr_remains_unchanged(self):
        result = convert_to_displayable(self.bgr_image)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, self.bgr_image.shape)

    def test_display_image_runs(self):
        try:
            display_image(self.bgr_image, title="Test Image", scale_length_pixels=50, scale_text="20 µm")
        except Exception as e:
            self.fail(f"display_image raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()

class TestRealScaleBar(unittest.TestCase):

    def setUp(self):
        self.image_path = "tests/data/image_with_real_scale_bar.tif"
        if not os.path.exists(self.image_path):
            self.skipTest("Real scale bar image not found. Please add it to tests/data/")

        self.expected_scale_length_pixels = 130  # This is what display_image() uses
        self.scale_region_fraction = 0.25  # Only look at bottom-right corner

    def test_real_scale_bar_length(self):
        img = cv2.imread(self.image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img_gray.shape
        cropped = img_gray[int(h*(1 - self.scale_region_fraction)):, int(w*(1 - self.scale_region_fraction)):]

        # Apply thresholding to isolate the scale bar
        _, thresh = cv2.threshold(cropped, 220, 255, cv2.THRESH_BINARY)  # works well for white bar
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.fail("No contours detected — scale bar may not be visible or threshold may be incorrect.")

        # Find the longest contour (assume it's the scale bar)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_bar, h_bar = cv2.boundingRect(largest_contour)

        actual_scale_length = max(w_bar, h_bar)  # Use max in case it’s vertical

        print(f"\n📏 Detected actual scale bar length: {actual_scale_length} pixels")

        # Compare to expected length from your display_image()
        tolerance = 5  # pixels
        self.assertTrue(
            abs(actual_scale_length - self.expected_scale_length_pixels) <= tolerance,
            f"Scale bar pixel length ({actual_scale_length}) does not match expected ({self.expected_scale_length_pixels})"
        )

if __name__ == "__main__":
    unittest.main()