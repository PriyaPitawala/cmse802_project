# Module for unit testing the image_with_scale_bar.py.

# This module contains functions for unit testing all the functions in the age_with_scale_bar.py.
# The scale bar size is verified by comparing the pixel length of an actual scale bar, and a debugging
# image is saved inside `tests/output`. Correctly detected scale bar should be outlined in red color.


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
            display_image(self.bgr_image, title="Test Image")
        except Exception as e:
            self.fail(f"display_image raised an exception: {e}")

class TestRealScaleBar(unittest.TestCase):

    def setUp(self):
        self.image_path = "tests/data/image_with_real_scale_bar.tif"
        if not os.path.exists(self.image_path):
            self.skipTest("Real scale bar image not found. Please add it to tests/data/")

        self.output_debug_path = "tests/output/debug_scale_bar_detection.png"
        os.makedirs(os.path.dirname(self.output_debug_path), exist_ok=True)

        self.expected_scale_length_pixels = 137  # Expected drawn scale bar length
        self.scale_region_fraction = 0.25        # Search bottom-right quarter

    def test_real_black_scale_bar_length(self):
        img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.fail("Image could not be loaded. Check format or path.")

        # Convert to grayscale if necessary
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        h, w = img_gray.shape
        cropped = img_gray[int(h * (1 - self.scale_region_fraction)):, int(w * (1 - self.scale_region_fraction)):]
        cropped_color = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

        # Step 1: Detect white background (scale label area)
        _, white_mask = cv2.threshold(cropped, 220, 255, cv2.THRESH_BINARY)
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not white_contours:
            self.fail("No white background detected.")

        # Largest white object = background of scale bar
        white_bg = max(white_contours, key=cv2.contourArea)
        x_bg, y_bg, w_bg, h_bg = cv2.boundingRect(white_bg)
        white_bg_length = max(w_bg, h_bg)
        print(f"Detected white background length: {white_bg_length} pixels")

        # Draw white background box
        cv2.rectangle(cropped_color, (x_bg, y_bg), (x_bg + w_bg, y_bg + h_bg), (0, 255, 255), 2)

        # Step 2: Focus only inside the white box
        white_bg_region = cropped[y_bg:y_bg + h_bg, x_bg:x_bg + w_bg]
        _, black_thresh = cv2.threshold(white_bg_region, 50, 255, cv2.THRESH_BINARY_INV)

        # Step 3: Detect black bar contours inside white box
        contours, _ = cv2.findContours(black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_lengths = []
        for contour in contours:
            x, y, w_bar, h_bar = cv2.boundingRect(contour)
            aspect_ratio = max(w_bar, h_bar) / (min(w_bar, h_bar) + 1e-5)
            length = max(w_bar, h_bar)
            print(f"  -> Inside white box: w={w_bar}, h={h_bar}, aspect_ratio={aspect_ratio:.2f}, length={length}")
            if aspect_ratio > 1.5:
                candidate_lengths.append((contour, length, (x, y, w_bar, h_bar)))

        if not candidate_lengths:
            self.fail("No black bar detected inside the white background region.")

        # Use the best candidate (longest contour)
        best_contour, actual_scale_length, (x_rel, y_rel, w_bar, h_bar) = max(candidate_lengths, key=lambda x: x[1])

        # Draw red box around the detected black scale bar on full cropped image
        x_abs = x_bg + x_rel
        y_abs = y_bg + y_rel
        cv2.rectangle(cropped_color, (x_abs, y_abs), (x_abs + w_bar, y_abs + h_bar), (0, 0, 255), 2)

        print(f"Detected black scale bar length (inside white background): {actual_scale_length} pixels")
        cv2.imwrite(self.output_debug_path, cropped_color)
        print(f"Debug image saved to: {self.output_debug_path}")

        # Check if it's close to expected
        tolerance = 5
        self.assertTrue(
            abs(actual_scale_length - self.expected_scale_length_pixels) <= tolerance,
            f"Scale bar pixel length ({actual_scale_length}) does not match expected ({self.expected_scale_length_pixels})"
        )

# Only one entry point at the end
if __name__ == "__main__":
    unittest.main()
