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
