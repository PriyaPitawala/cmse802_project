# Module for unit testing the segmentation.

# This module contains functions for unit testing all the functions in the watershed_segmentation.py.
# It computes markers according to preprocessing_old.py.

# Author: Priyangika Pitawala
# Date: March 2025

import unittest
import numpy as np
import cv2
import os
import sys

# Set matplotlib to non-interactive mode if needed
import matplotlib
matplotlib.use('Agg')

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the watershed segmentation module
from segmentation.watershed_segmentation import apply_watershed
# Import marker generator from preprocessing_old (you can swap with preprocessing if needed)
from preprocessing import preprocessing_old

class TestWatershedSegmentation(unittest.TestCase):

    def setUp(self):
        self.test_image_path = "tests/data/test_pom_image.tif"
        if not os.path.exists(self.test_image_path):
            self.skipTest("Test image not found. Please place a test image at tests/data/test_pom_image.tif")
        self.image = cv2.imread(self.test_image_path)
        self.assertIsNotNone(self.image, "Failed to load test image.")

    def test_apply_watershed_runs(self):
        binary = preprocessing_old.preprocess_image(self.image)
        markers = preprocessing_old.compute_markers(binary)
        result = apply_watershed(self.image.copy(), markers)
        self.assertEqual(result.shape, self.image.shape, "Output shape should match original image")
        self.assertEqual(result.dtype, np.uint8, "Output image should be in uint8 format")

    def test_apply_watershed_adds_boundaries(self):
        binary = preprocessing_old.preprocess_image(self.image)
        markers = preprocessing_old.compute_markers(binary)
        result = apply_watershed(self.image.copy(), markers)
        # Count how many green pixels are in the result (boundary pixels)
        green_mask = np.all(result == [0, 255, 0], axis=-1)
        green_pixel_count = np.sum(green_mask)
        self.assertGreater(green_pixel_count, 0, "Watershed boundaries should be marked in green")

    def test_apply_watershed_with_invalid_inputs(self):
        with self.assertRaises(ValueError):
            apply_watershed(None, np.zeros((10, 10), dtype=np.int32))
        with self.assertRaises(ValueError):
            apply_watershed(np.zeros((10, 10, 3), dtype=np.uint8), None)
        with self.assertRaises(ValueError):
            apply_watershed(np.array(5), np.array(5))

if __name__ == '__main__':
    unittest.main()
