# Module for unit testing the preprocessing

# This module contains functions for unit testing all the functions in the preprocessing_old.py, which can
# sucessfully compute markers for segmenting Maltese crosses. The unit testing is also conducted on
# preprocessing.py which is undergoing debugging in order to segment spherulites.

# Author: Priyangika Pitawala
# Date: March 2025

import unittest
import numpy as np
import cv2
import os
import sys
import matplotlib

# Use a non-interactive backend for testing
matplotlib.use("Agg")

# Add src/ to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Import both preprocessing modules
from preprocessing import preprocessing as debug_preprocessing
from preprocessing import preprocessing_old


class TestPreprocessingModules(unittest.TestCase):

    def setUp(self):
        self.test_image_path = "tests/data/test_pom_image.tif"
        if not os.path.exists(self.test_image_path):
            self.skipTest(
                "Test image not found. Please place a test image at tests/data/test_pom_image.tif"
            )
        self.image = cv2.imread(self.test_image_path)
        self.assertIsNotNone(self.image, "Failed to load test image.")

    def test_preprocess_image_old_returns_binary(self):
        binary = preprocessing_old.preprocess_image(self.image)
        unique_vals = np.unique(binary)
        self.assertTrue(
            set(unique_vals).issubset({0, 255}),
            f"Old version output not binary: {unique_vals}",
        )
        self.assertEqual(binary.shape, self.image.shape[:2])

    def test_compute_markers_old_produces_labels(self):
        binary = preprocessing_old.preprocess_image(self.image)
        markers = preprocessing_old.compute_markers(binary)
        num_labels = len(np.unique(markers))
        self.assertGreater(
            num_labels, 1, "Old version should produce multiple marker regions"
        )

    def test_preprocess_image_debug_returns_binary(self):
        binary = debug_preprocessing.preprocess_image(self.image)
        unique_vals = np.unique(binary)
        self.assertTrue(
            set(unique_vals).issubset({0, 255}),
            f"Debug version output not binary: {unique_vals}",
        )
        self.assertEqual(binary.shape, self.image.shape[:2])

    def test_compute_markers_debug_produces_labels(self):
        binary = debug_preprocessing.preprocess_image(self.image)
        markers = debug_preprocessing.compute_markers(binary)
        num_labels = len(np.unique(markers))
        self.assertGreater(
            num_labels, 1, "Debug version should produce multiple marker regions"
        )


if __name__ == "__main__":
    unittest.main()
