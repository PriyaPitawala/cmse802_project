"""
Unit tests for image_loader.py

This module verifies the functionality of the `load_image` function which loads
image files into NumPy arrays using OpenCV. Tests include both valid and invalid
image path scenarios.

Test Cases:
-----------
- test_load_valid_image: Confirms successful image loading and expected format.
- test_load_invalid_image_raises: Ensures error is raised for invalid paths.

Author: Priyangika Pitawala
Date: April 2025
"""

import os
import sys
import unittest
import numpy as np
import cv2

from data_loading.image_loader import load_image


class TestImageLoader(unittest.TestCase):
    """
    Unit tests for the load_image function in image_loader.py
    """

    def setUp(self):
        """
        Creates a valid sample image file for testing and defines
        both valid and invalid image paths.
        """
        self.valid_image_path = os.path.join(
            os.path.dirname(__file__), "data", "sample_image.jpg"
        )
        self.invalid_image_path = "non_existent_path/image.jpg"

        # Create dummy image if not already present
        if not os.path.exists(self.valid_image_path):
            dummy_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
            os.makedirs(os.path.dirname(self.valid_image_path), exist_ok=True)
            cv2.imwrite(self.valid_image_path, dummy_img)

    def tearDown(self):
        """
        Removes the test image created in setUp() to ensure a clean test environment.
        """
        if os.path.exists(self.valid_image_path):
            os.remove(self.valid_image_path)

    def test_load_valid_image(self):
        """
        Tests that a valid image is loaded as a NumPy array with 3 color channels (BGR).
        """
        image = load_image(self.valid_image_path)
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape[2], 3)  # Check for 3 channels (color)

    def test_load_invalid_image_raises(self):
        """
        Tests that attempting to load an image from an invalid path
        raises FileNotFoundError.
        """
        with self.assertRaises(FileNotFoundError):
            load_image(self.invalid_image_path)


if __name__ == "__main__":
    # Ensure src is in path when running directly (optional safety)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    sys.path.append(project_root)

    unittest.main()
