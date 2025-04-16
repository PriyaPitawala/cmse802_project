# Module for unit testing the image_loader.py.

# This module contains functions for unit testing all the functions in the image_loader.py.

# Author: Priyangika Pitawala
# Date: March 2025


import unittest
import numpy as np
import os
import sys

# Add src/ to the path for import resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Import the load_image function from the new location
from data_loading.image_loader import load_image


class TestImageLoader(unittest.TestCase):

    def setUp(self):
        # Set up a valid and an invalid path
        self.valid_image_path = os.path.join(
            os.path.dirname(__file__), "data", "sample_image.jpg"
        )
        self.invalid_image_path = "non_existent_path/image.jpg"

        # Create a small test image if it doesn't exist
        if not os.path.exists(self.valid_image_path):
            import cv2

            dummy_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
            os.makedirs(os.path.dirname(self.valid_image_path), exist_ok=True)
            cv2.imwrite(self.valid_image_path, dummy_img)

    def test_load_valid_image(self):
        image = load_image(self.valid_image_path)
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape[2], 3)  # Expect color image with 3 channels

    def test_load_invalid_image_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_image(self.invalid_image_path)


if __name__ == "__main__":
    unittest.main()
