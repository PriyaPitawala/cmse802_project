"""
image_loader.py

This module provides utility functions for loading images into NumPy arrays
using OpenCV.

Functionality:
--------------
- load_image(): Loads an image from disk as a NumPy array in BGR format.
  It validates the file path and raises a clear error if the image cannot be read.

Usage:
------
Used in the preprocessing and segmentation pipelines to feed raw POM images
into analysis workflows.

Author: Priyangika Pitawala
Date: April 2025
"""

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the given path using OpenCV and returns it as a NumPy array.

    Parameters:
    -----------
    image_path : str
        Path to the image file.

    Returns:
    --------
    np.ndarray
        Image loaded in OpenCV's default BGR format.

    Raises:
    -------
    FileNotFoundError
        If the image file cannot be loaded or does not exist.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    return image
