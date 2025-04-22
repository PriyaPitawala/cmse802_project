"""
watershed_segmentation.py

This module provides functionality to apply watershed segmentation to labeled images
using OpenCV. It overlays the segmentation boundaries on the original image and
returns both the visual result and the updated marker labels.

Main Function:
- apply_watershed(image, markers): Applies watershed segmentation to an input image
  using a set of labeled markers and returns the result with green boundaries overlaid.

Usage:
This function is typically used after generating foreground/background markers using
distance transforms or other segmentation techniques. It is suitable for visualizing
spherulite boundaries or segmenting grain-like regions in microscopy images.

Example:
    result_image, updated_markers = apply_watershed(input_image, marker_mask)

Requirements:
- OpenCV (cv2)
- NumPy

Returns:
- A BGR image with segmentation boundaries overlaid in green.
- The updated marker array (int32) with watershed results.

Raises:
- ValueError: If the input image or marker array is invalid.

#Author: Priyangika Pitawala
#Date: April 2025
"""

import cv2
import numpy as np


def apply_watershed(image: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """
    Applies the watershed algorithm to segment regions in the given image.

    Parameters:
    - image (np.ndarray): Original color image (BGR format).
    - markers (np.ndarray): Marker image generated from compute_markers().

    Returns:
    - np.ndarray: Image with watershed segmentation boundary overlaid in green.
    """
    if image is None or markers is None:
        raise ValueError("Input image and markers must be valid numpy arrays.")

    if len(image.shape) < 2 or len(markers.shape) < 2:
        raise ValueError("Invalid image or marker dimensions.")

    # Convert image to grayscale for display
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(
        gray_image, cv2.COLOR_GRAY2BGR
    )  # Convert back to 3-channel for overlay

    # Ensure markers are in int32 format as required by OpenCV
    markers = markers.astype(np.int32)

    # Apply watershed algorithm
    cv2.watershed(image, markers)

    # Create a thin boundary mask
    boundary_mask = markers == -1  # Boundary pixels
    gray_image[boundary_mask] = [
        0,
        255,
        0,
    ]  # Mark boundaries in green (BGR format: Blue=0, Green=255, Red=0)

    return gray_image, markers
