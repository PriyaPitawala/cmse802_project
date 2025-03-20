# Module for watershed segmentation algorithm.

# This module contains  a function for segmenting an image using watershed algorith.
# The function intakes a raw image and markers for segmentation, and outputs a grayscale
# version of the raw image overlaid with the segmented boundaries. 

# Author: Priyangika Pitawala
# Date: March 2025

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
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel for overlay

    # Ensure markers are in int32 format as required by OpenCV
    markers = markers.astype(np.int32)

    # Apply watershed algorithm
    cv2.watershed(image, markers)
    
    # Create a thin boundary mask
    boundary_mask = markers == -1  # Boundary pixels
    gray_image[boundary_mask] = [0, 255, 0]  # Mark boundaries in green (BGR format: Blue=0, Green=255, Red=0)
    
    return gray_image
