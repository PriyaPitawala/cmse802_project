# Module for edge-based segmentation.

# This module contains a function for processing a grayscale image and generating a binary image of the foreground,and 
# detecting the boundaries between the white regions surrounded by the black environment. 
# This is an edge-based segmentation method.

# Author: Priyangika Pitawala
# Date: March 2025

import cv2
import numpy as np

def segment_by_edges(gray_image: np.ndarray,
                     foreground_mask: np.ndarray) -> np.ndarray:
    """
    Segments spherulites by labeling connected white regions in the foreground mask.
    Avoids edge detection entirely and treats each white region as a separate spherulite.

    Parameters:
    - gray_image (np.ndarray): CLAHE-enhanced grayscale image (not used here but retained for compatibility).
    - foreground_mask (np.ndarray): Binary mask of known spherulites (0 or 255).

    Returns:
    - labels (np.ndarray): Labeled mask where each connected region gets a unique ID.
    """
    # Ensure mask is binary 0 and 255
    binary_mask = (foreground_mask > 0).astype(np.uint8)

    # Label connected white regions
    _, labels = cv2.connectedComponents(binary_mask)

    return labels
