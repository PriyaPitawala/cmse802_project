"""
segmentation_utils.py

Provides a utility function to compute marker images for watershed
segmentation from a binary foreground mask. This module is essential
for separating merged or touching regions such as spherulites
in polarized optical microscopy (POM) images.

Main Function:
--------------
compute_markers(foreground_mask, ...):
    - Cleans the input mask with morphological operations.
    - Computes sure foreground and sure background regions.
    - Identifies unknown regions (boundary zones between grains).
    - Labels connected components and removes small regions.
    - Prepares the final marker image suitable for use with cv2.watershed.

Parameters:
-----------
- foreground_mask (np.ndarray): Binary image (0 background, 255 spherulites).
- morph_kernel_size (tuple): Size of structuring element
for morphological operations (default: (3, 3)).
- dilation_iter (int): Number of dilation iterations
to define sure background (default: 2).
- dist_transform_factor (float): Fraction of max distance transform used
for foreground thresholding (default: 0.3).
- min_foreground_area (int): Minimum area in pixels for a marker to be retained
(default: 50).

Returns:
--------
- np.ndarray: Integer marker image (int32) where:
    - Label 0 marks unknown region (to be filled by watershed).
    - Label 1 is reserved for background.
    - Labels 2 and above correspond to individual spherulites or segments.

Dependencies:
-------------
- OpenCV (cv2)
- NumPy (np)

Example:
--------
```python
from segmentation.segmentation_utils import compute_markers
markers = compute_markers(foreground_mask)
```

#Author: Priyangika Pitawala
#Date: April 2025
"""

import cv2
import numpy as np


def compute_markers(
    foreground_mask: np.ndarray,
    morph_kernel_size=(3, 3),
    dilation_iter=2,
    dist_transform_factor=0.3,
    min_foreground_area=50,
) -> np.ndarray:
    """
    Computes marker image for watershed segmentation based on cleaned foreground mask.

    Parameters:
    - foreground_mask (np.ndarray): Binary mask of foreground (spherulites), 0 or 255.
    - morph_kernel_size (tuple): Kernel size for morphological operations.
    - dilation_iter (int): Dilation iterations for sure background.
    - dist_transform_factor (float): Distance transform threshold factor
    for sure foreground.
    - min_foreground_area (int): Minimum area to retain a marker.

    Returns:
    - markers (np.ndarray): Marker image suitable for watershed (int32 format).
    """
    kernel = np.ones(morph_kernel_size, np.uint8)

    # Morphological opening to remove noise
    opening = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Sure background via dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=dilation_iter)

    # Distance transform for sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, dist_transform_factor * dist_transform.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)

    # Unknown = background - foreground
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Connected components on sure foreground
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # Filter small markers
    for label in range(1, num_labels):
        if np.sum(markers == label) < min_foreground_area:
            markers[markers == label] = 0

    # Increment all labels so background is 1 (not 0)
    markers += 1
    markers[unknown == 255] = 0  # Mark unknown as 0

    return markers
