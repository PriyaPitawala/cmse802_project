"""
split_merged_spherulites.py

This module provides a function to refine edge-based spherulite segmentations
by applying watershed splitting within large connected regions.
It detects internal intensity features using gradient and distance-based markers,
and applies local watershed segmentation to separate potentially merged grains.

Function:
---------
split_merged_spherulites(gray_image, edge_labels, ...):
    - Iterates over labeled regions in the edge_labels mask.
    - Applies gradient magnitude computation within each region.
    - Extracts seeds using distance transform and peak_local_max.
    - Applies cv2.watershed() locally within merged regions.
    - Updates and returns a new label map with refined boundaries.

Parameters:
-----------
- gray_image (np.ndarray): CLAHE-enhanced grayscale input image.
- edge_labels (np.ndarray): Initial labeled mask from edge-based segmentation.
- area_threshold (int): Minimum area (in pixels) to consider a region
as potentially merged.
- grad_blur_ksize (int): Sobel kernel size for computing image gradients (default: 3).
- dist_thresh_factor (float): Distance threshold factor for foreground
seed extraction (default: 0.5).

Returns:
--------
- refined_labels (np.ndarray): Updated labeled image after splitting merged regions.

Notes:
------
- This method is designed for splitting spherulites that appear fused
in binary segmentation
  masks but exhibit distinct internal texture patterns.
- Markers are generated using both morphological and gradient cues.
- The watershed is only applied locally to improve performance
and avoid oversegmentation.

Example:
--------
```python
from segmentation.split_merged_spherulites import split_merged_spherulites
refined = split_merged_spherulites(gray_image, edge_labels)
```

#Author: Priyangika Pitawala
#Date: April 2025
"""

import cv2
import numpy as np
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


def split_merged_spherulites(
    gray_image: np.ndarray,
    edge_labels: np.ndarray,
    area_threshold: int = 5000,
    grad_blur_ksize: int = 3,
    dist_thresh_factor: float = 0.5,
) -> np.ndarray:
    """
    Splits large spherulite regions using gradient-based watershed segmentation.

    Parameters:
    - gray_image (np.ndarray): CLAHE-enhanced grayscale image.
    - edge_labels (np.ndarray): Labeled regions from edge-based segmentation.
    - area_threshold (int): Minimum area to consider a region as potentially merged.
    - grad_blur_ksize (int): Kernel size for Sobel gradient.
    - dist_thresh_factor (float): Distance threshold factor for seed region extraction.

    Returns:
    - refined_labels (np.ndarray): Updated label map with merged regions split.
    """
    refined_labels = edge_labels.copy()
    current_max_label = np.max(refined_labels)

    for label_id in np.unique(edge_labels):
        if label_id == 0:
            continue  # skip background

        mask = (edge_labels == label_id).astype(np.uint8)
        region_area = np.sum(mask)

        if region_area < area_threshold:
            continue  # skip small regions

        # Extract region-specific grayscale
        region_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)

        # Gradient magnitude
        grad_x = cv2.Sobel(region_gray, cv2.CV_64F, 1, 0, ksize=grad_blur_ksize)
        grad_y = cv2.Sobel(region_gray, cv2.CV_64F, 0, 1, ksize=grad_blur_ksize)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_mag = cv2.convertScaleAbs(grad_mag)
        grad_mag = cv2.bitwise_and(grad_mag, grad_mag, mask=mask)

        # Distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # Use peak_local_max for marker seeds
        coordinates = peak_local_max(
            dist, labels=mask, min_distance=10, exclude_border=False
        )
        local_maxi = np.zeros_like(dist, dtype=bool)
        local_maxi[tuple(coordinates.T)] = True

        # Label the seed points
        markers = ndi.label(local_maxi)[0]

        # Watershed markers
        markers += 1

        # Convert grayscale to BGR for watershed
        region_bgr = cv2.cvtColor(region_gray, cv2.COLOR_GRAY2BGR)
        cv2.watershed(region_bgr, markers)

        # Zero out original region first
        refined_labels[edge_labels == label_id] = 0

        # Update with new labels after clearing old one
        for m_id in np.unique(markers):
            if m_id <= 1:
                continue
            refined_labels[(markers == m_id)] = current_max_label + 1
            current_max_label += 1

    return refined_labels
