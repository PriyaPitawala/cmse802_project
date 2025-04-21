"""
preprocessing_old.py

Legacy preprocessing module for preparing polarized optical microscopy (POM) images
for segmentation of semicrystalline polymer structures such as spherulites and
nucleation centers (e.g., Maltese crosses).

This module provides flexible image preprocessing and watershed marker generation
with extensive options for thresholding, contrast enhancement, edge detection, and
grain boundary identification.

Main Functionalities:
---------------------
1. preprocess_image:
   - Converts an input BGR image to grayscale
   - Optionally enhances local contrast using CLAHE
   - Applies Gaussian blur to reduce noise
   - Supports adaptive and global thresholding methods
   - Optionally integrates:
     - Canny edge detection
     - Gradient-based thresholding using the Laplacian
     - Morphological gradient for grain boundary detection

2. compute_markers:
   - Prepares marker seeds for watershed segmentation
   - Performs morphological filtering and distance transforms
   - Identifies sure foreground and background regions
   - Excludes regions below a minimum area threshold
   - Labels connected components for watershed input

Intended Use:
-------------
Primarily used for legacy experiments or comparison studies where
earlier versions of segmentation pipelines are required for benchmarking
against improved modules like `mask_generator.py`.

Typical Workflow:
-----------------
```python
from preprocessing import preprocessing_old

# Step 1: Preprocess POM image with optional features
binary_mask = preprocessing_old.preprocess_image(
    image=raw_image,
    enhance_contrast=True,
    use_edge_detection=True,
    detect_grain_boundaries=True
)

# Step 2: Generate markers for watershed segmentation
markers = preprocessing_old.compute_markers(binary_mask)
```

Dependencies:
-------------
- OpenCV (cv2)
- NumPy (np)

Returns:
--------
- Preprocessed binary masks (np.ndarray)
- Marker labels for watershed segmentation (np.ndarray)

#Author: Priyangika Pitawala
#Date: April 2025
"""

import cv2
import numpy as np


def preprocess_image(
    image: np.ndarray,
    blur_kernel=(5, 5),
    threshold_method=cv2.THRESH_BINARY,
    threshold_value=127,
    adaptive=False,
    block_size=11,
    C=2,
    use_edge_detection=False,
    edge_low_threshold=50,
    edge_high_threshold=150,
    use_gradient_thresholding=False,
    enhance_contrast=False,
    detect_grain_boundaries=False,
):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for local contrast enhancement
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray_image = clahe.apply(gray_image)

    # Apply Gaussian Blur
    image_blur = cv2.GaussianBlur(gray_image, blur_kernel, 0)

    # Apply thresholding
    if adaptive:
        processed = cv2.adaptiveThreshold(
            image_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_method,
            block_size,
            C,
        )
    else:
        _, processed = cv2.threshold(image_blur, threshold_value, 255, threshold_method)

    # Apply edge detection if enabled
    if use_edge_detection:
        edges = cv2.Canny(processed, edge_low_threshold, edge_high_threshold)
        processed = cv2.bitwise_or(
            processed, edges
        )  # Merge edges into thresholded image

    # Apply gradient-based thresholding if enabled
    if use_gradient_thresholding:
        gradient = cv2.Laplacian(processed, cv2.CV_64F)  # Compute Laplacian
        gradient = cv2.convertScaleAbs(gradient)  # Convert to 8-bit format
        processed = cv2.bitwise_or(processed, gradient)  # Merge with thresholded image

    # Apply grain boundary detection
    if detect_grain_boundaries:
        # Compute the morphological gradient (dilation - erosion) to highlight
        # grain boundaries
        kernel = np.ones((3, 3), np.uint8)
        grain_boundaries = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
        processed = cv2.bitwise_or(
            processed, grain_boundaries
        )  # Merge grain boundaries with existing features

    return processed


def compute_markers(
    binary_image: np.ndarray,
    morph_kernel_size=(3, 3),
    dilation_iter=3,
    dist_transform_factor=0.5,
    min_foreground_area=100,
):
    kernel = np.ones(morph_kernel_size, np.uint8)

    # Noise removal using morphological opening
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area using dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=dilation_iter)

    # Distance transform and thresholding for sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, dist_transform_factor * dist_transform.max(), 255, 0
    )

    # Convert sure foreground to uint8
    sure_fg = np.uint8(sure_fg)

    # Unknown region (subtracting sure foreground from sure background)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # Filter out small regions
    for label in range(1, num_labels):
        if np.sum(markers == label) < min_foreground_area:
            markers[markers == label] = 0

    # Add 1 to all labels so that the background is not zero
    markers = markers + 1

    # Mark the unknown regions with zero
    markers[unknown == 255] = 0

    return markers
