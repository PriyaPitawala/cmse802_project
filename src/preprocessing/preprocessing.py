"""
preprocessing.py

Another legacy image preprocessing module for segmenting spherulitic features in
polarized optical microscopy (POM) images of semicrystalline photopolymers.

This module implements advanced techniques for noise suppression, grain boundary
detection, foreground refinement, and robust watershed marker generation.

Key Features:
-------------
- CLAHE contrast enhancement
- Edge-based segmentation via morphological gradients
- Intensity- and texture-based region filling
- Foreground marker generation using peak-local-max and distance transforms
- Optional suppression of known background and dark-edge artifacts

Functions:
----------
1. preprocess_image:
   - Converts input BGR image to grayscale
   - Optionally enhances contrast (CLAHE)
   - Computes morphological gradients to identify grain boundaries
   - Suppresses dark backgrounds and fills high-intensity, high-variation regions
   - Accepts known background masks to exclude from further segmentation
   - Returns either the final mask or a debug dictionary of intermediate steps

2. compute_markers:
   - Applies morphological opening and distance transforms
   - Suppresses edge-based noise before applying distance transform
   - Uses peak-local-maximum to seed watershed markers
   - Returns a labeled marker image

3. visualize_markers:
   - Converts watershed markers to a color-coded RGB overlay for inspection

Usage Example:
--------------
```python
from preprocessing import preprocess_image, compute_markers, visualize_markers

# Step 1: Preprocess the image
foreground_mask = preprocess_image(raw_image)

# Step 2: Generate markers for watershed segmentation
markers = compute_markers(foreground_mask)

# Step 3: Visualize the marker regions
overlay = visualize_markers(markers)
```

Dependencies:
-------------
- OpenCV (cv2)
- NumPy
- SciPy (ndimage, gaussian_filter)
- scikit-image (peak_local_max)

#Author: Priyangika Pitawala
#Date: April 2025
"""

import cv2
import numpy as np
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter


def preprocess_image(
    image: np.ndarray,
    blur_kernel=(5, 5),
    enhance_contrast=True,
    detect_grain_boundaries=True,
    background_threshold=30,
    fill_intensity_threshold=60,
    variation_threshold=10,
    grain_smoothing_kernel=(9, 9),
    suppress_dark_edges=True,
    known_background_mask: np.ndarray = None,
    return_debug=False,
) -> dict:
    debug = {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug["gray"] = gray.copy()
    original_gray = gray.copy()
    dark_mask = gray < background_threshold

    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        debug["clahe"] = gray.copy()

    smoothed = cv2.GaussianBlur(gray, blur_kernel, 0)
    debug["blurred"] = smoothed.copy()

    if detect_grain_boundaries:
        kernel = np.ones(grain_smoothing_kernel, np.uint8)
        gradient = cv2.morphologyEx(smoothed, cv2.MORPH_GRADIENT, kernel)
    else:
        gradient = smoothed.copy()

    if suppress_dark_edges:
        gradient[dark_mask] = 0
    debug["gradient"] = gradient.copy()

    _, edges_binary = cv2.threshold(
        gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    debug["edges"] = edges_binary.copy()

    contours, _ = cv2.findContours(
        edges_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled_mask = np.zeros_like(edges_binary)

    for contour in contours:
        contour_mask = np.zeros_like(edges_binary)
        cv2.drawContours(contour_mask, [contour], -1, color=255, thickness=-1)
        mean_intensity = cv2.mean(original_gray, mask=contour_mask)[0]
        std_dev = np.std(original_gray[contour_mask == 255])

        if (
            mean_intensity >= fill_intensity_threshold
            and std_dev >= variation_threshold
        ):
            filled_mask = cv2.bitwise_or(filled_mask, contour_mask)

    debug["filled_mask"] = filled_mask.copy()

    combined_mask = cv2.bitwise_or(edges_binary, filled_mask)

    if known_background_mask is not None:
        combined_mask[known_background_mask == 255] = 0
        debug["background_mask"] = known_background_mask.copy()

    debug["combined_mask"] = combined_mask.copy()
    debug["final_mask"] = combined_mask.copy()

    return debug if return_debug else combined_mask


def compute_markers(
    binary_image: np.ndarray,
    morph_kernel_size=(3, 3),
    dilation_iter=2,
    dist_transform_factor=0.3,
    min_foreground_area=50,
) -> np.ndarray:
    kernel = np.ones(morph_kernel_size, np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=dilation_iter)

    edges = cv2.Canny(binary_image, 50, 150)
    suppress_mask = cv2.bitwise_not(edges)
    masked_fill = cv2.bitwise_and(opening, opening, mask=suppress_mask)

    dist_transform = cv2.distanceTransform(masked_fill, cv2.DIST_L2, 5)
    dist_transform = gaussian_filter(dist_transform, sigma=1.0)

    coordinates = peak_local_max(
        dist_transform, labels=masked_fill, footprint=np.ones((3, 3)), min_distance=5
    )
    local_max = np.zeros_like(dist_transform, dtype=bool)
    local_max[tuple(coordinates.T)] = True
    markers, _ = ndi.label(local_max)

    unknown = cv2.subtract(sure_bg, masked_fill)
    markers[unknown == 255] = 0

    return markers


def visualize_markers(markers: np.ndarray) -> np.ndarray:
    display = cv2.applyColorMap(
        cv2.convertScaleAbs(markers, alpha=10), cv2.COLORMAP_JET
    )
    display[markers == 1] = [0, 0, 0]
    display[markers == 0] = [255, 255, 255]
    return display
