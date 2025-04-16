"""
mask_generator.py

Module for preprocessing grayscale microscopy images and generating binary masks
for foreground (spherulites), background (Maltese crosses), and removed regions.

Main Functionalities:
----------------------
- Grayscale conversion and CLAHE contrast enhancement
- Foreground and background binary mask generation
- Removal of small objects based on physical size (in microns)
- Overlay of removed regions on grayscale image

Dependencies:
-------------
- Requires calibration constants from `scale_bar_config.py`

Author: Priyangika Pitawala
Date: April 2025
"""

import cv2
import numpy as np
from data_loading import scale_bar_config


def preprocess_image(
    image: np.ndarray,
    enhance_contrast: bool = True,
    clip_limit: float = 1.5,
    tile_grid_size: tuple = (8, 8),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts an RGB image to grayscale.
    Optionally applies CLAHE contrast enhancement.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    enhance_contrast : bool
        Whether to apply CLAHE.
    clip_limit : float
        CLAHE contrast clip limit.
    tile_grid_size : tuple
        Tile size for CLAHE.

    Returns
    -------
    gray_image : np.ndarray
        Preprocessed grayscale image.
    original_gray : np.ndarray
        Original grayscale image before enhancement.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_gray = gray_image.copy()

    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        gray_image = clahe.apply(gray_image)

    return gray_image, original_gray


def compute_foreground_background_masks(
    gray_image: np.ndarray,
    background_thresh: int = 60,
    min_object_length_um: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates cleaned binary masks for background and foreground objects,
    along with a combined mask of all removed regions.

    Parameters
    ----------
    gray_image : np.ndarray
        Input grayscale image.
    background_thresh : int
        Intensity threshold separating background from foreground.
    min_object_length_um : float
        Minimum object size to retain, in microns.

    Returns
    -------
    background_mask : np.ndarray
        Cleaned binary mask of background (0/255).
    foreground_mask : np.ndarray
        Cleaned binary mask of foreground (0/255).
    removed_mask : np.ndarray
        Combined binary mask of removed regions (0/255).
    """
    background_raw = (gray_image < background_thresh).astype(np.uint8)
    foreground_raw = (gray_image >= background_thresh).astype(np.uint8)

    bg_clean, bg_removed = remove_small_objects_by_size(
        background_raw, min_object_length_um, return_removed_mask=True
    )
    fg_clean, fg_removed = remove_small_objects_by_size(
        foreground_raw, min_object_length_um, return_removed_mask=True
    )

    removed_combined = cv2.bitwise_or(bg_removed, fg_removed)

    return bg_clean * 255, fg_clean * 255, removed_combined * 255


def remove_small_objects_by_size(
    binary_mask: np.ndarray,
    min_object_length_um: float,
    return_removed_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Removes connected components below a minimum physical length in microns.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary image with objects (0 or 1).
    min_object_length_um : float
        Minimum physical length to keep (in microns).
    return_removed_mask : bool
        If True, also return a mask of removed objects.

    Returns
    -------
    filtered_mask : np.ndarray
        Binary image with small objects removed.
    removed_mask : np.ndarray (optional)
        Binary image showing only the removed regions.
    """
    px_per_um = scale_bar_config.SCALE_BAR_PIXELS / scale_bar_config.SCALE_BAR_MICRONS
    min_size_px = int(np.ceil(min_object_length_um * px_per_um))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    filtered = np.zeros_like(binary_mask)
    removed = np.zeros_like(binary_mask)

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        max_dim = max(w, h)
        if max_dim >= min_size_px:
            filtered[labels == label] = 1
        else:
            removed[labels == label] = 1

    if return_removed_mask:
        return filtered, removed
    return filtered


def overlay_removed_regions(
    gray_image: np.ndarray,
    removed_mask: np.ndarray,
    color: tuple = (0, 0, 255),
) -> np.ndarray:
    """
    Overlays removed region contours onto the original grayscale image.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale image to display.
    removed_mask : np.ndarray
        Binary mask of removed regions.
    color : tuple
        BGR color for overlay (default = red).

    Returns
    -------
    overlay_img : np.ndarray
        Color image with overlay drawn.
    """
    overlay_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    overlay_img[removed_mask > 0] = color
    return overlay_img
