# Module for generating masks for foreground and background of the image.

# This module contains functions for preprocesing a raw image by converting to grayscale and enhancing contrast (if enabled),
# computing masks for the foreground and background, removing noise (defined to be smaller than a specified size), and returning
# a grayscale of the raw image overlaid with the removed regions for visual verification.

# Author: Priyangika Pitawala
# Date: March 2025

import cv2
import numpy as np
from data_loading import scale_bar_config


def preprocess_image(
    image: np.ndarray, enhance_contrast: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_image = gray_image.copy()

    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray_image = clahe.apply(gray_image)

    return gray_image, original_image


def compute_foreground_background_masks(
    gray_image: np.ndarray,
    background_thresh: int = 60,
    min_physical_length_um: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates binary masks and a combined removed regions mask.

    Returns:
    - known_background_mask (np.ndarray): Cleaned binary background mask (0 or 255).
    - known_foreground_mask (np.ndarray): Cleaned binary foreground mask (0 or 255).
    - combined_removed_mask (np.ndarray): Binary mask of all removed regions (0 or 255).
    """
    background_mask_raw = (gray_image < background_thresh).astype(np.uint8)
    foreground_mask_raw = (gray_image >= background_thresh).astype(np.uint8)

    # Get cleaned masks and removed region masks
    bg_clean, bg_removed = remove_small_objects_by_size(
        background_mask_raw, min_physical_length_um, return_removed_mask=True
    )
    fg_clean, fg_removed = remove_small_objects_by_size(
        foreground_mask_raw, min_physical_length_um, return_removed_mask=True
    )

    # Combine removed regions
    combined_removed_mask = cv2.bitwise_or(bg_removed, fg_removed)

    return bg_clean * 255, fg_clean * 255, combined_removed_mask * 255


def remove_small_objects_by_size(
    binary_mask: np.ndarray,
    min_physical_length_um: float,
    return_removed_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Removes small objects and optionally returns a mask of removed regions.

    Parameters:
    - binary_mask (np.ndarray): Binary image with objects.
    - min_physical_length_um (float): Minimum size in microns.
    - return_removed_mask (bool): If True, also return mask of removed regions.

    Returns:
    - filtered_mask (np.ndarray): Cleaned binary mask.
    - removed_mask (np.ndarray, optional): Binary mask of removed regions.
    """
    px_per_micron = (
        scale_bar_config.SCALE_BAR_PIXELS / scale_bar_config.SCALE_BAR_MICRONS
    )
    min_size_px = int(np.ceil(min_physical_length_um * px_per_micron))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    filtered_mask = np.zeros_like(binary_mask)
    removed_mask = np.zeros_like(binary_mask)

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        max_dim = max(w, h)
        if max_dim >= min_size_px:
            filtered_mask[labels == label] = 1
        else:
            removed_mask[labels == label] = 1

    if return_removed_mask:
        return filtered_mask, removed_mask
    return filtered_mask


def overlay_removed_regions(
    gray_image: np.ndarray, removed_mask: np.ndarray, color=(0, 0, 255)
) -> np.ndarray:
    """
    Overlays removed regions in color on a grayscale image.

    Parameters:
    - gray_image (np.ndarray): Grayscale background image.
    - removed_mask (np.ndarray): Binary mask of removed regions.
    - color (tuple): BGR color for overlay (default: red).

    Returns:
    - overlay_img (np.ndarray): Color image with overlay.
    """
    overlay_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    overlay_img[removed_mask > 0] = color
    return overlay_img
