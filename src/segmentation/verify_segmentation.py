"""
verify_segmentation.py

This module provides utility functions for visualizing segmentation results
in binary masks and label images. It overlays watershed and edge-based boundaries
on spherulite masks, allowing for visual verification of segmentation performance.

Functions:
- overlay_boundaries_on_mask: Draws watershed boundaries on a binary mask.
- overlay_combined_boundaries: Merges watershed and edge-based boundaries.
- overlay_labels_as_boundaries: Highlights region boundaries from any labeled mask.

#Author: Priyangika Pitawala
#Date: April 2025
"""

import cv2
import numpy as np


def overlay_boundaries_on_mask(
    foreground_mask: np.ndarray, markers: np.ndarray, color: tuple = (0, 255, 0)
) -> np.ndarray:
    """
    Overlays watershed segmentation boundaries on the foreground mask.

    Parameters:
    - foreground_mask (np.ndarray): Binary spherulite mask (0 or 255).
    - markers (np.ndarray): Marker image from watershed (int32),
    with -1 marking boundaries.
    - color (tuple): BGR color for boundary overlay (default: green).

    Returns:
    - overlay (np.ndarray): 3-channel BGR image with watershed boundaries overlaid.
    """
    fg_rgb = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)
    boundary_mask = markers == -1
    fg_rgb[boundary_mask] = color
    return fg_rgb


def overlay_combined_boundaries(
    foreground_mask: np.ndarray,
    watershed_markers: np.ndarray,
    edge_labels: np.ndarray,
    color_watershed: tuple = (0, 255, 0),
    color_edges: tuple = (255, 0, 0),
) -> np.ndarray:
    """
    Overlays both watershed and edge-based segmentation boundaries on a foreground mask.

    Parameters:
    - foreground_mask (np.ndarray): Binary mask of spherulites (0 or 255).
    - watershed_markers (np.ndarray): Markers from watershed segmentation (int32).
    - edge_labels (np.ndarray): Labeled regions from edge-based segmentation.
    - color_watershed (tuple): BGR color for watershed boundaries (default: green).
    - color_edges (tuple): BGR color for edge-based boundaries (default: blue).

    Returns:
    - overlay (np.ndarray): 3-channel BGR image with both boundary types overlaid.
    """
    fg_rgb = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)
    fg_rgb[watershed_markers == -1] = color_watershed

    edge_boundaries = cv2.Laplacian(edge_labels.astype(np.uint8), cv2.CV_8U)
    fg_rgb[edge_boundaries > 0] = color_edges
    return fg_rgb


def overlay_labels_as_boundaries(
    base_image: np.ndarray, label_mask: np.ndarray, color: tuple = (0, 0, 255)
) -> np.ndarray:
    """
    Overlays the boundaries of labeled regions onto a grayscale or binary base image.

    Parameters:
    - base_image (np.ndarray): 2D grayscale or binary image used for display background.
    - label_mask (np.ndarray): Labeled mask where each region has a unique ID.
    - color (tuple): BGR color to draw the region boundaries (default: red).

    Returns:
    - overlay (np.ndarray): 3-channel BGR image with region boundaries overlaid.
    """
    base_bgr = (
        cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        if len(base_image.shape) == 2
        else base_image.copy()
    )
    boundary_mask = cv2.Laplacian(label_mask.astype(np.uint8), cv2.CV_8U)
    base_bgr[boundary_mask > 0] = color
    return base_bgr
