import cv2
import numpy as np

def overlay_boundaries_on_mask(foreground_mask: np.ndarray,
                                markers: np.ndarray,
                                color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Overlays watershed boundaries (markers == -1) on the cleaned foreground mask.

    Parameters:
    - foreground_mask (np.ndarray): Binary mask of spherulites (0 or 255).
    - markers (np.ndarray): Marker image from watershed (int32), with -1 as boundary.
    - color (tuple): BGR color to draw boundaries (default: green).

    Returns:
    - overlay (np.ndarray): Foreground mask in 3-channel BGR with boundaries overlaid.
    """
    # Convert binary mask to 3-channel BGR image
    fg_rgb = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)

    # Boundary mask: where watershed marked -1
    boundary_mask = markers == -1
    fg_rgb[boundary_mask] = color

    return fg_rgb

def overlay_combined_boundaries(foreground_mask: np.ndarray,
                                watershed_markers: np.ndarray,
                                edge_labels: np.ndarray,
                                color_watershed=(0, 255, 0),
                                color_edges=(255, 0, 0)) -> np.ndarray:
    """
    Overlays watershed and edge-based boundaries on the foreground mask.

    Parameters:
    - foreground_mask (np.ndarray): Binary mask of spherulites (0 or 255).
    - watershed_markers (np.ndarray): Markers from watershed (boundary = -1).
    - edge_labels (np.ndarray): Labeled image from edge segmentation.
    - color_watershed (tuple): BGR color for watershed boundaries.
    - color_edges (tuple): BGR color for edge-based boundaries.

    Returns:
    - overlay (np.ndarray): Foreground mask with both boundary types overlaid.
    """
    fg_rgb = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)

    # Watershed boundaries
    fg_rgb[watershed_markers == -1] = color_watershed

    # Edge-based boundaries (contour outlines)
    edge_boundaries = cv2.Laplacian(edge_labels.astype(np.uint8), cv2.CV_8U)
    fg_rgb[edge_boundaries > 0] = color_edges

    return fg_rgb

def overlay_labels_as_boundaries(base_image: np.ndarray,
                                 label_mask: np.ndarray,
                                 color: tuple = (0, 0, 255)) -> np.ndarray:
    """
    Overlays label boundaries on a base image.

    Parameters:
    - base_image (np.ndarray): Grayscale or binary image (2D), used as background.
    - label_mask (np.ndarray): Labeled regions to extract boundaries from.
    - color (tuple): BGR color to overlay the boundaries.

    Returns:
    - overlay (np.ndarray): BGR image with boundaries highlighted.
    """
    # Convert base to BGR for overlay
    if len(base_image.shape) == 2:
        base_bgr = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        base_bgr = base_image.copy()

    # Get boundaries via Laplacian
    boundary_mask = cv2.Laplacian(label_mask.astype(np.uint8), cv2.CV_8U)
    base_bgr[boundary_mask > 0] = color

    return base_bgr
