import cv2
import numpy as np

def compute_markers(foreground_mask: np.ndarray,
                    morph_kernel_size=(3, 3),
                    dilation_iter=2,
                    dist_transform_factor=0.3,
                    min_foreground_area=50) -> np.ndarray:
    """
    Computes marker image for watershed segmentation based on cleaned foreground mask.

    Parameters:
    - foreground_mask (np.ndarray): Binary mask of foreground (spherulites), 0 or 255.
    - morph_kernel_size (tuple): Kernel size for morphological operations.
    - dilation_iter (int): Dilation iterations for sure background.
    - dist_transform_factor (float): Distance transform threshold factor for sure foreground.
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
    _, sure_fg = cv2.threshold(dist_transform,
                               dist_transform_factor * dist_transform.max(),
                               255, 0)
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
