import cv2
import numpy as np

def apply_watershed(image: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """
    Applies the watershed algorithm to segment regions in the given image and overlays the segmentation boundary on a grayscale version of the image.
    
    Parameters:
    - image (np.ndarray): Original color image (BGR format).
    - markers (np.ndarray): Marker image generated from compute_markers().
    
    Returns:
    - np.ndarray: Grayscale image with watershed segmentation boundary overlaid in red.
    """
    if image is None or markers is None:
        raise ValueError("Input image and markers must be valid numpy arrays.")

    if len(image.shape) < 2 or len(markers.shape) < 2:
        raise ValueError("Invalid image or marker dimensions.")

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel for overlay

    # Ensure markers are in int32 format as required by OpenCV
    markers = markers.astype(np.int32)

    # Apply watershed algorithm
    cv2.watershed(image, markers)
    
    # Create a thin boundary mask
    boundary_mask = markers == -1  # Boundary pixels
    gray_image[boundary_mask] = [0, 0, 255]  # Mark boundaries in red (BGR format: Blue=0, Green=0, Red=255)
    
    return gray_image
