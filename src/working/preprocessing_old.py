import cv2
import numpy as np

def preprocess_image(image: np.ndarray, blur_kernel=(5,5), threshold_method=cv2.THRESH_BINARY, 
                     threshold_value=127, adaptive=False, block_size=11, C=2,
                     use_edge_detection=False, edge_low_threshold=50, edge_high_threshold=150,
                     use_gradient_thresholding=False, enhance_contrast=False, detect_grain_boundaries=False):
    """
    Preprocesses an image for segmentation, with optional edge detection, gradient thresholding,
    contrast enhancement (CLAHE), and grain boundary detection.

    Parameters:
    - image (np.ndarray): Input image in BGR format.
    - blur_kernel (tuple): Kernel size for Gaussian blur.
    - threshold_method (int): OpenCV thresholding method.
    - threshold_value (int): Global threshold value (ignored if adaptive=True).
    - adaptive (bool): If True, use adaptive thresholding.
    - block_size (int): Size of the neighborhood for adaptive thresholding.
    - C (int): Constant subtracted from the mean in adaptive thresholding.
    - use_edge_detection (bool): If True, apply Canny edge detection.
    - edge_low_threshold (int): Lower threshold for Canny edge detection.
    - edge_high_threshold (int): Upper threshold for Canny edge detection.
    - use_gradient_thresholding (bool): If True, apply Laplacian-based gradient thresholding.
    - enhance_contrast (bool): If True, apply CLAHE to enhance local contrast.
    - detect_grain_boundaries (bool): If True, apply Structure Tensor-based boundary detection.

    Returns:
    - np.ndarray: Preprocessed binary image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for local contrast enhancement
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        gray_image = clahe.apply(gray_image)

    # Apply Gaussian Blur
    image_blur = cv2.GaussianBlur(gray_image, blur_kernel, 0)
    
    # Apply thresholding
    if adaptive:
        processed = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          threshold_method, block_size, C)
    else:
        _, processed = cv2.threshold(image_blur, threshold_value, 255, threshold_method)

    # Apply edge detection if enabled
    if use_edge_detection:
        edges = cv2.Canny(processed, edge_low_threshold, edge_high_threshold)
        processed = cv2.bitwise_or(processed, edges)  # Merge edges into thresholded image

    # Apply gradient-based thresholding if enabled
    if use_gradient_thresholding:
        gradient = cv2.Laplacian(processed, cv2.CV_64F)  # Compute Laplacian
        gradient = cv2.convertScaleAbs(gradient)  # Convert to 8-bit format
        processed = cv2.bitwise_or(processed, gradient)  # Merge with thresholded image

    # Apply grain boundary detection
    if detect_grain_boundaries:
        # Compute the morphological gradient (dilation - erosion) to highlight grain boundaries
        kernel = np.ones((3,3), np.uint8)
        grain_boundaries = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
        processed = cv2.bitwise_or(processed, grain_boundaries)  # Merge grain boundaries with existing features

    return processed

def compute_markers(binary_image: np.ndarray, morph_kernel_size=(3,3), dilation_iter=3, 
                    dist_transform_factor=0.5, min_foreground_area=100):
    """
    Computes markers for watershed segmentation.

    Parameters:
    - binary_image (np.ndarray): Preprocessed binary image.
    - morph_kernel_size (tuple): Kernel size for morphological operations.
    - dilation_iter (int): Number of iterations for dilation.
    - dist_transform_factor (float): Factor of the max distance transform to threshold the foreground.
    - min_foreground_area (int): Minimum area to keep a connected component in the foreground.

    Returns:
    - np.ndarray: Marker image for watershed algorithm.
    """
    kernel = np.ones(morph_kernel_size, np.uint8)

    # Noise removal using morphological opening
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area using dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=dilation_iter)

    # Distance transform and thresholding for sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, dist_transform_factor * dist_transform.max(), 255, 0)

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