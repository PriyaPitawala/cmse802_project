# Module for image preprocessing.

# This module contains functions for preprocessing an image to prepare it for segmentation.
# It converts the image to grayscale, applies filtering and thresholding, 
# with optional enhancement of contrast and boundary definition. Furthermore, it computes markers
# to enable watershed algorithm in latter stages.

# Author: Priyangika Pitawala
# Date: March 2025

import cv2
import numpy as np

def preprocess_image(image: np.ndarray, blur_kernel=(5,5), threshold_method=cv2.THRESH_BINARY, 
                     threshold_value=127, adaptive=False, block_size=11, C=2,
                     use_edge_detection=False, edge_low_threshold=50, edge_high_threshold=150,
                     use_gradient_thresholding=False, enhance_contrast=False, detect_grain_boundaries=False,
                     merge_inner_outer=False, morph_kernel_size=(5,5)):
    """
    Preprocesses an image for segmentation, with optional edge detection, gradient thresholding,
    contrast enhancement (CLAHE), grain boundary detection, and inner-outer region merging.

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
    - detect_grain_boundaries (bool): If True, apply morphological gradient for boundary detection.
    - merge_inner_outer (bool): If True, flood-fill inner regions to merge with outer layers.
    - morph_kernel_size (tuple): Kernel size for morphological operations (closing).

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
        kernel = np.ones(morph_kernel_size, np.uint8)
        grain_boundaries = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
        processed = cv2.bitwise_or(processed, grain_boundaries)  # Merge grain boundaries with existing features

    # Merge inner and outer regions using flood-fill
    if merge_inner_outer:
        floodfilled_image = processed.copy()
        h, w = floodfilled_image.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)  # Flood-fill requires a slightly larger mask

        # Flood-fill from multiple seed points to merge bright inner regions
        for y in range(0, h, 50):  # Sample grid points for filling
            for x in range(0, w, 50):
                if floodfilled_image[y, x] == 255:  # Only fill white regions
                    cv2.floodFill(floodfilled_image, mask, (x, y), 128)  # Fill with gray (128)

        # Convert all filled regions back to white (255)
        floodfilled_image[floodfilled_image == 128] = 255

        # Apply Morphological Closing to Connect Gaps
        processed = cv2.morphologyEx(floodfilled_image, cv2.MORPH_CLOSE, kernel)

    # Remove small white noise in the preprocessed image before computing markers
    opening_kernel = np.ones((3,3), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, opening_kernel, iterations=1)


    return processed

def compute_markers(binary_image: np.ndarray, morph_kernel_size=(3,3), dilation_iter=1, 
                    dist_transform_factor=0.4, min_foreground_area=50):
    """
    Computes markers for watershed segmentation.
    """
    kernel = np.ones(morph_kernel_size, np.uint8)

    # Step 1: Remove noise using morphological opening
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # Debug 1: Display the result after morphological opening
    cv2.imshow("Morphological Opening", opening)
    cv2.waitKey(0)

    # Step 2: Define the background region
    sure_bg = cv2.dilate(opening, kernel, iterations=dilation_iter)

    # Debug 2: Display sure background
    cv2.imshow("Sure Background", sure_bg)
    cv2.waitKey(0)

    # **Key Fix: Ensure Distance Transform Input Is Not Fully White**
    opening_uint8 = np.uint8(opening)  # Ensure proper format

    # Invert binary image if needed (to make sure foreground is detected correctly)
    foreground_mask = cv2.bitwise_not(opening_uint8)
    cv2.imshow("Foreground Mask for Distance Transform", foreground_mask)
    cv2.waitKey(0)

    # Step 3: Verify Binary Image Before Distance Transform
    binary_check = np.uint8(opening > 0) * 255  # Ensure proper binary format
    cv2.imshow("Binary Image Before Distance Transform", binary_check)
    cv2.waitKey(0)


    # Step 3: Compute Distance Transform
    dist_transform = cv2.distanceTransform(opening_uint8, cv2.DIST_L2, 5)

    # Normalize the distance transform to remove the vertical gradient effect
    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)

    # Debug 3: Display the normalized distance transform
    cv2.imshow("Normalized Distance Transform", dist_transform.astype(np.uint8))
    cv2.waitKey(0)

    # Step 4: Threshold the distance transform to obtain sure foreground
    _, sure_fg = cv2.threshold(dist_transform, dist_transform_factor * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Debug 4: Display the sure foreground after thresholding
    cv2.imshow("Sure Foreground", sure_fg)
    cv2.waitKey(0)

    # Step 5: Compute unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 6: Label connected components
    _, markers = cv2.connectedComponents(sure_fg)

    # Debug 5: Display the connected component markers
    cv2.imshow("Initial Markers", markers.astype(np.uint8) * 50)
    cv2.waitKey(0)

    # Step 7: Adjust marker labels
    markers = markers + 1
    markers[unknown == 255] = 0

    # Debug 6: Display the final marker image before returning
    cv2.imshow("Final Markers", markers.astype(np.uint8) * 50)
    cv2.waitKey(0)

    return markers
