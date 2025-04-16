# Module for image preprocessing specifically to segment Maltese crosses.

# This module contains functions for preprocessing an image to prepare it for segmentation of Maltese crosses.
# It converts the image to grayscale, enhances contrast (if enabled), applies a Gaussian blur to smooth over noise, and
# applies adaptive thresholding (if enabled).

# Author: Priyangika Pitawala
# Date: March 2025

import cv2
import numpy as np


def preprocess_image(
    image: np.ndarray,
    blur_kernel=(5, 5),
    threshold_method=cv2.THRESH_BINARY_INV,
    threshold_value=120,
    adaptive=True,
    block_size=17,
    C=3,
    enhance_contrast=True,
) -> np.ndarray:
    """
    Preprocess image to isolate Maltese crosses via thresholding and contrast enhancement.

    Parameters:
    - image (np.ndarray): Input BGR image.
    - blur_kernel (tuple): Gaussian blur kernel size.
    - threshold_method (int): OpenCV thresholding method.
    - threshold_value (int): Value for global thresholding.
    - adaptive (bool): Whether to use adaptive thresholding.
    - block_size (int): Block size for adaptive thresholding.
    - C (int): Constant subtracted in adaptive thresholding.
    - enhance_contrast (bool): Use CLAHE for local contrast enhancement.

    Returns:
    - binary_mask (np.ndarray): Binary image highlighting Maltese cross regions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # Apply thresholding
    if adaptive:
        binary_mask = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_method,
            block_size,
            C,
        )
    else:
        _, binary_mask = cv2.threshold(blurred, threshold_value, 255, threshold_method)

    return binary_mask
