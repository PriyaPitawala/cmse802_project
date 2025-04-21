"""
preprocess_malcross.py

This module provides image preprocessing utilities for isolating Maltese crosses
in polarized optical microscopy (POM) images of semicrystalline photopolymers.

Main Functionality:
-------------------
- Converts input BGR image to grayscale
- Applies CLAHE contrast enhancement (optional)
- Uses Gaussian blur to suppress noise
- Applies either adaptive or global thresholding to generate a binary mask
  that highlights bright cross-like nucleation centers (Maltese crosses)

Typical Use:
------------
Used in workflows where nucleation centers need to be extracted separately
from spherulitic regions for further segmentation, quantification, or
crystallinity analysis.

Function:
---------
- preprocess_image:
    Accepts a raw image and outputs a binary mask where Maltese cross regions
    are isolated based on intensity and local contrast features.

Dependencies:
-------------
- OpenCV (cv2)
- NumPy (np)

Example:
--------
```python
from preprocessing import preprocess_malcross

binary_mask = preprocess_malcross.preprocess_image(
    image=raw_image,
    adaptive=True,
    enhance_contrast=True
)

#Author: Priyangika Pitawala
#Date: April 2025
"""

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
    Preprocess image to isolate Maltese crosses via thresholding and
    contrast enhancement.


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
