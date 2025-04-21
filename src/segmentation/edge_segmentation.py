"""
edge_segmentation.py

This module provides edge-free segmentation for spherulites using simple connected
component labeling on a binary foreground mask.

Overview:
---------
The segmentation is performed without explicitly detecting edges. Instead, the function
`segment_by_edges()` treats each white region in the binary foreground mask as a unique
spherulite. It assigns a unique integer label to each connected region using OpenCV's
connected components analysis.

Use Case:
---------
- Best suited for clean binary masks where spherulites are well-separated.
- Avoids over-segmentation or boundary noise from gradient-based or edge-based methods.
- Compatible with hybrid segmentation workflows (e.g., combining with watershed).

Function:
---------
segment_by_edges(gray_image, foreground_mask)
    - gray_image (np.ndarray): Input grayscale image (unused in current logic,
    retained for compatibility).
    - foreground_mask (np.ndarray): Binary mask (0 background, 255 foreground).

Returns:
--------
- np.ndarray: Labeled mask where each connected region (spherulite) is assigned
a unique integer ID.
  Background remains labeled as 0.

Dependencies:
-------------
- OpenCV (cv2)
- NumPy

Example:
--------
```python
from edge_segmentation import segment_by_edges
labeled = segment_by_edges(gray_image, foreground_mask)
```

#Author: Priyangika Pitawala
#Date: April 2025
"""

import cv2
import numpy as np


def segment_by_edges(gray_image: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    """
    Segments spherulites by labeling connected white regions in the foreground mask.
    Avoids edge detection entirely and treats each white region as a separate
    spherulite.

    Parameters:
    - gray_image (np.ndarray): CLAHE-enhanced grayscale image (not used here
    but retained for compatibility).
    - foreground_mask (np.ndarray): Binary mask of known spherulites (0 or 255).

    Returns:
    - labels (np.ndarray): Labeled mask where each connected region gets a unique ID.
    """
    # Ensure mask is binary 0 and 255
    binary_mask = (foreground_mask > 0).astype(np.uint8)

    # Label connected white regions
    _, labels = cv2.connectedComponents(binary_mask)

    return labels
