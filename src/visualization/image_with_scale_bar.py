"""
image_with_scale_bar.py

This module provides utilities for visualizing microscopy images with a
scale bar overlay.

It standardizes image display by:
- Converting raw image formats (e.g., grayscale, float, or 16-bit) into BGR
  for consistent rendering
- Overlaying a calibrated scale bar with customizable appearance and label
- Reading visual parameters from `scale_bar_config.py` to ensure visual
  consistency across datasets

Intended for use in:
- Manual image inspection
- Documentation and figure generation
- Visualization verification in segmentation workflows

Author: Priyangika Pitawala
Date: April 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams

from data_loading.scale_bar_config import (
    SCALE_BAR_PIXELS,
    SCALE_BAR_TEXT,
    SCALE_BAR_HEIGHT,
    SCALE_BAR_MARGIN,
    SCALE_BAR_BACKGROUND_HEIGHT,
    FONT_FAMILY,
    FONT_SIZE,
)

# Apply global font configuration for all matplotlib text
rcParams["font.family"] = FONT_FAMILY


def convert_to_displayable(image: np.ndarray) -> np.ndarray:
    """
    Converts an input image to displayable format (8-bit 3-channel BGR).

    This is useful for ensuring compatibility with OpenCV and matplotlib,
    especially when working with grayscale, float, or high-bit-depth images.

    Parameters
    ----------
    image : np.ndarray
        Input image in grayscale, float32/float64, or 16-bit format.

    Returns
    -------
    np.ndarray
        Display-ready image in 8-bit, 3-channel BGR format.
    """
    if image.dtype in [np.int32, np.int64]:
        image = image.astype(np.uint8)
    elif image.dtype in [np.float32, np.float64]:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def display_image(image: np.ndarray, title="Image"):
    """
    Displays the image with an overlaid scale bar and title using matplotlib.

    This function:
    - Converts grayscale or float images to displayable BGR format
    - Adds a white background box for visual contrast
    - Draws a calibrated scale bar and label (e.g., "60 µm") at the bottom right

    Parameters
    ----------
    image : np.ndarray
        Image to display. Can be grayscale or color, any bit-depth.

    title : str, optional
        Title for the displayed image (default is "Image").

    Notes
    -----
    - Uses visual style parameters from `scale_bar_config.py`
    - Only renders if the matplotlib backend supports interactive display
    """
    image = convert_to_displayable(image)
    height, width = image.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")

    # === This is the crucial alignment fix ===
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    # === Custom padding values ===
    padding_left = 20
    padding_right = 20
    padding_bottom = 15
    label_above_bar_offset = 30
    extra_buffer = 20  # Optional, ensures full coverage

    # === Background box dimensions (DYNAMICALLY CALCULATED) ===
    background_width = SCALE_BAR_PIXELS + padding_left + padding_right
    background_height = (
        SCALE_BAR_HEIGHT + label_above_bar_offset + padding_bottom + extra_buffer
    )

    # === Anchor white box at bottom-right corner of image ===
    background_x = width - background_width
    background_y = height - background_height

    rect = Rectangle(
        (background_x, background_y),
        background_width,
        background_height,
        color="white",
        zorder=2,
    )
    ax.add_patch(rect)

    # Bar position: a bit above bottom of white box
    x_start = background_x + padding_left
    y_start = background_y + background_height - padding_bottom

    # === Draw black scale bar ===
    ax.plot(
        [x_start, x_start + SCALE_BAR_PIXELS],
        [y_start, y_start],
        color="black",
        linewidth=SCALE_BAR_HEIGHT,
        zorder=3,
    )

    # === Draw scale bar label ===
    ax.text(
        x_start + SCALE_BAR_PIXELS / 2,
        y_start - label_above_bar_offset,
        SCALE_BAR_TEXT,
        fontsize=FONT_SIZE,
        fontweight="bold",
        color="black",
        ha="center",
        va="center",
        zorder=4,
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2"),
    )

    # Show only in interactive backends
    import matplotlib

    if matplotlib.get_backend() != "Agg":
        plt.show()

    return fig
