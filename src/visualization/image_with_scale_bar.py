# Module for image visualization.

# This module contains functions for visualizing an image to by converting it to the compatible format,
# and displaying the image with a scale bar matching the calibration of the lens used for capturing the image.

# Author: Priyangika Pitawala
# Date: March 2025

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
    FONT_SIZE
)

# Set global font
rcParams["font.family"] = FONT_FAMILY

def convert_to_displayable(image: np.ndarray) -> np.ndarray:
    if image.dtype in [np.int32, np.int64]:
        image = image.astype(np.uint8)
    elif image.dtype in [np.float32, np.float64]:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image

def display_image(image: np.ndarray, title="Image"):
    image = convert_to_displayable(image)
    height, width = image.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")

    # === Positioning (Lower right) ===
    x_start = width - SCALE_BAR_MARGIN - SCALE_BAR_PIXELS
    y_start = height - SCALE_BAR_MARGIN

    # === Draw white background box ===
    background_x = x_start - 10
    background_y = y_start - SCALE_BAR_BACKGROUND_HEIGHT + 10
    background_width = SCALE_BAR_PIXELS + 20
    background_height = SCALE_BAR_BACKGROUND_HEIGHT

    rect = Rectangle((background_x, background_y), background_width, background_height,
                     color="white", zorder=2)
    ax.add_patch(rect)

    # === Draw thick black scale bar ===
    ax.plot([x_start, x_start + SCALE_BAR_PIXELS],
            [y_start, y_start],
            color="black", linewidth=SCALE_BAR_HEIGHT, zorder=3)

    # === Draw text centered above bar ===
    ax.text(x_start + SCALE_BAR_PIXELS / 2,
            y_start - SCALE_BAR_BACKGROUND_HEIGHT / 2 + 5,
            SCALE_BAR_TEXT,
            color="black", fontsize=FONT_SIZE,
            ha="center", va="center", zorder=4,
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

    import matplotlib
    if matplotlib.get_backend() != "Agg":
        plt.show()
