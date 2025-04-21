"""
scale_bar_config.py

Contains constants used to overlay a scale bar and its label on images
for visual calibration.

The scale bar is calibrated based on a known physical distance in microns
and its corresponding length in image pixels. This allows consistent and
accurate scale rendering across all processed images.

Constants:
----------
SCALE_BAR_MICRONS : int
    Physical length of the scale bar in microns (e.g., 100 µm).

SCALE_BAR_PIXELS : int
    Length of the scale bar in pixels (measured from a calibration image).

SCALE_BAR_TEXT : str
    Display label for the scale bar (e.g., "100 µm").

SCALE_BAR_HEIGHT : int
    Thickness of the black bar in pixels.

SCALE_BAR_MARGIN : int
    Distance from the image's bottom-right corner to place the scale bar.

SCALE_BAR_BACKGROUND_HEIGHT : int
    Height of the white rectangle behind the scale bar and label.

FONT_FAMILY : str
    Font family for rendering the text label.

FONT_SIZE : int
    Font size (in points) for the label.

Author: Priyangika Pitawala
Date: April 2025
"""

# Physical meaning
SCALE_BAR_MICRONS = 100
SCALE_BAR_PIXELS = 228  # Calibrated from real image
SCALE_BAR_TEXT = f"{SCALE_BAR_MICRONS} µm"

# Visual style
SCALE_BAR_HEIGHT = 10  # Thickness of black bar in pixels
SCALE_BAR_MARGIN = 20  # Distance from bottom and right edges
SCALE_BAR_BACKGROUND_HEIGHT = 100  # Height of the white background box

# Font style
FONT_FAMILY = "Arial"
FONT_SIZE = 24
