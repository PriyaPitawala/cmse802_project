# Module for configuration of the scale bar.

# This module contains the configuration for recreating and overlaying a scale bar over polarized optical microscope (POM) images.
# The dimensions were extracted from an actual scale bar that matches the spatial calibration of the lens used for the project.

# Author: Priyangika Pitawala
# Date: March 2025

# Physical meaning
# Calibration constants for scale bar overlay

# Physical meaning
SCALE_BAR_MICRONS = 60
SCALE_BAR_PIXELS = 137  # Calibrated from real image
SCALE_BAR_TEXT = f"{SCALE_BAR_MICRONS} µm"

# Visual style
SCALE_BAR_HEIGHT = 4  # Thickness of black bar in pixels
SCALE_BAR_MARGIN = 20  # Distance from bottom and right edges
SCALE_BAR_BACKGROUND_HEIGHT = 60  # Height of the white background box

# Font style
FONT_FAMILY = "Arial"
FONT_SIZE = 12
