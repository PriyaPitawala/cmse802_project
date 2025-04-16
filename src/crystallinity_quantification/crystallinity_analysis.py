"""
crystallinity_analysis.py

This module computes crystallite size distribution and percent crystallinity
for segmented POM images. It processes a single image's region features and
returns a summary DataFrame.

Typical usage:
--------------
Use compute_crystallinity_metrics() after segmenting an image and extracting
region-level properties like area and equivalent diameter.

Returns:
--------
- image_id, image shape
- crystallite count, area, percent crystallinity
- diameter statistics (mean, median, std)
- metadata (thickness, light intensity, photoabsorber), if provided

Author: Priyangika Pitawala
Date: April 2025
"""

import pandas as pd


def compute_crystallinity_metrics(
    feature_df: pd.DataFrame,
    image_shape: tuple,
    image_id: str,
    metadata: dict = None,
) -> pd.DataFrame:
    """
    Computes crystallite size distribution and percent crystallinity
    for a single image.

    Parameters:
    -----------
    feature_df : pd.DataFrame
        Region-level features from segmentation. Must include 'area' and
        'equivalent_diameter' columns.

    image_shape : tuple
        Shape of the raw image (height, width).

    image_id : str
        Unique identifier for the image/sample.

    metadata : dict, optional
        Dictionary with optional metadata:
        e.g., {'thickness': 100, 'light_intensity': 5, 'photoabsorber': 0.01}

    Returns:
    --------
    pd.DataFrame
        One-row summary DataFrame with crystallinity metrics and metadata.
    """
    image_area = image_shape[0] * image_shape[1]
    total_crystallite_area = feature_df["area"].sum()
    percent_crystallinity = 100 * total_crystallite_area / image_area

    summary = {
        "image_id": image_id,
        "image_height": image_shape[0],
        "image_width": image_shape[1],
        "num_crystallites": len(feature_df),
        "total_crystallite_area": total_crystallite_area,
        "percent_crystallinity": percent_crystallinity,
        "mean_diameter": feature_df["equivalent_diameter"].mean(),
        "median_diameter": feature_df["equivalent_diameter"].median(),
        "std_diameter": feature_df["equivalent_diameter"].std(),
    }

    if metadata:
        summary.update(metadata)

    return pd.DataFrame([summary])
