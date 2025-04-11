import numpy as np
import pandas as pd

def compute_crystallinity_metrics(feature_df: pd.DataFrame, image_shape: tuple, image_id: str, metadata: dict = None) -> pd.DataFrame:
    """
    Computes crystallite size distribution and percent crystallinity for a single image.

    Parameters:
    - feature_df (pd.DataFrame): Region-level features from segmentation.
    - image_shape (tuple): Shape of the original image (height, width).
    - image_id (str): Identifier for the image (e.g., filename or sample ID).
    - metadata (dict): Optional dictionary with metadata (e.g., thickness, light intensity).

    Returns:
    - pd.DataFrame: Summary DataFrame for the image.
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
