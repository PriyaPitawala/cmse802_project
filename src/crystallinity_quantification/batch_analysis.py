"""
batch_analysis.py

Processes and stores crystallinity analysis results for individual images.

Main function:
- process_and_append_sample(): Computes crystallinity summary and saves results
  to features and summary tables.

Inputs:
- features_df: DataFrame of region-level features (area, diameter)
- image_shape: tuple of raw image height and width
- metadata: dict containing sample-specific info (e.g., thickness, light intensity)
- image_id: unique ID string for the image

Author: Priyangika Pitawala
Date: April 2025
"""

import os
import pandas as pd

from crystallinity_quantification.crystallinity_analysis import compute_crystallinity_metrics
from crystallinity_quantification.visualize_crystallinity import plot_crystallite_size_distribution


def get_project_root() -> str:
    """
    Returns the absolute path to the root of the project by going two levels up from the current
    working directory. Assumes you're running from `notebooks/exploratory/` or similar inside project.
    """
    return os.path.abspath(os.path.join(os.getcwd(), "../../"))


def process_and_append_sample(
    image_id: str,
    features_df: pd.DataFrame,
    image_shape: tuple,
    metadata: dict,
    quality_weight: float = 1.0,
    notes: str = "",
    output_csv_name: str = "crystallinity_summary_all_images.csv",
):
    """
    Computes crystallinity metrics and appends them to a global summary CSV.

    Parameters:
    ----------
    image_id : str
        Unique identifier for the sample/image.
    
    features_df : pd.DataFrame
        Extracted region features (e.g., area, equivalent diameter).
    
    image_shape : tuple
        Shape of the original image (height, width).
    
    metadata : dict
        Dictionary of experimental parameters (e.g., thickness, light intensity).
    
    quality_weight : float, optional
        Segmentation quality score (default is 1.0).
    
    notes : str, optional
        Optional annotation for manual notes.
    
    output_csv_name : str, optional
        Filename for the global summary table.

    Outputs:
    -------
    - Saves histogram plot to default plot directory (handled by visualize module)
    - Updates results/tables/{output_csv_name} with one-row summary
    - Saves raw features to results/features/{image_id}_features.csv
    """
    project_root = get_project_root()

    # Plot histogram for visual check
    plot_crystallite_size_distribution(features_df, image_id=image_id, metadata=metadata)

    # Compute summary row
    summary_df = compute_crystallinity_metrics(
        feature_df=features_df,
        image_shape=image_shape,
        image_id=image_id,
        metadata=metadata
    )

    summary_df["segmentation_quality"] = quality_weight
    summary_df["notes"] = notes

    # Save summary to cumulative CSV
    output_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, output_csv_name)

    if os.path.exists(output_csv_path):
        existing = pd.read_csv(output_csv_path)
        existing = existing[existing["image_id"] != image_id]
        updated = pd.concat([existing, summary_df], ignore_index=True)
    else:
        updated = summary_df

    updated.to_csv(output_csv_path, index=False)
    print(f"✔ Summary for {image_id} saved to:\n{output_csv_path}")

    # Save region features as CSV
    features_dir = os.path.join(project_root, "results", "features")
    os.makedirs(features_dir, exist_ok=True)

    features_path = os.path.join(features_dir, f"{image_id}_features.csv")
    features_df.to_csv(features_path, index=False)
