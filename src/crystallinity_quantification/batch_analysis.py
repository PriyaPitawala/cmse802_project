"""
Module: batch_analysis.py

This module handles batch-level crystallinity analysis for polarized optical microscopy (POM) images.
It integrates feature extraction results, computes percent crystallinity and related metrics,
generates histograms, and appends the results to a global summary CSV.

Typical usage:
- After extracting features from a segmented image, use `process_and_append_sample()` to update
  the group dataset and save image-specific metrics for statistical analysis.

Dependencies:
- pandas
- os
- compute_crystallinity_metrics() from crystallinity_analysis.py
- plot_crystallite_size_distribution(s) from visualize_crystallinity.py

# Author: Priyangika Pitawala
# Date: April 2025
"""

import os
import pandas as pd
from crystallinity_quantification.crystallinity_analysis import (
    compute_crystallinity_metrics,
)
from crystallinity_quantification.visualize_crystallinity import (
    plot_crystallite_size_distribution,
)


def get_project_root() -> str:
    """
    Returns the absolute path to the root of the project by going two levels up from the current working directory.
    Assumes you're running from `notebooks/exploratory/` or similar inside project.
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
    Processes and saves crystallinity analysis results for a single image.

    This function:
    - Generates and saves a histogram of relative crystallite size distribution.
    - Computes percent crystallinity and related summary metrics.
    - Appends or updates the summary results in a cumulative CSV.
    - Saves raw region features to a dedicated directory.

    Parameters:
    ----------
    image_id : str
        Unique identifier for the image/sample (e.g., filename without extension).

    features_df : pd.DataFrame
        DataFrame containing features of segmented crystallites (e.g., area, diameter).

    image_shape : tuple
        Tuple representing the original image shape as (height, width).

    metadata : dict
        Dictionary containing sample-specific experimental parameters.
        Required keys: 'thickness', 'light_intensity', 'photoabsorber'.

    quality_weight : float, optional
        User-assigned score (0–1) reflecting segmentation quality for this image. Default is 1.0.

    notes : str, optional
        Optional text annotation for tracking manual notes or observations. Default is "".

    output_csv_name : str, optional
        Filename for the cumulative summary table stored under `results/tables/`.

    Outputs:
    -------
    - Updates `results/tables/{output_csv_name}` with new crystallinity metrics.
    - Saves histogram figure to default plotting path.
    - Saves extracted features to `results/features/{image_id}_features.csv`.
    """

    # Defining the path to the project root and the output directory
    # for the summary CSV file.
    project_root = get_project_root()
    output_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    output_csv_path = os.path.join(output_dir, output_csv_name)

    # Plot histogram
    plot_crystallite_size_distribution(
        features_df, image_id=image_id, metadata=metadata
    )

    # Compute new summary row
    summary_df = compute_crystallinity_metrics(
        feature_df=features_df,
        image_shape=image_shape,
        image_id=image_id,
        metadata=metadata,
    )

    # Attach segmentation quality score from a scale of 0 to 1.
    # 0 = poor, 1 = excellent.
    # This is a user-defined score that reflects the segmentation quality.
    summary_df["segmentation_quality"] = quality_weight
    summary_df["notes"] = notes

    # Append or update
    if os.path.exists(output_csv_path):
        existing = pd.read_csv(output_csv_path)
        existing = existing[existing["image_id"] != image_id]
        updated = pd.concat([existing, summary_df], ignore_index=True)
    else:
        updated = summary_df

    updated.to_csv(output_csv_path, index=False)
    print(f"✔ Summary for {image_id} saved to:\n{output_csv_path}")

    # Save features to a separate CSV file
    # for computing average distributions for each sample group.
    features_dir = os.path.join(project_root, "results", "features")
    os.makedirs(features_dir, exist_ok=True)

    features_path = os.path.join(features_dir, f"{image_id}_features.csv")
    features_df.to_csv(features_path, index=False)


if __name__ == "__main__":
    print("This module is designed to be imported, not run directly.")
