import os
import pandas as pd
from crystallinity_quantification.crystallinity_analysis import compute_crystallinity_metrics
from crystallinity_quantification.visualize_crystallinity import plot_crystallite_size_distribution

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
    notes: str = "",  # New notes parameter
    output_csv_name: str = "crystallinity_summary_all_images.csv"
):


    """
    Plots histogram and appends crystallinity summary for a single image to a global CSV.
    If the image_id already exists in the CSV, it will be overwritten.

    Parameters:
    - image_id (str): Unique ID for the sample/image.
    - features_df (pd.DataFrame): Extracted region features.
    - image_shape (tuple): Shape of the raw image (height, width).
    - metadata (dict): Dict with 'thickness', 'light_intensity', 'photoabsorber' keys.
    - output_csv_name (str): Filename for the cumulative CSV (within cmse802_project/results/tables).
    """

    project_root = get_project_root()
    output_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    output_csv_path = os.path.join(output_dir, output_csv_name)

    # Plot histogram
    plot_crystallite_size_distribution(features_df, image_id=image_id, metadata=metadata)

    # Compute new summary row
    summary_df = compute_crystallinity_metrics(
        feature_df=features_df,
        image_shape=image_shape,
        image_id=image_id,
        metadata=metadata
    )

    # Attach segmentation quality score
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

    features_dir = os.path.join(project_root, "results", "features")
    os.makedirs(features_dir, exist_ok=True)

    features_path = os.path.join(features_dir, f"{image_id}_features.csv")
    features_df.to_csv(features_path, index=False)

