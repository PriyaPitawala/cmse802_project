import matplotlib.pyplot as plt
import pandas as pd
import os

# Define scale
MICRONS_PER_PIXEL = 60 / 137  # ≈ 0.438

def plot_crystallite_size_distribution(
    feature_df: pd.DataFrame,
    image_id: str,
    metadata: dict,
    bin_width_um: float = 2.0
):
    """
    Plots a histogram of crystallite sizes in microns, including sample metadata in the title.

    Parameters:
    - feature_df (pd.DataFrame): DataFrame containing 'equivalent_diameter' in pixels.
    - image_id (str): Unique identifier for the image (used in title and filename).
    - metadata (dict): Dictionary with keys like 'light_intensity', 'thickness', 'photoabsorber'.
    - bin_width_um (float): Bin width for histogram in microns.
    """
    if "equivalent_diameter" not in feature_df.columns:
        raise ValueError("Feature DataFrame must include 'equivalent_diameter'.")

    diameters_um = feature_df["equivalent_diameter"].dropna() * MICRONS_PER_PIXEL
    if diameters_um.empty:
        print(f"No crystallite data to plot for {image_id}.")
        return

    bins = int((diameters_um.max() - diameters_um.min()) // bin_width_um) or 10

    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist(diameters_um, bins=bins, color='steelblue', edgecolor='black', alpha=0.85)

    title = f"Crystallite Size Distribution – {image_id}"
    subtitle = (
        f"Light Intensity: {metadata.get('light_intensity', 'N/A')} mW/cm², "
        f"Thickness: {metadata.get('thickness', 'N/A')} µm, "
        f"Photoabsorber: {metadata.get('photoabsorber', 'N/A')} wt%"
    )

    plt.title(f"{title}\n{subtitle}", fontsize=12)
    plt.xlabel("Equivalent Diameter (µm)")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
