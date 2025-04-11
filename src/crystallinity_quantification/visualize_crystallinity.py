import matplotlib.pyplot as plt
import pandas as pd

# Define your pixel-to-micron scale
MICRONS_PER_PIXEL = 60 / 137  # ≈ 0.438

def plot_crystallite_size_distribution(feature_df: pd.DataFrame, image_id: str, bin_width_um: float = 2.0):
    """
    Plots a histogram of crystallite equivalent diameters (in microns) for a single image.

    Parameters:
    - feature_df (pd.DataFrame): DataFrame with 'equivalent_diameter' in pixels.
    - image_id (str): Identifier for the image (used in the plot title).
    - bin_width_um (float): Width of bins in microns.
    """
    if "equivalent_diameter" not in feature_df.columns:
        raise ValueError("Feature DataFrame must include 'equivalent_diameter' column.")

    # Convert from pixels to microns
    diameters_um = feature_df["equivalent_diameter"].dropna() * MICRONS_PER_PIXEL

    if diameters_um.empty:
        print(f"No crystallite data to plot for {image_id}.")
        return

    bins = int((diameters_um.max() - diameters_um.min()) // bin_width_um) or 10

    plt.figure(figsize=(8, 6))
    plt.hist(diameters_um, bins=bins, color='steelblue', edgecolor='black', alpha=0.85)
    plt.title(f"Crystallite Size Distribution - {image_id}")
    plt.xlabel("Equivalent Diameter (µm)")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
