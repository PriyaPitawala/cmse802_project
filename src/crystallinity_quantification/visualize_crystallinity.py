"""
visualize_crystallinity.py

This module provides visualization tools for crystallite morphology extracted from
segmented polarized optical microscope (POM) images.

Main Feature:
-------------
- plot_crystallite_size_distribution():
    Generates a histogram of crystallite equivalent diameters (in microns) from a
    segmented image and displays metadata in the title.

Assumptions:
------------
- Equivalent diameters are provided in pixel units.
- Pixel-to-micron scale is defined globally via MICRONS_PER_PIXEL.

Usage:
------
Used within the batch analysis pipeline to confirm distribution characteristics for
individual images and experimental conditions.

Author: Priyangika Pitawala
Date: April 2025
"""

import matplotlib.pyplot as plt
import pandas as pd

# Global scale factor: 60 μm scale bar corresponds to 137 pixels
MICRONS_PER_PIXEL = 60 / 137  # ≈ 0.438 μm/px


def plot_crystallite_size_distribution(
    feature_df: pd.DataFrame,
    image_id: str,
    metadata: dict,
    bin_width_um: float = 2.0,
):
    """
    Plots a histogram of crystallite sizes (in microns) using equivalent diameter data.
    Metadata is embedded in the figure title.

    Parameters:
    -----------
    feature_df : pd.DataFrame
        DataFrame containing at least a column 'equivalent_diameter' (in pixels).

    image_id : str
        Unique identifier for the image; used in plot title.

    metadata : dict
        Dictionary of experimental metadata. Expected keys:
        - 'light_intensity' (in mW/cm²)
        - 'thickness' (in µm)
        - 'photoabsorber' (in wt%)
        If any key is missing, 'N/A' will be displayed.

    bin_width_um : float, optional (default=2.0)
        Width of histogram bins in microns.

    Raises:
    -------
    ValueError
        If 'equivalent_diameter' is not present in the input DataFrame.

    Displays:
    ---------
    - A histogram plot via matplotlib with:
        - Micron-scale x-axis
        - Metadata-annotated title
        - Grid and tight layout formatting
    """
    if "equivalent_diameter" not in feature_df.columns:
        raise ValueError("Feature DataFrame must include 'equivalent_diameter'.")

    # Convert diameters from pixels to microns
    diameters_um = feature_df["equivalent_diameter"].dropna() * MICRONS_PER_PIXEL

    if diameters_um.empty:
        print(f"[INFO] No crystallite data to plot for image: {image_id}")
        return

    # Bin calculation using range and bin width
    bin_range = diameters_um.max() - diameters_um.min()
    bins = int(bin_range // bin_width_um) or 10  # Ensure at least 10 bins

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(
        diameters_um,
        bins=bins,
        color="steelblue",
        edgecolor="black",
        alpha=0.85,
    )

    # Construct titles
    title = f"Crystallite Size Distribution – {image_id}"
    subtitle = (
        f"Light Intensity: {metadata.get('light_intensity', 'N/A')} mW/cm², "
        f"Thickness: {metadata.get('thickness', 'N/A')} µm, "
        f"Photoabsorber: {metadata.get('photoabsorber', 'N/A')} wt%"
    )

    # Apply figure styling
    plt.title(f"{title}\n{subtitle}", fontsize=12)
    plt.xlabel("Equivalent Diameter (µm)")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
