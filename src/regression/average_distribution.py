import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constants
MICRONS_PER_PIXEL = 60 / 137
SUMMARY_CSV = os.path.abspath(os.path.join(os.getcwd(), "../../results/tables/crystallinity_summary_all_images.csv"))
FEATURES_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../results/features/"))

def load_matching_features(thickness, light_intensity, photoabsorber):
    """
    Load all features_df files for samples with matching experimental parameters.
    """
    summary = pd.read_csv(SUMMARY_CSV)

    # Filter by input criteria
    group = summary[
        (summary["thickness"] == thickness) &
        (summary["light_intensity"] == light_intensity) &
        (summary["photoabsorber"] == photoabsorber)
    ]

    if group.empty:
        print("⚠ No matching samples found.")
        return []

    all_diameters_um = []

    for image_id in group["image_id"]:
        features_path = os.path.join(FEATURES_DIR, f"{image_id}_features.csv")
        if os.path.exists(features_path):
            features_df = pd.read_csv(features_path)
            if "equivalent_diameter" in features_df.columns:
                diameters_um = features_df["equivalent_diameter"].dropna() * MICRONS_PER_PIXEL
                all_diameters_um.append(diameters_um)
            else:
                print(f"⚠ No diameter data in {image_id}")
        else:
            print(f"⚠ Missing features file: {features_path}")

    return all_diameters_um

def plot_average_distribution(all_diameters_um, title="Average Crystallite Size Distribution", bin_width=2.0):
    """
    Plot the average crystallite size distribution by dividing bin counts by the number of samples.

    Parameters:
    - all_diameters_um (List of pd.Series): One Series of diameters (in µm) per sample.
    - title (str): Plot title.
    - bin_width (float): Bin width in microns.
    """
    if not all_diameters_um:
        print("⚠ No data to plot.")
        return

    # Combine all data to get consistent bin edges
    combined = pd.concat(all_diameters_um, ignore_index=True)
    min_val, max_val = combined.min(), combined.max()
    bins = np.arange(min_val, max_val + bin_width, bin_width)

    # Compute normalized histogram per sample
    all_counts = []
    for sample_diameters in all_diameters_um:
        counts, _ = np.histogram(sample_diameters, bins=bins)
        all_counts.append(counts)

    # Stack and average across samples
    counts_matrix = np.stack(all_counts)
    avg_counts = counts_matrix.mean(axis=0)

    # Plot
    plt.figure(figsize=(8, 6))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.bar(bin_centers, avg_counts, width=bin_width*0.9, color='darkorange', edgecolor='black', alpha=0.85)

    plt.title(title)
    plt.xlabel("Equivalent Diameter (µm)")
    plt.ylabel("Average Count per Sample")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0,60)
    plt.ylim(0,350)
    plt.tight_layout()
    plt.show()

def plot_weighted_group_distribution(thickness, light_intensity, photoabsorber, bin_width=2.0):
    """
    Plots the weighted average crystallite size distribution for a group of samples
    with the same thickness, light intensity, and photoabsorber concentration.
    Weights are taken from the 'segmentation_quality' column.

    Parameters:
    - thickness (int or float)
    - light_intensity (int or float)
    - photoabsorber (float)
    - bin_width (float): width of histogram bins in microns
    """
    summary = pd.read_csv(SUMMARY_CSV)

    # Filter by sample group
    group = summary[
        (summary["thickness"] == thickness) &
        (summary["light_intensity"] == light_intensity) &
        (summary["photoabsorber"] == photoabsorber)
    ]

    if group.empty:
        print("⚠ No matching samples found.")
        return

    all_diameters = []
    weights = []

    for _, row in group.iterrows():
        image_id = row["image_id"]
        weight = row.get("segmentation_quality", 1.0)
        features_path = os.path.join(FEATURES_DIR, f"{image_id}_features.csv")

        if os.path.exists(features_path):
            features_df = pd.read_csv(features_path)
            if "equivalent_diameter" in features_df.columns:
                diameters = features_df["equivalent_diameter"].dropna() * MICRONS_PER_PIXEL
                all_diameters.append(diameters)
                weights.append(weight)
            else:
                print(f"⚠ No 'equivalent_diameter' in {image_id}")
        else:
            print(f"⚠ Missing features file for {image_id}")

    if not all_diameters:
        print("⚠ No valid feature data loaded.")
        return

    # Compute common bins
    combined = pd.concat(all_diameters)
    bins = np.arange(combined.min(), combined.max() + bin_width, bin_width)

    # Histogram each sample and apply weighting
    all_counts = []
    for diameters in all_diameters:
        counts, _ = np.histogram(diameters, bins=bins)
        all_counts.append(counts)

    counts_matrix = np.stack(all_counts)

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Weighted average
    weighted_avg_counts = np.average(counts_matrix, axis=0, weights=weights)

    # Plot
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, weighted_avg_counts, width=bin_width*0.9, color='royalblue', edgecolor='black', alpha=0.85)
    plt.title(f"Weighted Avg – {light_intensity} mW/cm², {thickness} µm, {photoabsorber} wt%")
    plt.xlabel("Equivalent Diameter (µm)")
    plt.ylabel("Weighted Average Count")
    plt.xlim(0,60)
    plt.ylim(0,500)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
