"""
plot_crystallinity_trends.py

Provides functions to visualize group-level trends in crystallinity metrics
(e.g., median diameter or percent crystallinity) as a function of processing
parameters like light intensity or layer thickness.

Features:
---------
- Plots grouped line charts with asymmetric error bars (e.g., IQR).
- Supports sorting, custom axis limits, and consistent styling for publication.
- Automatically saves the output figure to a specified path.

Intended Use:
-------------
This module is designed for grouped sample analysis and is integrated with
your crystallinity statistics pipeline.

Author: Priyangika Pitawala
Date: April 2025
"""

import os
import matplotlib.pyplot as plt

# Aesthetics and style maps
marker_map = ["o", "s", "^", "*", "D", "v"]
color_map = ["blue", "red", "black", "green", "orange", "cyan"]

# Global Matplotlib style
plt.rc("font", size=8)
plt.rc("axes", linewidth=2)
plt.rc("xtick", labelsize=22)
plt.rc("ytick", labelsize=22)
plt.rc("legend", fontsize=22)


def plot_group_trend(
    df,
    x_col,
    y_col,
    group_col,
    x_label,
    y_label,
    output_path,
    yerr_lower_col=None,
    yerr_upper_col=None,
    ylim=None,
    xlim=None,
    sort_x=True,
):
    """
    Plots a grouped trendline with optional asymmetric error bars (e.g., IQR).

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated DataFrame with group-level statistics.

    x_col : str
        Column name for x-axis values (e.g., 'light_intensity').

    y_col : str
        Column name for y-axis values (e.g., 'median_diameter').

    group_col : str
        Column name to group and color by (e.g., 'thickness').

    x_label : str
        Label for the x-axis.

    y_label : str
        Label for the y-axis.

    output_path : str
        Full path to save the output plot (saved as .png with 300 dpi).

    yerr_lower_col : str, optional
        Column name for lower bounds of asymmetric error bars.

    yerr_upper_col : str, optional
        Column name for upper bounds of asymmetric error bars.

    ylim : tuple, optional
        y-axis limits as (min, max). If None, bottom=0 is enforced.

    xlim : tuple, optional
        x-axis limits as (min, max). If None, left=0 is enforced.

    sort_x : bool, default=True
        Whether to sort x-values within each group before plotting.

    Returns
    -------
    None. Saves the plot to the specified location.
    """
    plt.figure(figsize=(10, 9))

    if sort_x:
        df = df.sort_values(x_col)

    for i, (group_value, group_data) in enumerate(df.groupby(group_col)):
        marker = marker_map[i % len(marker_map)]
        color = color_map[i % len(color_map)]
        x_vals = group_data[x_col].values
        y_vals = group_data[y_col].values

        if yerr_lower_col and yerr_upper_col:
            yerr = [
                group_data[yerr_lower_col].values,
                group_data[yerr_upper_col].values,
            ]
        else:
            yerr = None

        plt.errorbar(
            x_vals,
            y_vals,
            yerr=yerr,
            fmt=marker,
            markersize=10,
            elinewidth=2,
            capsize=5,
            linewidth=0,
            color=color,
            label=f"{group_value} µm",
        )

    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)

    plt.ylim(ylim if ylim else (0, None))
    plt.xlim(xlim if xlim else (0, None))

    legend = plt.legend(title="Layer Thickness", loc="lower left", title_fontsize=22)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.5)

    plt.tick_params(axis="both", which="major", length=10, width=2)
    plt.tick_params(axis="both", which="minor", length=7, width=1.5)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")

    ax.set_facecolor("white")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Plot saved to {output_path}")


def plot_group_boxplot(
    df,
    x_col,
    y_col,
    group_col,
    x_label,
    y_label,
    output_path,
    ylim=None,
    xlim=None,
    order=None,
):
    """
    Creates grouped boxplots using Matplotlib only.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with individual data points.

    x_col : str
        Column name for x-axis grouping (e.g., 'light_intensity').

    y_col : str
        Column name for the y-axis values (e.g., 'crystallite_diameter').

    group_col : str
        Column name to color/separate by (e.g., 'thickness').

    x_label : str
        Label for the x-axis.

    y_label : str
        Label for the y-axis.

    output_path : str
        Full path to save the output plot (saved as .png with 300 dpi).

    ylim : tuple, optional
        y-axis limits as (min, max).

    xlim : tuple, optional
        x-axis limits as (min, max).

    order : list, optional
        Custom sort order for x_col values.

    Returns
    -------
    None
    """
    import numpy as np

    plt.figure(figsize=(10, 9))
    ax = plt.gca()

    # Set consistent order of x-values
    x_vals = sorted(df[x_col].unique()) if order is None else order
    group_vals = sorted(df[group_col].unique())

    # Compute spacing
    width = 0.8  # total width for each group of boxes
    box_width = width / len(group_vals)
    positions = []

    for i, group in enumerate(group_vals):
        group_data = df[df[group_col] == group]
        color = color_map[i % len(color_map)]
        box_pos = []

        for j, x in enumerate(x_vals):
            xpos = j - width / 2 + i * box_width + box_width / 2
            subset = group_data[group_data[x_col] == x][y_col].values

            if len(subset) > 0:
                bp = ax.boxplot(
                    subset,
                    positions=[xpos],
                    widths=box_width * 0.9,
                    patch_artist=True,
                    showfliers=True,
                    boxprops=dict(facecolor=color, color="black", linewidth=2),
                    capprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="o", markersize=5, color="black", alpha=0.5),
                    medianprops=dict(color="black", linewidth=2),
                )
            box_pos.append(xpos)

        positions.append(box_pos)

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(x) for x in x_vals], fontsize=22)

    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)

    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim(bottom=0)
    if xlim:
        plt.xlim(xlim)

    # Custom legend
    handles = [
        plt.Line2D([0], [0], color=color_map[i % len(color_map)], lw=10)
        for i in range(len(group_vals))
    ]
    labels = [f"{g} µm" for g in group_vals]
    legend = ax.legend(
        handles,
        labels,
        title="Layer Thickness",
        loc="lower left",
        title_fontsize=22,
        fontsize=22,
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.5)

    # Style
    ax.tick_params(axis="both", which="major", length=10, width=2)
    ax.tick_params(axis="both", which="minor", length=7, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")
    ax.set_facecolor("white")

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Boxplot saved to {output_path}")


