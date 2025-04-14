import os
import matplotlib.pyplot as plt

# Aesthetics and style maps
marker_map = ['o', 's', '^', '*', 'D', 'v']
color_map = ['blue', 'red', 'black', 'green', 'orange', 'cyan']

# Global Matplotlib style
plt.rc('font', size=8)
plt.rc('axes', linewidth=2)
plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)
plt.rc('legend', fontsize=22)


def plot_group_trend(df, x_col, y_col, group_col, x_label, y_label, output_path,
                     yerr_lower_col=None, yerr_upper_col=None, ylim=None, xlim=None, sort_x=True):
    """
    Plots grouped trend with asymmetric error bars (e.g., IQR), grouped by a categorical variable.

    Parameters:
    - df (pd.DataFrame): DataFrame with aggregated group-level statistics
    - x_col (str): Column for x-axis (e.g., light_intensity)
    - y_col (str): Column for y-axis (e.g., weighted_median)
    - group_col (str): Column to group and color by (e.g., thickness)
    - x_label (str), y_label (str): Axis labels
    - output_path (str): Path to save the figure
    - yerr_lower_col (str or None): Column name for lower error (asymmetric)
    - yerr_upper_col (str or None): Column name for upper error
    - ylim (tuple or None), xlim (tuple or None): Axis limits
    - sort_x (bool): Whether to sort x-axis within groups
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
            yerr_lower = group_data[yerr_lower_col].values
            yerr_upper = group_data[yerr_upper_col].values
            yerr = [yerr_lower, yerr_upper]
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
            label=f"{group_value} µm"
        )

    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)

    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim(bottom=0)

    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim(left=0)

    legend = plt.legend(title="Layer Thickness", loc='lower left', title_fontsize=22)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)

    plt.tick_params(axis='both', which='major', length=10, width=2)
    plt.tick_params(axis='both', which='minor', length=7, width=1.5)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    ax.set_facecolor('white')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Plot saved to {output_path}")
