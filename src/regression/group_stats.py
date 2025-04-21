"""
group_stats.py

This module provides functions to compute weighted statistical summaries of
crystallinity data grouped by experimental parameters such as light intensity
and thickness. These summaries include weighted medians and percentile ranges
of crystallite sizes and percent crystallinity.

Functionality:
--------------
1. weighted_percentile:
   - Computes a weighted percentile from a 1D data array using associated weights.

2. compute_weighted_median_iqr:
   - Calculates the weighted median and interquartile range (IQR) of a target column
     (e.g., crystallite diameter) grouped by user-defined keys.
   - Useful for visualizing central tendency and dispersion across
   processing conditions.

3. compute_weighted_crystallinity_percentile_range:
   - Similar to the above, but designed for percent crystallinity values.
   - Computes a user-defined percentile range (default 10%–90%) and
   distances from the median.

Parameters:
-----------
- summary_csv_path (str): Path to the crystallinity summary CSV file.
- photoabsorber (float): Filter criterion for selecting relevant sample groups.
- group_keys (list): Grouping columns in the summary CSV
(e.g., ['light_intensity', 'thickness']).
- value_col (str): Column name whose statistics are computed (e.g., 'median_diameter').
- weight_col (str): Column name used for weighting (e.g., 'segmentation_quality').
- lower_pct, upper_pct (float): Percentiles for crystallinity range computation.

Returns:
--------
- pd.DataFrame: Group-level statistics including weighted medians
and asymmetrical error margins.

Example Usage:
--------------
```python
compute_weighted_median_iqr(
    summary_csv_path="results/tables/crystallinity_summary_all_images.csv",
    photoabsorber=0.01
)

compute_weighted_crystallinity_percentile_range(
    summary_csv_path="results/tables/crystallinity_summary_all_images.csv",
    photoabsorber=0.01
)
```

Dependencies:
-------------
- pandas
- numpy

#Author: Priyangika Pitawala
#Date: April 2025
"""

import pandas as pd
import numpy as np


def weighted_percentile(data, weights, percentile):
    """
    Compute the weighted percentile of a 1D array.
    """
    data, weights = np.array(data), np.array(weights)
    sorter = np.argsort(data)
    data, weights = data[sorter], weights[sorter]
    cum_weights = np.cumsum(weights)
    total_weight = cum_weights[-1]
    return np.interp(percentile * total_weight, cum_weights, data)


def compute_weighted_median_iqr(
    summary_csv_path: str,
    photoabsorber: float,
    group_keys: list = ["light_intensity", "thickness"],
    value_col: str = "median_diameter",
    weight_col: str = "segmentation_quality",
) -> pd.DataFrame:
    """
    Computes weighted median and IQR for grouped crystallinity data.

    Returns:
    - DataFrame with group-level weighted median, lower IQR, upper IQR
    """
    summary = pd.read_csv(summary_csv_path)
    summary = summary[summary["photoabsorber"] == photoabsorber].copy()

    grouped = summary.groupby(group_keys)
    group_stats = []

    for group_vals, group in grouped:
        weights = group[weight_col].fillna(1.0)
        values = group[value_col]

        weighted_median = weighted_percentile(values, weights, 0.50)
        lower_q = weighted_percentile(values, weights, 0.0)
        upper_q = weighted_percentile(values, weights, 1.0)

        result = dict(zip(group_keys, group_vals))
        result.update(
            {
                "weighted_median": weighted_median,
                "iqr_lower": weighted_median - lower_q,
                "iqr_upper": upper_q - weighted_median,
            }
        )
        group_stats.append(result)

    return pd.DataFrame(group_stats)


def compute_weighted_crystallinity_percentile_range(
    summary_csv_path: str,
    photoabsorber: float,
    group_keys: list = ["light_intensity", "thickness"],
    value_col: str = "percent_crystallinity",
    weight_col: str = "segmentation_quality",
    lower_pct: float = 0.10,
    upper_pct: float = 0.90,
) -> pd.DataFrame:
    """
    Computes weighted median and percentile range (e.g., 10%–90%)
    for percent crystallinity by group.

    Returns:
    - DataFrame with group-level weighted median, lower, and
    upper percentile distance from the median
    """
    summary = pd.read_csv(summary_csv_path)
    summary = summary[summary["photoabsorber"] == photoabsorber].copy()

    grouped = summary.groupby(group_keys)
    group_stats = []

    for group_vals, group in grouped:
        weights = group[weight_col].fillna(1.0)
        values = group[value_col]

        weighted_median = weighted_percentile(values, weights, 0.50)
        lower_q = weighted_percentile(values, weights, lower_pct)
        upper_q = weighted_percentile(values, weights, upper_pct)

        result = dict(zip(group_keys, group_vals))
        result.update(
            {
                "weighted_crystallinity_median": weighted_median,
                "crystallinity_iqr_lower": weighted_median - lower_q,
                "crystallinity_iqr_upper": upper_q - weighted_median,
            }
        )
        group_stats.append(result)

    return pd.DataFrame(group_stats)
