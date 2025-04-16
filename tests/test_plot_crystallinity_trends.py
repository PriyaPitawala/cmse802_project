"""
Unit tests for plot_crystallinity_trends.py using pytest.

Tests:
------
- Ensures plots are created and saved correctly
- Verifies error bar handling and group sorting logic

Uses matplotlib's Agg backend for headless rendering.

Author: Priyangika Pitawala
Date: April 2025
"""

import pandas as pd
import matplotlib
import pytest

from visualization.plot_crystallinity_trends import plot_group_trend

matplotlib.use("Agg")  # Headless backend for testing


@pytest.fixture
def sample_df():
    """Create a minimal DataFrame for grouped plotting."""
    return pd.DataFrame({
        "light_intensity": [5, 10, 5, 10],
        "weighted_median": [4.0, 5.0, 6.0, 7.0],
        "thickness": [100, 100, 200, 200],
        "iqr_low": [3.5, 4.5, 5.5, 6.5],
        "iqr_high": [4.5, 5.5, 6.5, 7.5],
    })


def test_plot_saves_file(sample_df, tmp_path):
    """Ensure the plot is saved to the expected location."""
    output_path = tmp_path / "trend_plot.png"
    plot_group_trend(
        df=sample_df,
        x_col="light_intensity",
        y_col="weighted_median",
        group_col="thickness",
        x_label="Light Intensity (mW/cm²)",
        y_label="Median Diameter (µm)",
        output_path=str(output_path),
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_handles_asymmetric_error(sample_df, tmp_path):
    """Check if asymmetric error bars are accepted and plot is saved."""
    output_path = tmp_path / "asymmetric_plot.png"
    plot_group_trend(
        df=sample_df,
        x_col="light_intensity",
        y_col="weighted_median",
        group_col="thickness",
        x_label="Light Intensity (mW/cm²)",
        y_label="Median Diameter (µm)",
        yerr_lower_col="iqr_low",
        yerr_upper_col="iqr_high",
        output_path=str(output_path),
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_group_sorting(tmp_path):
    """Ensure sorting logic handles unsorted input."""
    df_unsorted = pd.DataFrame({
        "light_intensity": [10, 5, 10, 5],
        "weighted_median": [5.0, 4.0, 7.0, 6.0],
        "thickness": [100, 100, 200, 200],
    })
    output_path = tmp_path / "sorted_plot.png"
    plot_group_trend(
        df=df_unsorted,
        x_col="light_intensity",
        y_col="weighted_median",
        group_col="thickness",
        x_label="Light Intensity (mW/cm²)",
        y_label="Median Diameter (µm)",
        output_path=str(output_path),
        sort_x=True,
    )
    assert output_path.exists()


def test_plot_no_crash_on_minimal_input(tmp_path):
    """Test function doesn’t crash with one group and one point."""
    df = pd.DataFrame({
        "light_intensity": [5],
        "weighted_median": [4.5],
        "thickness": [100],
    })
    output_path = tmp_path / "minimal_plot.png"
    plot_group_trend(
        df=df,
        x_col="light_intensity",
        y_col="weighted_median",
        group_col="thickness",
        x_label="X",
        y_label="Y",
        output_path=str(output_path),
    )
    assert output_path.exists()
