"""
Unit tests for visualize_crystallinity.py

Tests the behavior of plot_crystallite_size_distribution() under:
- Valid data input
- Missing required column
- Empty equivalent diameter data

All plotting functions (plt.show, plt.hist) are mocked to avoid rendering during test.

Author: Priyangika Pitawala
Date: April 2025
"""

import pytest
import pandas as pd
from unittest.mock import patch

from crystallinity_quantification.visualize_crystallinity import (
    plot_crystallite_size_distribution,
)


@pytest.fixture
def mock_metadata():
    return {
        "light_intensity": 10,
        "thickness": 200,
        "photoabsorber": 0.01,
    }


def test_plot_called_with_valid_data(mock_metadata):
    """Ensure that plt.hist and plt.show are called when valid diameters exist."""
    df = pd.DataFrame({"equivalent_diameter": [10, 15, 20, 25]})

    with (
        patch("matplotlib.pyplot.hist") as mock_hist,
        patch("matplotlib.pyplot.show") as mock_show,
    ):

        plot_crystallite_size_distribution(
            feature_df=df,
            image_id="test_img",
            metadata=mock_metadata,
        )

        assert mock_hist.called
        assert mock_show.called


def test_missing_equivalent_diameter_column_raises():
    """Test that missing column raises ValueError."""
    df = pd.DataFrame({"area": [100, 200]})  # No 'equivalent_diameter'

    with pytest.raises(ValueError, match="must include 'equivalent_diameter'"):
        plot_crystallite_size_distribution(
            feature_df=df,
            image_id="test_missing",
            metadata={},
        )


def test_empty_diameter_data_skips_plot(mock_metadata):
    """Test that no plot is shown when equivalent_diameter column is empty."""
    df = pd.DataFrame({"equivalent_diameter": []})

    with (
        patch("matplotlib.pyplot.hist") as mock_hist,
        patch("matplotlib.pyplot.show") as mock_show,
    ):

        plot_crystallite_size_distribution(
            feature_df=df,
            image_id="empty_img",
            metadata=mock_metadata,
        )

        assert not mock_hist.called
        assert not mock_show.called
