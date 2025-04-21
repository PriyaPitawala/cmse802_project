"""
Unit tests for group_stats.py using pytest.

This test suite validates core statistical utilities used for summarizing crystallinity
properties grouped by experimental parameters. These include weighted percentile
calculations, weighted medians, interquartile ranges (IQRs), and percentile ranges for
crystallite size and percent crystallinity.

Test Scope:
-----------
1. weighted_percentile:
   - Validates basic uniform weights and uneven weighted percentiles.

2. compute_weighted_median_iqr:
   - Confirms group-level weighted medians and IQRs for crystallite size.
   - Uses mock CSV input with segmentation weights.

3. compute_weighted_crystallinity_percentile_range:
   - Verifies output of weighted percent crystallinity stats over a percentile range.
   - Supports custom lower/upper percentile bounds.

Fixtures:
---------
- mock_summary_csv: Creates a temporary CSV simulating summary results
with grouped data.

Run with:
---------
    pytest tests/test_group_stats.py

Note:
-----
This module focuses on statistical correctness and does not include
plotting functionality.

#Author: Priyangika Pitawala
#Date: April 2025
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

from regression import group_stats

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def test_weighted_percentile_simple():
    data = [10, 20, 30, 40]
    weights = [1, 1, 1, 1]
    result = group_stats.weighted_percentile(data, weights, 0.5)
    assert result == 20.0  # not 25


def test_weighted_percentile_uneven():
    data = [10, 20, 30, 40]
    weights = [1, 2, 3, 4]
    result = group_stats.weighted_percentile(data, weights, 0.5)
    assert np.isclose(result, 26.666666666666668)


@pytest.fixture
def mock_summary_csv(tmp_path):
    df = pd.DataFrame({
        "light_intensity": [10, 10, 20],
        "thickness": [100, 100, 100],
        "photoabsorber": [0.01, 0.01, 0.01],
        "median_diameter": [10, 20, 30],
        "percent_crystallinity": [45, 55, 60],
        "segmentation_quality": [1.0, 0.5, 1.0]
    })
    csv_path = tmp_path / "summary.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_compute_weighted_median_iqr(mock_summary_csv):
    df = group_stats.compute_weighted_median_iqr(
        summary_csv_path=str(mock_summary_csv),
        photoabsorber=0.01,
        group_keys=["light_intensity", "thickness"]
    )
    assert not df.empty
    assert "weighted_median" in df.columns
    assert "iqr_lower" in df.columns
    assert "iqr_upper" in df.columns


def test_compute_weighted_crystallinity_percentile_range(mock_summary_csv):
    df = group_stats.compute_weighted_crystallinity_percentile_range(
        summary_csv_path=str(mock_summary_csv),
        photoabsorber=0.01,
        group_keys=["light_intensity", "thickness"],
        lower_pct=0.10,
        upper_pct=0.90
    )
    assert not df.empty
    assert "weighted_crystallinity_median" in df.columns
    assert "crystallinity_iqr_lower" in df.columns
    assert "crystallinity_iqr_upper" in df.columns
