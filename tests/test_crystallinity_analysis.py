"""
Unit tests for crystallinity_analysis.py

Tests compute_crystallinity_metrics() to ensure:
- Correct output shape (with and without metadata)
- Accurate percent crystallinity calculation
- Correct diameter statistics
- Proper metadata inclusion

Author: Priyangika Pitawala
Date: April 2025
"""

import pytest
import pandas as pd
from crystallinity_quantification.crystallinity_analysis import (
    compute_crystallinity_metrics,
)


@pytest.fixture
def sample_feature_df():
    """Mock region features for 5 crystallites."""
    return pd.DataFrame(
        {
            "area": [50, 80, 100, 40, 30],
            "equivalent_diameter": [8.0, 10.1, 12.5, 7.2, 6.4],
        }
    )


@pytest.fixture
def sample_image_shape():
    """Returns mock image shape (height, width)."""
    return (100, 100)


@pytest.fixture
def sample_metadata():
    """Returns metadata dictionary."""
    return {"thickness": 100, "light_intensity": 5, "photoabsorber": 0.01}


def test_output_shape_without_metadata(sample_feature_df, sample_image_shape):
    """Test output is a single-row DataFrame with at least base fields (no metadata)."""
    result = compute_crystallinity_metrics(
        sample_feature_df, sample_image_shape, "image_001"
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 1
    assert result.shape[1] >= 9


def test_output_shape_with_metadata(
    sample_feature_df, sample_image_shape, sample_metadata
):
    """Test output shape when metadata is included (should add 3 fields)."""
    result = compute_crystallinity_metrics(
        sample_feature_df, sample_image_shape, "image_001", metadata=sample_metadata
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 12)  # 9 base + 3 metadata fields


def test_percent_crystallinity(sample_feature_df, sample_image_shape):
    """Test correct computation of percent crystallinity."""
    total_area = sum(sample_feature_df["area"])
    expected_percent = (
        100 * total_area / (sample_image_shape[0] * sample_image_shape[1])
    )
    result = compute_crystallinity_metrics(
        sample_feature_df, sample_image_shape, "image_001"
    )
    assert (
        pytest.approx(result["percent_crystallinity"].iloc[0], 0.01) == expected_percent
    )


def test_diameter_statistics(sample_feature_df, sample_image_shape):
    """Test diameter statistics: mean, median, std."""
    result = compute_crystallinity_metrics(
        sample_feature_df, sample_image_shape, "image_001"
    )

    assert (
        pytest.approx(result["mean_diameter"].iloc[0], 0.01)
        == sample_feature_df["equivalent_diameter"].mean()
    )
    assert (
        pytest.approx(result["median_diameter"].iloc[0], 0.01)
        == sample_feature_df["equivalent_diameter"].median()
    )
    assert pytest.approx(result["std_diameter"].iloc[0], 0.01) == (
        sample_feature_df["equivalent_diameter"].std()
    )


def test_metadata_inclusion(sample_feature_df, sample_image_shape, sample_metadata):
    """Test whether metadata keys are included in the output."""
    result = compute_crystallinity_metrics(
        sample_feature_df, sample_image_shape, "image_001", metadata=sample_metadata
    )
    for key in sample_metadata:
        assert key in result.columns
        assert result[key].iloc[0] == sample_metadata[key]
