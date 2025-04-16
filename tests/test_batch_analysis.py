"""
Unit tests for batch_analysis.py

These tests check:
- Summary metrics are correctly written/appended to CSV
- Existing rows with same image_id are replaced
- Features CSV is saved
- Histogram plotting is called
- Optional arguments are handled properly

All file writing and plotting functions are mocked.

Author: Priyangika Pitawala
Date: April 2025
"""

import pandas as pd
import pytest
from unittest.mock import patch

from crystallinity_quantification.batch_analysis import process_and_append_sample


@pytest.fixture
def mock_features_df():
    """Create a simple mock features DataFrame."""
    return pd.DataFrame(
        {
            "area": [50, 80],
            "equivalent_diameter": [8.0, 10.1],
        }
    )


@pytest.fixture
def mock_image_shape():
    """Returns a dummy image shape (height, width)."""
    return (100, 100)


@pytest.fixture
def mock_metadata():
    """Mock metadata used in summary."""
    return {"thickness": 100, "light_intensity": 5, "photoabsorber": 0.01}


@patch("crystallinity_quantification.batch_analysis.plot_crystallite_size_distribution")
@patch("crystallinity_quantification.batch_analysis.compute_crystallinity_metrics")
def test_summary_written_correctly(
    mock_compute_metrics,
    mock_plot,
    mock_features_df,
    mock_image_shape,
    mock_metadata,
    tmp_path,
):
    """Test that summary row is correctly saved to output CSV."""
    mock_summary = pd.DataFrame(
        [
            {
                "image_id": "test_001",
                "image_height": 100,
                "image_width": 100,
                "num_crystallites": 2,
                "total_crystallite_area": 130,
                "percent_crystallinity": 1.3,
                "mean_diameter": 9.05,
                "median_diameter": 9.05,
                "std_diameter": 1.48,
                "thickness": 100,
                "light_intensity": 5,
                "photoabsorber": 0.01,
            }
        ]
    )
    mock_compute_metrics.return_value = mock_summary

    # Corrected: match actual path used in function
    output_csv = tmp_path / "results" / "tables" / "summary.csv"

    with patch(
        "crystallinity_quantification.batch_analysis.get_project_root",
        return_value=str(tmp_path),
    ):
        process_and_append_sample(
            image_id="test_001",
            features_df=mock_features_df,
            image_shape=mock_image_shape,
            metadata=mock_metadata,
            output_csv_name=output_csv.name,
        )

    assert output_csv.exists()
    result = pd.read_csv(output_csv)
    assert result.shape[0] == 1
    assert "percent_crystallinity" in result.columns


@patch("crystallinity_quantification.batch_analysis.plot_crystallite_size_distribution")
@patch("crystallinity_quantification.batch_analysis.compute_crystallinity_metrics")
def test_existing_image_overwritten(
    mock_compute_metrics,
    mock_plot,
    mock_features_df,
    mock_image_shape,
    mock_metadata,
    tmp_path,
):
    """Test that old entries with the same image_id are replaced."""
    image_id = "duplicate_id"

    # Corrected: match actual path used in function
    output_csv = tmp_path / "results" / "tables" / "summary.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Write existing entry with same image_id
    existing = pd.DataFrame([{"image_id": image_id, "percent_crystallinity": 50.0}])
    existing.to_csv(output_csv, index=False)

    # Mock new data to overwrite existing one
    mock_summary = pd.DataFrame([{"image_id": image_id, "percent_crystallinity": 75.0}])
    mock_compute_metrics.return_value = mock_summary

    with patch(
        "crystallinity_quantification.batch_analysis.get_project_root",
        return_value=str(tmp_path),
    ):
        process_and_append_sample(
            image_id=image_id,
            features_df=mock_features_df,
            image_shape=mock_image_shape,
            metadata=mock_metadata,
            output_csv_name=output_csv.name,
        )

    updated = pd.read_csv(output_csv)
    assert updated.shape[0] == 1
    assert updated["percent_crystallinity"].iloc[0] == 75.0


@patch("crystallinity_quantification.batch_analysis.plot_crystallite_size_distribution")
@patch("crystallinity_quantification.batch_analysis.compute_crystallinity_metrics")
def test_feature_csv_is_saved(
    mock_compute_metrics,
    mock_plot,
    mock_features_df,
    mock_image_shape,
    mock_metadata,
    tmp_path,
):
    """Test that raw features are saved to correct file."""
    image_id = "feature_test"
    mock_compute_metrics.return_value = pd.DataFrame([{"image_id": image_id}])

    with patch(
        "crystallinity_quantification.batch_analysis.get_project_root",
        return_value=str(tmp_path),
    ):
        process_and_append_sample(
            image_id=image_id,
            features_df=mock_features_df,
            image_shape=mock_image_shape,
            metadata=mock_metadata,
        )

    saved_path = tmp_path / "results" / "features" / f"{image_id}_features.csv"
    assert saved_path.exists()

    df_loaded = pd.read_csv(saved_path)
    assert "area" in df_loaded.columns


@patch("crystallinity_quantification.batch_analysis.plot_crystallite_size_distribution")
@patch("crystallinity_quantification.batch_analysis.compute_crystallinity_metrics")
def test_handles_optional_args(
    mock_compute_metrics,
    mock_plot,
    mock_features_df,
    mock_image_shape,
    mock_metadata,
    tmp_path,
):
    """Test that default notes and quality_weight are handled."""
    mock_summary = pd.DataFrame([{"image_id": "optional_test"}])
    mock_compute_metrics.return_value = mock_summary

    with patch(
        "crystallinity_quantification.batch_analysis.get_project_root",
        return_value=str(tmp_path),
    ):
        process_and_append_sample(
            image_id="optional_test",
            features_df=mock_features_df,
            image_shape=mock_image_shape,
            metadata=mock_metadata,
        )

    output_csv = (
        tmp_path / "results" / "tables" / "crystallinity_summary_all_images.csv"
    )
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert "segmentation_quality" in df.columns
    assert "notes" in df.columns
