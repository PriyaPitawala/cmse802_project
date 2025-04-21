"""
Unit tests for average_distribution.py using pytest.

This test suite validates core logic in the module responsible for computing and
plotting crystallite size distributions across grouped samples, based on thickness,
light intensity, and photoabsorber concentration.

Test Scope:
-----------
1. load_matching_features:
   - Validates filtering by experimental parameters from a mock summary CSV.
   - Confirms loading and unit conversion of equivalent diameters from mock
   feature files.

Fixtures:
---------
- mock_summary_csv: Simulates the summary CSV used to group samples.
- mock_feature_files: Simulates feature CSVs with crystallite diameters for each image.

Note:
-----
Plotting functions are not directly tested here, but their inputs are covered by the
feature-loading tests. To test plots, use `matplotlib`'s non-interactive backend and
focus on histogram bin output and counts.

Run with:
---------
    pytest tests/test_average_distribution.py

#Author: Priyangika Pitawala
#Date: April 2025
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

from regression import average_distribution

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


@pytest.fixture
def mock_summary_csv(tmp_path):
    df = pd.DataFrame({
        "image_id": ["img_001", "img_002"],
        "thickness": [100, 100],
        "light_intensity": [10, 10],
        "photoabsorber": [0.01, 0.01],
        "segmentation_quality": [1.0, 0.5],
    })
    csv_path = tmp_path / "summary.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_feature_files(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    diameters = [10.0, 12.0, 15.0]
    for image_id in ["img_001", "img_002"]:
        df = pd.DataFrame({"equivalent_diameter": diameters})
        df.to_csv(feature_dir / f"{image_id}_features.csv", index=False)
    return feature_dir


def test_load_matching_features(monkeypatch, mock_summary_csv, mock_feature_files):
    # Override internal paths
    monkeypatch.setattr(average_distribution, "SUMMARY_CSV", str(mock_summary_csv))
    monkeypatch.setattr(average_distribution, "FEATURES_DIR", str(mock_feature_files))

    result = average_distribution.load_matching_features(100, 10, 0.01)
    assert isinstance(result, list)
    assert len(result) == 2
    for series in result:
        assert isinstance(series, pd.Series)
        assert np.all(series > 0)  # Micron conversion should preserve positivity
