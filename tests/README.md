# Unit Tests (`/tests/`)

This directory contains unit tests to validate the correctness of the project’s implementation.

## Contents
- `test_data_loading.py` - Tests for `image_loader.py`.
- `test_image_scale.py` - Tests for `image_with_scale_bar.py`.
- `test_preprocessing.py` - Tests for `preprocessing.py`.
- `test_segmentation.py` - Tests for `watershed_segmentation.py`.

## Notes
- `data/` contains sample image for testing image_loader.py and an image with an actual sacle bar for testing image_with_scale_bar.py
- `output/` contains saved debugging outputs for the unit tests.

## Running Tests
To run all tests, use:
```powershell (or bash)
python -m unittest discover -s tests



