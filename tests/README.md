# Unit Tests (`/tests/`)

This directory contains unit tests to validate the correctness of the project’s implementation.

## Contents
- `test_data_loading.py` - Tests for `image_loader.py`.
- `test_image_scale.py` - Tests for `image_with_scale_bar.py`.
- `test_preprocessing.py` - Tests for `preprocessing.py` and `preprocessing_old.py`.
- `test_segmentation.py` - Tests for `watershed_segmentation.py`.

## Notes
- `data/` contains sample image for testing image_loader.py and an image with an actual sacle bar for testing image_with_scale_bar.py
- `output/` contains saved debugging outputs for the unit tests.
- `preprocessing_old.py` contains the functions that can sucessfully compute markers for segmenting Maltese crosses.
- `preprocessing.py` is undergoing debugging to attempt segmenting spherulites.
- `test_segmentation.py` computes the markers for segmentation using the sucessful proeprocessing module version (currently its is `preprocessing.py`).

## Running Tests
To run all tests, use:
```powershell (or bash)
python -m unittest discover -s tests



