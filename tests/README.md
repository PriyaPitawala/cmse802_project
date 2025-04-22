# Unit Tests (`/tests/`)

This directory contains unit tests to validate the correctness of the project’s implementation.

## Naming Convention
- `test_(module_name).py` for testing `module_name.py` files inside the `src/` subfolder.

## Notes
- `data/` contains sample image for testing image_loader.py and an image with an actual sacle bar for testing image_with_scale_bar.py
- `output/` contains saved debugging outputs for the unit tests.
- `test_preprocessing.py` tests the legacy processing modules `preprocessing_old.py` and `preprocessing.py` inside `src/preprocessing/`.
- `test_segmentation.py` tests the `src/segmentation/watershed_segmentation.py` module.

## Running Tests
To run all tests, use:
```powershell (or bash)
$env:PYTHONPATH="src"; pytest tests/test_test_(module_name).py



