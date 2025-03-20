# Jupyter Notebooks (`/notebooks/`)

This directory contains Jupyter notebooks used for data analysis and visualization.

## Contents
- `exploratory/` - Initial exploration of each POM image and crystallinity analysis.
- `final/` - Final analysis of the correlation between percent crystallinity and material properties.

## Notes
- The `single_POM_analysis.ipynb` file is designed to load a raw POM image and control segmentation parameters without modifying `preprocessing.py` and `watershed_segmentation.py`.
- Output of crystallization analysis is saved in a dataframe.
- `single_POM_analysis_old.ipynb` file has the parameters that can successfully segment Maltese crosses. 
- `single_POM_analysis.ipynb` file has the parameters that are an attempt to segment spherulites. However, due to the presence of       multiple layers of spherulites this has proven difficult. This notebook is currently undergoing optimization and debugging.