# Jupyter Notebooks (`/notebooks/`)

This directory contains Jupyter notebooks used for data analysis and visualization.

## Contents
- `exploratory/` - Initial exploration of each POM image and crystallinity analysis.
- `final/` - Final analysis of the correlation between percent crystallinity and material properties.

## Notes
- The `single_POM_analysis.ipynb` file is designed to load a raw POM image and control segmentation parameters without modifying `preprocessing.py` and `watershed_segmentation.py`.
- Output of crystallization analysis will be saved in a dataframe.
- `single_POM_analysis_old.ipynb` file has the parameters that can successfully segment Maltese crosses. 
- `single_POM_analysis.ipynb` and `POM_analysis_spherulite.ipynb` files have the parameters that are an attempt to segment spherulites. However, due to the presence of multiple layers of spherulites this has proven difficult. These notebooks are currently undergoing optimization and debugging.
- Although all Maltese crosses are identified in the segmentation step in `single_POM_analysis_old.ipynb` file, issues with merging of crosses and splitting of crosses persist. Therefore, for improving the feature extraction accuracy, the preprocessing parameters need more tuning.