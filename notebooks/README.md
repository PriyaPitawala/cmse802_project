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
- `POM_analysis_spherulite.ipynb` file has also used known foreground and background masking to use edge-based segmentation techniques, to segment the multilayer spherulites. This has successfully identified the outer edges of all boundaries, however, there is a merging of multiple sherulites into one segmented region. This notebook is currently undergoing refinement of the segmentation by splitting merged spherulites using watershed segmentation.
-`Avg_spherulite_distribution.ipynb` file computes the average crystallite size distribution (measured as the radial segment length of a spherulite) and the general trend of the samples grouped by their thickness, light intensity used during curing process, and the photoabsorber concentration of the formulation.