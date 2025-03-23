# Documentation (`/docs/`)

This directory contains the documentation on the project including the project description, structure, and dependencies.


## Project Title: Image Analysis and Percent Crystallinity Quantification


## Brief Description: 	

This project involves creating a Python-based tool that processes polarized optical microscope (POM) images of a semicrystalline  thiol-ene photopolymer system, and analyzes the percent crystallinity. It will also attempt to find a correlation between the percent crystallinity and the thermal and mechanical properties obtained from experimental data.


## Project Objectives:

1. To preprocess the POM images to enhance the contrast between crystalline and amorphous regions, and to reduce noise.
2. To define the features that characterize crystallinity and develop an algorithm to distinguish between crystalline and amorphous regions.
3. To calculate the percent crystallinity of each image and aggregate the results of multiple images using statistical methods.
4. To analyze the relationships between the percent crystallinity and the polymer properties.
5. To use data visualization tools to display the results.


## Folder Structure:
cmse802_project/
│
├── data/									         # Store all data files 
│   ├── raw/             					      # Raw, unprocessed data (not committed to GitHub)
│   ├── preprocessed/    					      # Preprocessed data
│   ├── processed/       					      # Segmented/processed data
│
├── notebooks/           					 # Jupyter notebooks
│   ├── exploratory/
│   ├── final/
│
├── src/             	  					 # Python scripts
│   ├── data_loading/						        # Image loading scripts
│   ├── preprocessing/ 					        # Preprocessing functions
│   ├── segmentation/	 					        # Segmentation functions (watershed algorithm)
│   ├── feature_extraction/				      # Extracting the features of percent crystallinity
│   ├── crystallinity_quantification/	 	# Statistical calculations of percent crystallinity
│   ├── regression/      					      # Regression analysis for property correlation
│   ├── visualization/      					  # Scripts for displaying images and plots
│
├── results/             					 # Store results such as figures, plots, and output data (not committed to GitHub)
│   ├── figures/         					      # Visualizations and plots
│   ├── tables/          					      # Tabular results
│
├── reports/             					 # Store progress reports on the project
│   ├── interim/         					      
│   ├── final/          					      
│
├── tests/               					 # Unit tests for scripts
├── env/               					   # Project environment (not committed to GitHub)
├── .gitignore           					 # Specify files/directories to ignore in Git
└── docs/            					     # Documentation for the project


## Instructions for Setting Up and Running the Code:

1. **Clone the Repository**:

2. **Set Up the Environment**:
- Create a virtual environment:
  ```
  python3 -m venv env
  source env/bin/activate  # On Windows: env\Scripts\activate
  ```
- Install dependencies:
  ```
  # On bash: pip install -r requirements.txt 
  ```

3. **Run the Code**:
- Open Jupyter notebooks in the `notebooks/exploratory` folder to run data loading, preprocessing, segmentation, feature extraction, and crystallinity quantification of each POM image.
- Open Jupyter notebooks in the `notebooks/final` folder to run regression steps.
  ```
- Alternatively, use Python scripts in the `src/` folder.
  ```

## Dependencies:
- Python 3.8+
- numpy – Used for numerical operations and array manipulation.
- opencv-python – For image loading, processing, and visualization.
- matplotlib – To generate and display figures.
- scikit-image – Provides advanced image processing functions.
- scipy – Used for additional image processing tasks.
- pandas – For handling dataframes.
- jupyter – Needed to run and execute .ipynb notebooks.

## Notes:
- ChatGPT was utilized for drafting python codes and troubleshooting errors.
```