# cmse802_project

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
├── data/									# Store all data files
│   ├── raw/             					# Raw, unprocessed data
│   ├── preprocessed/    					# Preprocessed data
│   ├── processed/       					# Segmented/processed data
│
├── notebooks/           					# Jupyter notebooks
│   ├── exploratory/
│   ├── final/
│
├── src/             	  					# Python scripts
│   ├── data_loading/						# Image loading script
│   ├── preprocessing/ 					# Preprocessing functions
│   ├── segmentation/	 					# Segmentation functions (watershed algorithm)
│   ├── feature_extraction/				# Extracting the features of percent crystallinity
│   ├── crystallinity_quantification/	 	# Statistical calculations of percent crystallinity
│   ├── regression/      					# Regression analysis for property correlation
│
├── results/             					# Store results such as figures, plots, and output data
│   ├── figures/         					# Visualizations and plots
│   ├── tables/          					# Tabular results
│
├── tests/               					# Unit tests for scripts
├── config/
├── .gitignore           					# Specify files/directories to ignore in Git
└── README.md            					# Documentation for the project


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
  pip install -r requirements.txt
  ```

3. **Run the Code**:
- Open Jupyter notebooks in the `notebooks/final` folder to run data loading, preprocessing, segmentation, feature extraction, crystallinity quantification, and regression steps.
  ```
- Alternatively, use Python scripts in the `src/` folder.
  ```

## Dependencies:
- Python 3.8+
- NumPy: Array operations (`pip install numpy`)
- OpenCV: Image processing (`pip install opencv-python`)
- scikit-image: Advanced image processing (`pip install scikit-image`)
- Matplotlib: Visualization (`pip install matplotlib`)
- Jupyter Notebook: Interactive coding (`pip install notebook`) or JupyterLab.