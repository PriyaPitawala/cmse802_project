# Homework 2: Implementation Framework and Initial Development

## Contents
- [Objective](#objective)
- [Instructions](#instructions)
- [Rubric](#rubric)
- [1. Project Structure and Documentation](#1-project-structure-and-documentation)
- [2. Data/Problem Understanding](#2-dataproblem-understanding)
- [3. Unit Testing Framework](#3-unit-testing-framework)
- [4. Initial Implementation](#4-initial-implementation)
- [5. Progress Report](#5-progress-report)
- [Congratulations, you're done!](#congratulations-youre-done)

## Objective

Building on your project plan from Homework 1, develop the foundational implementation framework for your computational modeling project. This assignment focuses on creating a structured approach to your project, understanding your data/problem space, establishing testing methods, and implementing core functionality.

## Instructions

Complete the following sections in this notebook. Be specific and detailed in your responses. Use code cells for implementation, testing, and demonstration of your progress.

This assignment accommodates both data-focused projects and algorithm implementation projects. Follow the appropriate instructions for your project type in each section.

## Rubric

### 1. Project Structure and Documentation (6 points)

**Code Structure Design (3 points)**
- 3: Comprehensive modular code structure with clear interfaces between components
- 2: Basic modular structure but with some unclear component relationships
- 1: Minimal structure with poor modularity
- 0: No code structure provided

**Documentation Quality (3 points)**
- 3: Detailed documentation with function descriptions, parameter explanations, and usage examples
- 2: Basic documentation covering most components
- 1: Minimal documentation with significant gaps
- 0: No documentation provided

### 2. Data/Problem Understanding (8 points)

**For Data-Focused Projects:**
- 4: Comprehensive EDA with statistical analysis, visualizations, and identified patterns
- 3: Good EDA with basic analysis and visualizations
- 2: Limited EDA with minimal insights
- 1: Superficial data examination
- 0: No data analysis performed

**For Algorithm-Focused Projects:**
- 4: Thorough mathematical description with test case development and theoretical analysis
- 3: Clear mathematical foundation with basic test cases
- 2: Basic algorithm description with limited mathematical foundation
- 1: Vague algorithm description without proper foundation
- 0: No algorithm analysis provided

**Data/Algorithm Implementation Plan (4 points)**
- 4: Detailed implementation plan with preprocessing steps and clear methodology
- 3: Solid implementation plan but missing some details
- 2: Basic implementation outline without sufficient detail
- 1: Vague implementation plan
- 0: No implementation plan provided

### 3. Unit Testing Framework (8 points)

**Test Case Design (4 points)**
- 4: Comprehensive test cases covering core functionality, edge cases, and expected failures
- 3: Good test coverage of main functionality
- 2: Basic test cases with limited coverage
- 1: Minimal testing approach
- 0: No test cases provided

**Testing Implementation (4 points)**
- 4: Well-implemented testing framework with automated validation
- 3: Functional testing approach but missing some automation
- 2: Basic testing implementation with manual components
- 1: Poorly implemented tests
- 0: No testing implementation

### 4. Initial Implementation (8 points)

**Core Functionality Implementation (4 points)**
- 4: Robust implementation of core computational methods with optimization considerations
- 3: Functional implementation of core methods
- 2: Partial implementation with significant gaps
- 1: Minimal implementation attempt
- 0: No implementation provided

**Visualization and Analysis (4 points)**
- 4: Effective visualizations that provide clear insights into results
- 3: Basic visualizations that adequately represent data/results
- 2: Limited visualizations with minimal explanatory value
- 1: Poor or irrelevant visualizations
- 0: No visualizations provided

### 5. Progress Report (5 points)

**Project Progress Assessment (3 points)**
- 3: Detailed assessment of progress against original plan with justified updates
- 2: Basic progress assessment with some plan updates
- 1: Minimal progress reporting without clear plan alignment
- 0: No progress assessment provided

**Course Concept Application (2 points)**
- 2: Clear explanation of how specific course concepts apply to the project
- 1: Vague connections to course concepts
- 0: No course concept connections provided

**Total: 35 points**

## 1. Project Structure and Documentation

### Code Structure Design

Design the overall structure of your project code, following these specific requirements:

**Required Code Organization:**
- Implement core functionality in Python scripts (`.py` files), not notebooks
- Use Jupyter notebooks **only** for data analysis, visualization, and demonstrating results
- Create a modular structure with clear separation of concerns
- Ensure models can be trained via scripts (not requiring notebook execution)
- Implement a command-line interface for running your core functionality

**Directory Structure Requirements:**
- `/src/` - All source code modules and scripts
- `/notebooks/` - Jupyter notebooks for analysis and visualization
- `/data/` - Data files (or documentation of data sources if too large)
- `/tests/` - Unit tests for your implementation
- `/docs/` - Documentation files
- `/results/` - Generated outputs (figures, model checkpoints, etc.)

Create a `README.md` for each directory explaining its purpose and contents.

Provide a diagram or textual description of your code structure below.

```python
## Folder Structure:
cmse802_project/
│
├── data/									         # Store all data files (not committed to GitHub)
│   ├── raw/             					      # Raw, unprocessed data
│   ├── preprocessed/    					      # Preprocessed data
│   ├── processed/       					      # Segmented/processed data
│
├── notebooks/           					 # Jupyter notebooks
│   ├── exploratory/
│   ├── final/
│
├── src/             	  					 # Python scripts
│   ├── data_loading/						        # Image loading script
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

```

### Documentation Strategy

Develop comprehensive documentation for your project following these guidelines:

**Script Documentation Requirements:**
- Include a header comment block in each `.py` file explaining its purpose
- Document all functions using a consistent style (NumPy or Google docstring format)
- Add inline comments for complex sections of code
- Create a `requirements.txt` file listing all dependencies

**Notebook Documentation Requirements:**
- Include markdown cells explaining the purpose of each analysis step
- Document the reasoning behind visualization choices
- Ensure notebooks can be understood without having to run them
- Add a table of contents at the beginning of each notebook

**Example Documentation:**

```python
# example_module.py
"""
Module for data preprocessing and feature engineering.

This module contains functions for cleaning, transforming, and
preparing data for modeling. It handles missing values, outlier
detection, and feature normalization.

Author: Your Name
Date: March 2025
"""

def preprocess_data(data, normalize=True, handle_missing=True):
    """
    Clean and preprocess input data for modeling.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Raw input data with features as columns
    normalize : bool, default=True
        Whether to normalize numerical features
    handle_missing : bool, default=True
        Whether to impute missing values
        
    Returns
    -------
    pandas.DataFrame
        Preprocessed data ready for modeling
    
    Examples
    --------
    >>> df = pd.read_csv('raw_data.csv')
    >>> processed_df = preprocess_data(df)
    """
    # Function implementation
    pass
```

✏️ Answer: All documentation can be seen in the GitHub repository for the project (https://github.com/PriyaPitawala/cmse802_project.git). The summary of project can be found in `docs/` subfolder.

## 2. Data/Problem Understanding

### For Data-Focused Projects:

Perform exploratory data analysis (EDA) on your dataset:
- Generate descriptive statistics
- Create visualizations of key variables
- Identify patterns, anomalies, or quality issues
- Document your findings and insights

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# data = pd.read_csv('your_data.csv')

# Exploratory analysis code here
```

### For Algorithm-Focused Projects:

Develop a mathematical foundation for your algorithm:
- Provide the mathematical formulation
- Analyze theoretical properties (stability, convergence, etc.)
- Create simple test cases with known solutions
- Document expected behavior under different conditions

```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Algorithm foundation code here
```

### Implementation Plan

Outline your approach to implementing your data processing pipeline or algorithm:
- Identify key steps and dependencies
- Determine preprocessing requirements
- Plan computational strategies
- Consider efficiency and optimization approaches

✏️ Answer:  

My project is data-focused, and all visualization, validation, and tuning can be seen in the exploratory notebooks in the `notebooks/` subfolder. The folder also contains a `README.md` file documenting issues, findings, and insights.

1. Key steps and dependencies: 
    - data loading step of polarized optical microscope (POM) images, preprocessing step to prepare for image segmentation, segmentation step using watershed algorithm, feature extraction step, percent crystallinity quantification step, and regression and visualization step. All steps are separated into modules and can be found inside the `src/` subfolder.
    - dependencies are numpy, matplotlib, opencv-python, scikit-image, scipy, and pandas. Comprehensive list can be found in `requirements.txt` file inside the `docs/` subfolder.
2. Preprocessing requirements:
    - grayscale conversion as image segmentation doesn't require color channels, noise reduction using Gaussian blur and further filtering, edge detection using Canny operators, gradient magnitude thresholding to highlight crystal boundaries, morphological operations to clean the binary image, and computation of segmentation markers.
3. Computational strategies:
    - Separate modules for handling the key steps.
    - Interactive exploration, real-time parameter tuning, and segmentation validation using Jupyter notebooks.
    - Vectorization of the image processing steps using NumPy and/or SciPy.
4. Efficiency and optimization approaches:
    - Version control by intermittently committing the progress to  GitHub, especially prior to tuning parameters to optimize segemntation.
    - Image downsampling by removing unnecassary information such as color channels. 

## 3. Unit Testing Framework

### Test Case Design

Design a comprehensive testing approach for your project:
- Identify critical functionality that requires testing
- Create test cases with expected inputs and outputs
- Include edge cases and boundary conditions
- Document how each test validates specific aspects of your implementation

```python
# Test case design code here
```

### Testing Implementation

Implement your testing framework:
- Create functions to automate testing
- Develop validation methods to verify results
- Establish criteria for passing tests
- Document the testing process

```python
# Example unit test implementation
import unittest

class TestYourImplementation(unittest.TestCase):
    def setUp(self):
        # Setup code here
        pass
        
    def test_function_name(self):
        # Test implementation here
        result = function_to_test(input_value)
        self.assertEqual(result, expected_output)
```

✏️ Answer:

## 4. Initial Implementation

### Core Functionality Implementation

Implement the foundational components of your project:

**For Data-Focused Projects:**
- Data loading and preprocessing
- Feature engineering
- Basic modeling or analysis functions

**For Algorithm-Focused Projects:**
- Core algorithm implementation
- Parameter handling
- Solution computation for simple cases

```python
# Core implementation code here
```

### Visualization and Analysis

Create visualizations to analyze your initial results:
- Generate plots that illustrate key findings or algorithm behavior
- Analyze the performance or characteristics of your implementation
- Document insights gained from these visualizations

```python
# Visualization code here
```

✏️ Answer:

## 5. Progress Report

### Project Progress Assessment

Evaluate your progress against your original project plan:
- Compare current status to your Gantt chart from HW1
- Identify areas where you're ahead or behind schedule
- Update your timeline based on new understanding
- Document any scope adjustments needed

### Course Concept Application

Explain how specific concepts from the course are being applied in your project:
- Identify at least 2-3 course topics that directly relate to your implementation
- Explain how these concepts informed your approach
- Describe how course materials helped overcome specific challenges

✏️ Answer:

## Congratulations, you're done!

Submit this assignment by uploading your notebook to the course Desire2Learn web page. Go to the "Homework" folder, find the appropriate submission link, and upload everything there. Make sure your name is on it!

© Copyright 2024, Department of Computational Mathematics, Science and Engineering at Michigan State University
