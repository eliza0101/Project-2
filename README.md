# Project-2 Fraud Detection System
#### Link to repo https://github.com/eliza0101/Project-2.git  
#### Link to presentation https://www.canva.com/design/DAGPMmNKy24/DOzfNhLtdTywsBonFecWRA/edit?utm_content=DAGPMmNKy24&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton 


## Overview
Create a POC model to detect transaction fraud. 


## Dependencies
* Pandas
* Matplotlib 
* seaborn
* sklearn


## Datasets (Eliza)
#### File location: Movie_Datasets
We will be using the datasets from: https://www.kaggle.com/competitions/ieee-fraud-detection/data
1. train_identity.csv
2. tran_transaction.csv

## Data Preparation and Cleaning (Eliza) 
#### File location: project_2_data cleaning.ipynb 
* Merge both datasets into one dataframe
* Clean data and remove colomns with more than 10% missing values
* Handle non-numeric data
* Create new csv for merged and cleaned files

## Exploratory Data Analysis (Eliza) 
#### File location: project_2_main.ipynb / project_2_main.py
* Missing Values Analysis
* Numerical Data Analysis
* Visualize class distribution
* Categorical Data Analysis
* Correlation Analysis 

## Model Development (Tunji) 
#### File location: project_2_main.ipynb / project_2_main.py
* Train Logistic Regression
* Train Random Forest Classifier
* Top Features Selection
* Confusion Matrix
* ROC Curve

## Model Optimization and Reporting (Will) 
#### File location: project_2_main.ipynb / project_2_main.py
* Hyperparameter Tuning (Tunji and Eliza)
* Best Parameters and Score (Tunji and Eliza)
* Line Plot of ROC AUC Scores (Tunji and Eliza)
* Summary Chart (Tunji and Eliza)

## Notes
The datafiles.zip needs to be unzipped before running the cleaning notebook.