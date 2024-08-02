# Loan Eligibility Model Solution

## Overview

This project provides a solution for determining loan eligibility based on historical credit data. It includes scripts for data preprocessing, model training, evaluation, and prediction. The solution is modularized to ensure flexibility and scalability.

## Project Structure

- **main.py**: The main script to run the application.
- **data/**: Contains the dataset used for training and testing.
  - **credit.csv**: The raw dataset.
  - **Processed_Credit_Dataset.csv**: The processed dataset used for modeling.
- **logs/**: Contains logs generated during the execution.
  - **app.log**: Log file for tracking application events.
- **src/**: Contains the core functionality.
  - **data/**: Data loading utilities.
  - **models/**: Scripts for training, evaluating, and predicting using the model.
  - **preprocess_partition/**: Scripts for data preprocessing and partitioning.
  - **utils/**: Utility functions.
  - **requirements.txt**: Lists the Python packages required to run the project.


## Dependencies
Plese see the requirement.txt file

## How to Run
To execute the project, navigate to your project directory and run the main.py script. Ensure that Python is installed and accessible via your command line:

python main.py
