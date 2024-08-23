# Interactive Machine Learning Model Selector

## Overview

The "Interactive Machine Learning Model Selector" is a Streamlit-based web application that enables users to explore various machine learning models interactively. It allows users to upload datasets, clean and preprocess data, visualize it, and apply different machine learning models. This application provides insights into model performance and helps users understand the effects of various features on model predictions.

## Features

- **Data Upload:** Upload CSV or SQL files for analysis.
- **Data Cleaning:** Options to remove rows with missing values and duplicates.
- **Feature Encoding:** Automatically encodes categorical features into numeric values.
- **Data Visualization:**
  - Feature Correlation Heatmap
  - Histograms for numeric columns
  - Scatter plots for selected features
- **Model Selection:** Choose from several machine learning models:
  - Linear Regression
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors
  - Naive Bayes
  - Clustering (K-Means)
- **Model Evaluation:** Display performance metrics such as accuracy, mean squared error, and confusion matrix.

## Installation

To run this project locally, you need to install the required libraries. You can use the `requirements.txt` file provided in the repository.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/ml-model-selector.git
