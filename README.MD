Machine Learning Model Explorer
This Streamlit application allows users to explore various machine learning models on their own datasets. Users can upload a CSV file, select target and feature columns, choose a machine learning model, and visualize the results including accuracy, confusion matrices, feature importance, and feature distributions.

Features
Upload CSV File: Upload your dataset in CSV format.
Select Target and Feature Columns: Choose the columns to be used as target and features.
Choose Model: Select from various machine learning models including:
Logistic Regression
Random Forest
Support Vector Machine
K-Means Clustering
Linear Regression
Decision Tree
Model Evaluation: View model accuracy, confusion matrices, mean squared error, and feature importance.
Feature Distribution Visualization: Display histograms of selected feature distributions.

Usage
Upload a CSV file: Use the file uploader in the sidebar to upload your dataset.
Select Columns: Choose the target column and feature columns for modeling.
Choose Model: Select the machine learning model from the sidebar.
View Results: After model training, view the accuracy, confusion matrix, mean squared error, feature importance, and feature distributions.
Models
Logistic Regression: Used for binary or multi-class classification tasks.
Random Forest: An ensemble method for classification and regression tasks.
Support Vector Machine (SVM): Suitable for classification tasks.
K-Means Clustering: An unsupervised learning algorithm used for clustering.
Linear Regression: Used for regression tasks.
Decision Tree: Can be used for both classification and regression tasks.
Notes
Ensure your dataset has a clearly defined target column and feature columns.
For classification tasks, the target variable should be categorical.
For regression tasks, the target variable should be continuous.
