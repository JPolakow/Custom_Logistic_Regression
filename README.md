# Logistic Regression on Simulated COVID-19 Dataset

## Overview
This project explores the development and application of custom logistic regression models to analyze a simulated dataset of hospital patients, determining the likelihood of COVID-19 infection based on health-related factors. The assignment focuses on understanding and implementing the core concepts of machine learning, data preprocessing, and logistic regression while addressing challenges like class imbalance and overfitting. This project was instrumental in deepening the understanding of the inner workings of ML & logistic regression models, including how weights and biases are adjusted iteratively.


## Key Features
-Custom Logistic Regression Model: Built from scratch without relying on external machine learning libraries.
-Data Cleaning and Preprocessing: Includes handling missing values, duplicates, unnecessary columns, and encoding boolean features.
-Exploratory Data Analysis (EDA): Visualizations and summaries to understand data distribution, feature relationships, and output label imbalance.
-Dimensionality Reduction: Utilizes PCA to reduce feature complexity and address overfitting.
-Evaluation and Visualizations: Comprehensive evaluation with metrics (R², accuracy, F1) and plots (ROC curve, confusion matrix, cost convergence).
-Scalable Analysis: Designed to handle a large dataset of 1 million rows using PySpark for efficient data processing.


## Project Workflow

### 1. Data Cleaning
Performed using **PySpark** to handle the large dataset efficiently.
- Removed unnecessary columns (e.g., unique patient identifiers).
- Handled missing values and duplicates.
- Converted Boolean features to int32 using an encoder.

### 2. Exploratory Data Analysis (EDA)
- **Output Label Distribution**: Visualized using a histogram, revealing class imbalance.
- **Cleaned Data Overview**: Presented summary statistics and data types after preprocessing.
- **Heatmap**: Showed correlations between features and with the output label, providing insights into feature interactions.

### 3. Custom Logistic Regression Models
Two custom logistic regression models were implemented from scratch without inheriting from existing libraries like SKLearn. These models provide hands-on understanding of:
- Training processes, including weight and bias adjustment.
- Sigmoid function and its role in logistic regression.
- Iterative cost function minimization for model learning.

#### Key Functionalities:
- **`.fit`**: Trains the model by adjusting weights and biases over iterations.
- **`.predict`**: Generates predictions on unseen data.
- Evaluation methods to calculate metrics such as accuracy, R², and F1 scores.

### 4. Data Processing
- **Scaling**: Standardized input features to improve model performance.
- **Principal Component Analysis (PCA)**: Reduced feature dimensionality to mitigate overfitting and improve computational efficiency.

### 5. Model Training and Evaluation
- Trained the models using the `.fit` method on processed data.
- Evaluated using:
  - **R²**: Measures model fit.
  - **Accuracy**: Assesses classification performance.
  - **F1 Score**: Balances precision and recall.
- Visualizations:
  - Confusion Matrix.
  - Cost Function Convergence Graph.
  - ROC Curve.


## Dataset
**Title**: Covid 19 Simulated Dataset by Abhilash Jash  
**Author**: Abhilash Jash  
**Date**: 2024  
**Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/abhilashjash/covid-19-simulated-dataset-by-abhilash-jash)  


## Technical Details
- **Development Environment**:
  - Visual Studio Code with Jupyter Notebook extension.
  - Anaconda3 (Python 3.11.7).

- **Primary Dependencies**:
  - Numpy
  - PySpark
  - SKLearn
