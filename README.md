# Linear Regression Implementation Using Gradient Descent

This repository contains Python code for implementing linear regression from scratch and validating it against the `sklearn` library. Linear regression is a fundamental technique in statistical modeling for predicting a continuous variable based on one or more predictor variables.

## About the Code

The code consists of a class `LinearRegression` which implements linear regression using gradient descent, with an option for regularization. It includes the following functionalities:

- Importing necessary libraries.
- Defining the `LinearRegression` class with methods for:
  - Normalizing training and testing data.
  - Gradient descent algorithm for model training.
  - Predicting output using current weights and bias.
  - Calculating the cost function value.
  - Plotting the cost function value over iterations.
  - Splitting data into training and testing sets.
  - Calculating Root Mean Square Error (RMSE) and Sum of Square Error (SSE).
- Fetching the California Housing dataset.
- Creating instances of the `LinearRegression` class with and without regularization.
- Training the models and printing results.

## Usage

You can use the provided code as follows:

```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.datasets import fetch_california_housing

# Definition of LinearRegression class and methods...

# Fetching the data
cal_housing = fetch_california_housing()

# Creating an instance of LinearRegression class without regularization
lr = LinearRegression(cal_housing.data, cal_housing.target, learning_rate=0.5, tolerance=0.000001, max_iterations=50000, regularization=False, lamda=0.05)

# Training and predicting
lr.fit()

# Creating an instance of LinearRegression class with regularization
lr_reg = LinearRegression(cal_housing.data, cal_housing.target, learning_rate=0.6, tolerance=0.000001, max_iterations=50000, regularization=True, lamda=0.000001)

# Training and predicting
lr_reg.fit()
