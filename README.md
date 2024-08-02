# unifiedbooster

Unified interface for Gradient Boosted Decision Trees algorithms

![PyPI](https://img.shields.io/pypi/v/unifiedbooster) [![PyPI - License](https://img.shields.io/pypi/l/unifiedbooster)](https://github.com/thierrymoudiki/unifiedbooster/blob/main/LICENSE) [![Downloads](https://pepy.tech/badge/unifiedbooster)](https://pepy.tech/project/unifiedbooster) 
[![Documentation](https://img.shields.io/badge/documentation-is_here-green)](https://techtonique.github.io/unifiedbooster/)

## Examples 

### classification 

```python
import unifiedbooster as ub
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

datasets = [load_iris(), load_breast_cancer(), load_wine()]

for dataset in datasets:

  X, y = dataset.data, dataset.target
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Initialize the unified regressor (example with XGBoost)
  regressor1 = ub.GBDTClassifier(model_type='xgboost')
  regressor2 = ub.GBDTClassifier(model_type='catboost')
  regressor3 = ub.GBDTClassifier(model_type='lightgbm')

  # Fit the model
  regressor1.fit(X_train, y_train)
  regressor2.fit(X_train, y_train)
  regressor3.fit(X_train, y_train)

  # Predict on the test set
  y_pred1 = regressor1.predict(X_test)
  y_pred2 = regressor2.predict(X_test)
  y_pred3 = regressor3.predict(X_test)

  # Evaluate the model
  accuracy1 = accuracy_score(y_test, y_pred1)
  accuracy2 = accuracy_score(y_test, y_pred2)
  accuracy3 = accuracy_score(y_test, y_pred3)
  print("-------------------------")
  print(f"Classification Accuracy xgboost: {accuracy1:.2f}")
  print(f"Classification Accuracy catboost: {accuracy2:.2f}")
  print(f"Classification Accuracy lightgbm: {accuracy3:.2f}")
```

### regression 

```python
import numpy as np
import unifiedbooster as ub
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


datasets = [fetch_california_housing(), load_diabetes()]

for dataset in datasets:

  # Load dataset
  X, y = dataset.data, dataset.target

  # Split dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Initialize the unified regressor (example with XGBoost)
  regressor1 = ub.GBDTRegressor(model_type='xgboost')
  regressor2 = ub.GBDTRegressor(model_type='catboost')
  regressor3 = ub.GBDTRegressor(model_type='lightgbm')

  # Fit the model
  regressor1.fit(X_train, y_train)
  regressor2.fit(X_train, y_train)
  regressor3.fit(X_train, y_train)

  # Predict on the test set
  y_pred1 = regressor1.predict(X_test)
  y_pred2 = regressor2.predict(X_test)
  y_pred3 = regressor3.predict(X_test)

  # Evaluate the model
  mse1 = np.sqrt(mean_squared_error(y_test, y_pred1))
  mse2 = np.sqrt(mean_squared_error(y_test, y_pred2))
  mse3 = np.sqrt(mean_squared_error(y_test, y_pred3))
  print("-------------------------")
  print(f"Regression Root Mean Squared Error xgboost: {mse1:.2f}")
  print(f"Regression Root Mean Squared Error catboost: {mse2:.2f}")
  print(f"Regression Root Mean Squared Error lightgbm: {mse3:.2f}")
```