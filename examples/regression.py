import unifiedbooster as ub
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

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
mse1 = mean_squared_error(y_test, y_pred1)
mse2 = mean_squared_error(y_test, y_pred2)
mse3 = mean_squared_error(y_test, y_pred3)
print(f"Regression Mean Squared Error xgboost: {mse1:.2f}")
print(f"Regression Mean Squared Error catboost: {mse2:.2f}")
print(f"Regression Mean Squared Error lightgbm: {mse3:.2f}")
