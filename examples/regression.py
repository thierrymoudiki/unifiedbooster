import unifiedbooster as ub
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the unified regr (example with XGBoost)
regr1 = ub.GBDTregr(model_type='xgboost')
#regr2 = ub.GBDTregr(model_type='catboost')
regr3 = ub.GBDTregr(model_type='lightgbm')
regr4 = ub.GBDTregr(model_type='gradientboosting', 
                    colsample=0.9)

# Fit the model
regr1.fit(X_train, y_train)
#regr2.fit(X_train, y_train)
regr3.fit(X_train, y_train)
regr4.fit(X_train, y_train)

# Predict on the test set
y_pred1 = regr1.predict(X_test)
#y_pred2 = regr2.predict(X_test)
y_pred3 = regr3.predict(X_test)
y_pred4 = regr4.predict(X_test)

# Evaluate the model
mse1 = mean_squared_error(y_test, y_pred1)
#mse2 = mean_squared_error(y_test, y_pred2)
mse3 = mean_squared_error(y_test, y_pred3)
mse4 = mean_squared_error(y_test, y_pred4)
print(f"Regression Mean Squared Error xgboost: {mse1:.2f}")
#print(f"Regression Mean Squared Error catboost: {mse2:.2f}")
print(f"Regression Mean Squared Error lightgbm: {mse3:.2f}")
print(f"Regression Mean Squared Error gradientboosting: {mse4:.2f}")
print(f"CV xgboost: {cross_val_score(regr1, X_train, y_train)}")
print(f"CV lightgbm: {cross_val_score(regr3, X_train, y_train)}")
print(f"CV gradientboosting: {cross_val_score(regr4, X_train, y_train)}")