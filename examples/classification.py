from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into training and testing sets
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
print(f"Classification Accuracy xgboost: {accuracy1:.2f}")
print(f"Classification Accuracy catboost: {accuracy2:.2f}")
print(f"Classification Accuracy lightgbm: {accuracy3:.2f}")