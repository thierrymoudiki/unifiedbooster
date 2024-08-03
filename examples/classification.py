import unifiedbooster as ub
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the unified clf (example with XGBoost)
clf1 = ub.GBDTClassifier(model_type='xgboost')
#clf2 = ub.GBDTClassifier(model_type='catboost')
clf3 = ub.GBDTClassifier(model_type='lightgbm')
clf4 = ub.GBDTClassifier(model_type='gradientboosting', 
                         colsample=0.9)

# Fit the model
clf1.fit(X_train, y_train)
#clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)
clf4.fit(X_train, y_train)

# Predict on the test set
y_pred1 = clf1.predict(X_test)
#y_pred2 = clf2.predict(X_test)
y_pred3 = clf3.predict(X_test)
y_pred4 = clf4.predict(X_test)

# Evaluate the model
accuracy1 = accuracy_score(y_test, y_pred1)
#accuracy2 = accuracy_score(y_test, y_pred2)
accuracy3 = accuracy_score(y_test, y_pred3)
accuracy4 = accuracy_score(y_test, y_pred4)
print(f"Classification Accuracy xgboost: {accuracy1:.2f}")
#print(f"Classification Accuracy catboost: {accuracy2:.2f}")
print(f"Classification Accuracy lightgbm: {accuracy3:.2f}")
print(f"Classification Accuracy gradientboosting: {accuracy4:.2f}")
print(f"CV xgboost: {cross_val_score(clf1, X_train, y_train)}")
print(f"CV lightgbm: {cross_val_score(clf3, X_train, y_train)}")
print(f"CV gradientboosting: {cross_val_score(clf4, X_train, y_train)}")