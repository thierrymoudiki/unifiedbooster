import numpy as np
import os 
import unifiedbooster as ub
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from time import time


print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

load_datasets = [load_iris(), load_breast_cancer(), load_wine()]
dataset_names = ["Iris", "Breast Cancer", "Wine"]

for i, dataset in enumerate(load_datasets):

    print(f"\n ----- Running: {dataset_names[i]} ----- \n")
    X, y = dataset.data, dataset.target

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the unified clf (example with XGBoost)
    print("\n ---------- Initialize the unified clf (example with XGBoost)")
    clf1 = ub.GBDTClassifier(model_type="xgboost", 
                            level=95, 
                            pi_method="tcp")

    # Fit the model
    start = time()
    clf1.fit(X_train, y_train)
    print(f"Time taken: {time() - start} seconds")
    # Predict with the model
    y_pred1 = clf1.predict(X_test)
    print(y_test)
    print(y_pred1.argmax(axis=1))
    # Calculate accuracy
    accuracy = (y_test == y_pred1.argmax(axis=1)).mean()
    print(f"\nAccuracy: {accuracy:.4f}")

    print("\n ---------- Initialize the unified clf (example with LightGBM)")
    clf2 = ub.GBDTClassifier(model_type="lightgbm", 
                            level=95, 
                            pi_method="icp")
    # Fit the model
    start = time()
    clf2.fit(X_train, y_train)
    print(f"Time taken: {time() - start} seconds")
    # Predict with the model
    y_pred2 = clf2.predict(X_test)
    print(y_pred2)

    # Calculate accuracy
    print(y_test)
    print(y_pred2.argmax(axis=1))
    accuracy = (y_test == y_pred2.argmax(axis=1)).mean()
    print(f"\nAccuracy: {accuracy:.4f}")