import matplotlib.pyplot as plt
import numpy as np
import os 
import unifiedbooster as ub
import warnings
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from time import time


print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

load_datasets = [fetch_california_housing(), load_diabetes()]
dataset_names = ["California Housing", "Diabetes"]

warnings.filterwarnings('ignore')

split_color = 'green'
split_color2 = 'orange'
local_color = 'gray'

def plot_func(x,
              y,
              y_u=None,
              y_l=None,
              pred=None,
              shade_color="lightblue",
              method_name="",
              title=""):

    fig = plt.figure()

    plt.plot(x, y, 'k.', alpha=.3, markersize=10,
             fillstyle='full', label=u'Test set observations')

    if (y_u is not None) and (y_l is not None):
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_u, y_l[::-1]]),
                 alpha=.3, fc=shade_color, ec='None',
                 label = method_name + ' Prediction interval')

    if pred is not None:
        plt.plot(x, pred, 'k--', lw=2, alpha=0.9,
                 label=u'Predicted value')

    #plt.ylim([-2.5, 7])
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.legend(loc='upper right')
    plt.title(title)

    plt.show()

for i, dataset in enumerate(load_datasets):

    print(f"\n ----- Running: {dataset_names[i]} ----- \n")
    X, y = dataset.data, dataset.target

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the unified regr (example with XGBoost)
    print("\n ---------- Initialize the unified regr (example with XGBoost)")
    regr1 = ub.GBDTRegressor(model_type="xgboost", 
                            level=95, 
                            pi_method="splitconformal")

    # Fit the model
    start = time()
    regr1.fit(X_train, y_train)
    print(f"Time taken: {time() - start} seconds")
    # Predict with the model
    y_pred1 = regr1.predict(X_test)
    # Coverage error
    coverage_error = (y_test >= y_pred1.lower) & (y_test <= y_pred1.upper)
    print(f"Coverage rate: {coverage_error.mean():.4f}")
    #x,
    #y,
    #y_u=None,
    #y_l=None,
    #pred=None,
    plot_func(range(len(y_test))[0:30], y_test[0:30],
              y_pred1.upper[0:30], y_pred1.lower[0:30], 
              y_pred1.mean[0:30], method_name="Split Conformal")
    
    print("\n ---------- Initialize the unified regr (example with LightGBM)")
    regr2 = ub.GBDTRegressor(model_type="lightgbm", 
                            level=95, 
                            pi_method="localconformal")
    # Fit the model
    start = time()
    regr2.fit(X_train, y_train)
    print(f"Time taken: {time() - start} seconds")
    # Predict with the model
    y_pred2 = regr2.predict(X_test)
    # Coverage error
    coverage_error = (y_test >= y_pred2.lower) & (y_test <= y_pred2.upper)
    print(f"Coverage rate: {coverage_error.mean():.4f}")
    #x,
    #y,
    #y_u=None,
    #y_l=None,
    #pred=None,
    plot_func(range(len(y_test))[0:30], y_test[0:30], 
              y_pred2.upper[0:30], y_pred2.lower[0:30], 
              y_pred2.mean[0:30], method_name="Local Conformal")