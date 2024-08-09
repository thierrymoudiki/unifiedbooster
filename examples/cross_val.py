import os 
import unifiedbooster as ub
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score
from time import time

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")    

dataset = load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n Example 1 -----")

res1 = ub.cross_val_optim(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    model_type="lightgbm",
    type_fit="classification",
    scoring="accuracy",
    n_estimators=100,
    surrogate_obj=None,
    cv=5,
    n_jobs=None,
    n_init=10,
    n_iter=190,
    abs_tol=1e-3,
    verbose=2,
    seed=123,
)
print(res1)

print("\n Example 2 -----")

res2 = ub.cross_val_optim(
    X_train,
    y_train,
    X_test=X_test,
    y_test=y_test,
    model_type="lightgbm",
    type_fit="classification",
    scoring="accuracy",
    n_estimators=100,
    surrogate_obj=None,
    cv=5,
    n_jobs=None,
    n_init=10,
    n_iter=190,
    abs_tol=1e-3,
    verbose=2,
    seed=123,
)
print(res2)

print("\n Example 3 -----")

res3 = ub.cross_val_optim(
    X_train,
    y_train,
    X_test=X_test,
    y_test=y_test,
    model_type="lightgbm",
    type_fit="classification",
    scoring="accuracy",
    n_estimators=100,
    surrogate_obj=KernelRidge(),
    cv=5,
    n_jobs=None,
    n_init=10,
    n_iter=190,
    abs_tol=1e-3,
    verbose=2,
    seed=123,
)
print(res3)

print("\n Example 4 -----")

res2 = ub.cross_val_optim(
    X_train,
    y_train,
    X_test=X_test,
    y_test=y_test,
    model_type="lightgbm",
    type_fit="classification",
    scoring="accuracy",
    surrogate_obj=None,
    cv=5,
    n_jobs=None,
    n_init=10,
    n_iter=190,
    abs_tol=1e-3,
    verbose=2,
    seed=123,
)
print(res2)

print("\n Example 5 -----")

res3 = ub.cross_val_optim(
    X_train,
    y_train,
    X_test=X_test,
    y_test=y_test,
    model_type="lightgbm",
    type_fit="classification",
    scoring="accuracy",
    surrogate_obj=KernelRidge(),
    cv=5,
    n_jobs=None,
    n_init=10,
    n_iter=190,
    abs_tol=1e-3,
    verbose=2,
    seed=123,
)
print(res3)
