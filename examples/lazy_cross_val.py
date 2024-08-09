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

# start = time()
# res3 = ub.lazy_cross_val_optim(
#     X_train,
#     y_train,
#     X_test=X_test,
#     y_test=y_test,
#     model_type="lightgbm",
#     type_fit="classification",
#     scoring="accuracy",
#     n_estimators=100,
#     cv=5,
#     n_jobs=None,
#     n_init=10,
#     n_iter=190,
#     abs_tol=1e-3,
#     seed=123,
#     customize=True
# )
# print(f"Elapsed: {time()-start}")

start = time()
res4 = ub.lazy_cross_val_optim(
    X_train,
    y_train,
    X_test=X_test,
    y_test=y_test,
    model_type="lightgbm",
    type_fit="classification",
    scoring="accuracy",
    n_estimators=100,
    cv=5,
    n_jobs=None,
    n_init=10,
    n_iter=190,
    abs_tol=1e-3,
    seed=123,
    customize=False
)
print(f"Elapsed: {time()-start}")
#print(res3)
print(res4)


start = time()
res4 = ub.lazy_cross_val_optim(
    X_train,
    y_train,
    X_test=X_test,
    y_test=y_test,
    model_type="lightgbm",
    type_fit="classification",
    scoring="accuracy",
    cv=5,
    n_jobs=None,
    n_init=10,
    n_iter=190,
    abs_tol=1e-3,
    seed=123,
    customize=False
)
print(f"Elapsed: {time()-start}")
#print(res3)
print(res4)
