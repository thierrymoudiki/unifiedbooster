import GPopt as gp
import nnetsauce as ns
import numpy as np
from collections import namedtuple
from .gbdt_classification import GBDTClassifier
from .gbdt_regression import GBDTRegressor
from sklearn.model_selection import cross_val_score
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import all_estimators
from sklearn import metrics


def cross_val_optim(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    model_type="xgboost",
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
):
    """Cross-validation function and hyperparameters' search

    Parameters:

        X_train: array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.

        y_train: array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.

        X_test: array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.

        y_test: array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.

        model_type: str
            type of gradient boosting algorithm: 'xgboost', 'lightgbm',
            'catboost', 'gradientboosting'

        type_fit: str
            "regression" or "classification"

        scoring: str
            scoring metric; see https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules

        n_estimators: int
            maximum number of trees that can be built

        surrogate_obj: an object;
            An ML model for estimating the uncertainty around the objective function

        cv: int;
            number of cross-validation folds

        n_jobs: int;
            number of jobs for parallel execution

        n_init: an integer;
            number of points in the initial setting, when `x_init` and `y_init` are not provided

        n_iter: an integer;
            number of iterations of the minimization algorithm

        abs_tol: a float;
            tolerance for convergence of the optimizer (early stopping based on acquisition function)

        verbose: int
            controls verbosity

        seed: int
            reproducibility seed

    Examples:

        ```python
        import unifiedbooster as ub
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split

        dataset = load_breast_cancer()
        X, y = dataset.data, dataset.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

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
        ```
    """

    def gbdt_cv(
        X_train,
        y_train,
        model_type="xgboost",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        rowsample=1.0,
        colsample=1.0,
        cv=5,
        n_jobs=None,
        type_fit="classification",
        scoring="accuracy",
        seed=123,
    ):
        if type_fit == "regression":
            estimator = GBDTRegressor(
                model_type=model_type,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                rowsample=rowsample,
                colsample=colsample,
                verbose=0,
                seed=seed,
            )
        elif type_fit == "classification":
            estimator = GBDTClassifier(
                model_type=model_type,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                rowsample=rowsample,
                colsample=colsample,
                verbose=0,
                seed=seed,
            )
        return -cross_val_score(
            estimator,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=0,
        ).mean()

    # objective function for hyperparams tuning
    def crossval_objective(xx):
        return gbdt_cv(
            X_train=X_train,
            y_train=y_train,
            model_type=model_type,
            n_estimators=n_estimators,
            learning_rate=10 ** xx[0],
            max_depth=int(xx[1]),
            rowsample=xx[2],
            colsample=xx[3],
            cv=cv,
            n_jobs=n_jobs,
            type_fit=type_fit,
            scoring=scoring,
            seed=seed,
        )

    if surrogate_obj is None:
        gp_opt = gp.GPOpt(
            objective_func=crossval_objective,
            lower_bound=np.array([-6, 1, 0.5, 0.5]),
            upper_bound=np.array([0, 16, 1.0, 1.0]),
            params_names=[
                "learning_rate",
                "max_depth",
                "rowsample",
                "colsample",
            ],
            method="bayesian",
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
        )
    else:
        gp_opt = gp.GPOpt(
            objective_func=crossval_objective,
            lower_bound=np.array([-6, 1, 0.5, 0.5]),
            upper_bound=np.array([0, 16, 1.0, 1.0]),
            params_names=[
                "learning_rate",
                "max_depth",
                "rowsample",
                "colsample",
            ],
            acquisition="ucb",
            method="splitconformal",
            surrogate_obj=ns.PredictionInterval(
                obj=surrogate_obj, method="splitconformal"
            ),
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
        )

    res = gp_opt.optimize(verbose=verbose, abs_tol=abs_tol)
    res.best_params["model_type"] = model_type
    res.best_params["n_estimators"] = int(n_estimators)
    res.best_params["learning_rate"] = 10 ** res.best_params["learning_rate"]
    res.best_params["max_depth"] = int(res.best_params["max_depth"])
    res.best_params["rowsample"] = res.best_params["rowsample"]
    res.best_params["colsample"] = res.best_params["colsample"]

    # out-of-sample error
    if X_test is not None and y_test is not None:
        if type_fit == "regression":
            estimator = GBDTRegressor(**res.best_params, verbose=0, seed=seed)
        elif type_fit == "classification":
            estimator = GBDTClassifier(**res.best_params, verbose=0, seed=seed)
        preds = estimator.fit(X_train, y_train).predict(X_test)
        # check error on y_test
        oos_err = getattr(metrics, scoring + "_score")(
            y_true=y_test, y_pred=preds
        )
        result = namedtuple("result", res._fields + ("test_" + scoring,))
        return result(*res, oos_err)
    else:
        return res


def lazy_cross_val_optim(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    model_type="xgboost",
    type_fit="classification",
    scoring="accuracy",
    customize=False,
    n_estimators=100,
    cv=5,
    n_jobs=None,
    n_init=10,
    n_iter=190,
    abs_tol=1e-3,
    verbose=1,
    seed=123,
):
    """Automated Cross-validation function and hyperparameters' search using multiple surrogates

    Parameters:

        X_train: array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.

        y_train: array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.

        X_test: array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.

        y_test: array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.

        model_type: str
            type of gradient boosting algorithm: 'xgboost', 'lightgbm',
            'catboost', 'gradientboosting'

        type_fit: str
            "regression" or "classification"

        scoring: str
            scoring metric; see https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules

        customize: boolean
            if True, the surrogate is transformed into a quasi-randomized network (default is False)
            
        n_estimators: int
            maximum number of trees that can be built

        cv: int;
            number of cross-validation folds

        n_jobs: int;
            number of jobs for parallel execution

        n_init: an integer;
            number of points in the initial setting, when `x_init` and `y_init` are not provided

        n_iter: an integer;
            number of iterations of the minimization algorithm

        abs_tol: a float;
            tolerance for convergence of the optimizer (early stopping based on acquisition function)

        verbose: int
            controls verbosity

        seed: int
            reproducibility seed

    Examples:

        ```python
        import os 
        import unifiedbooster as ub
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from time import time

        print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

        dataset = load_breast_cancer()
        X, y = dataset.data, dataset.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

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
        print(res4)
        ```
    """

    removed_regressors = [
        "TheilSenRegressor",
        "ARDRegression",
        "CCA",
        "GaussianProcessRegressor",
        "GradientBoostingRegressor",
        "HistGradientBoostingRegressor",
        "IsotonicRegression",
        "MultiOutputRegressor",
        "MultiTaskElasticNet",
        "MultiTaskElasticNetCV",
        "MultiTaskLasso",
        "MultiTaskLassoCV",
        "OrthogonalMatchingPursuit",
        "OrthogonalMatchingPursuitCV",
        "PLSCanonical",
        "PLSRegression",
        "RadiusNeighborsRegressor",
        "RegressorChain",
        "StackingRegressor",
        "VotingRegressor",
    ]

    results = []

    for est in all_estimators():
        if issubclass(est[1], RegressorMixin) and (
            est[0] not in removed_regressors
        ):
            try:
                if customize == True:
                    print(f"\n surrogate: CustomRegressor({est[0]})")
                    surr_obj = ns.CustomRegressor(obj=est[1]())
                else: 
                    print(f"\n surrogate: {est[0]}")
                    surr_obj = est[1]()
                res = cross_val_optim(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    model_type=model_type,
                    n_estimators=n_estimators,
                    surrogate_obj=surr_obj,
                    cv=cv,
                    n_jobs=n_jobs,
                    type_fit=type_fit,
                    scoring=scoring,
                    n_init=n_init,
                    n_iter=n_iter,
                    abs_tol=abs_tol,
                    verbose=verbose,
                    seed=seed,
                )
                print(f"\n result: {res}")
                if customize == True:
                    results.append((f"CustomRegressor({est[0]})", res))
                else:
                    results.append((est[0], res))                     
            except:
                pass

    return results
