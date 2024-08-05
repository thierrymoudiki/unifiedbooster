import numpy as np
from sklearn.base import BaseEstimator


class GBDT(BaseEstimator):
    """Gradient Boosted Decision Trees (GBDT) base class

    Attributes:

        model_type: str
            type of gradient boosting algorithm: 'xgboost', 'lightgbm',
            'catboost', 'gradientboosting'

        n_estimators: int
            maximum number of trees that can be built

        learning_rate: float
            shrinkage rate; used for reducing the gradient step

        max_depth: int
            maximum tree depth

        rowsample: float
            subsample ratio of the training instances

        colsample: float
            percentage of features to use at each node split

        verbose: int
            controls verbosity (default=0)

        seed: int
            reproducibility seed

        **kwargs: dict
            additional parameters to be passed to the class
    """

    def __init__(
        self,
        model_type="xgboost",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        rowsample=1.0,
        colsample=1.0,
        verbose=0,
        seed=123,
        **kwargs
    ):

        self.model_type = model_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.rowsample = rowsample
        self.colsample = colsample
        self.verbose = verbose
        self.seed = seed

        if self.model_type == "xgboost":
            self.params = {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "subsample": self.rowsample,
                "colsample_bynode": self.colsample,
                "max_depth": self.max_depth,
                "verbosity": self.verbose,
                "seed": self.seed,
                **kwargs,
            }
        elif self.model_type == "lightgbm":
            verbose = self.verbose - 1 if self.verbose == 0 else self.verbose
            self.params = {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "subsample": self.rowsample,
                "feature_fraction_bynode": self.colsample,
                "max_depth": self.max_depth,
                "verbose": verbose,  # keep this way
                "seed": self.seed,
                **kwargs,
            }
        elif self.model_type == "catboost":
            self.params = {
                "iterations": self.n_estimators,
                "learning_rate": self.learning_rate,
                "subsample": self.rowsample,
                "rsm": self.colsample,
                "depth": self.max_depth,
                "verbose": self.verbose,
                "random_seed": self.seed,
                "boosting_type": "Plain",
                "leaf_estimation_iterations": 1,                
                "bootstrap_type": "Bernoulli",
                **kwargs,
            }
        elif self.model_type == "gradientboosting":
            self.params = {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "subsample": self.rowsample,
                "max_features": self.colsample,
                "max_depth": self.max_depth,
                "verbose": self.verbose,
                "random_state": self.seed,
                **kwargs,
            }

    def fit(self, X, y, **kwargs):
        """Fit custom model to training data (X, y).

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples]
                Target values.

            **kwargs: additional parameters to be passed to
                        self.cook_training_set or self.obj.fit

        Returns:

            self: object
        """

        if getattr(self, "type_fit") == "classification":
            self.classes_ = np.unique(y)  # for compatibility with sklearn
            self.n_classes_ = len(
                self.classes_
            )  # for compatibility with sklearn
        if getattr(self, "model_type") == "gradientboosting":
            self.model.max_features = int(self.model.max_features * X.shape[1])
        return getattr(self, "model").fit(X, y, **kwargs)

    def predict(self, X):
        """Predict test data X.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            **kwargs: additional parameters to be passed to
                    self.cook_test_set

        Returns:

            model predictions: {array-like}
        """

        return getattr(self, "model").predict(X)
