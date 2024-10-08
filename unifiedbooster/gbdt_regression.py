from .gbdt import GBDT
from sklearn.base import RegressorMixin
from .predictioninterval import PredictionInterval

try:
    from xgboost import XGBRegressor
except:
    pass
try:
    from catboost import CatBoostRegressor
except:
    pass
try:
    from lightgbm import LGBMRegressor
except:
    pass
from sklearn.ensemble import GradientBoostingRegressor


class GBDTRegressor(GBDT, RegressorMixin):
    """GBDT Regression model

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

        level: float
            confidence level for prediction sets

        pi_method: str
            method for constructing the prediction intervals: 'splitconformal', 'localconformal'

        verbose: int
            controls verbosity (default=0)

        seed: int
            reproducibility seed

        **kwargs: dict
            additional parameters to be passed to the class

    Examples:

        ```python
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
        #regressor2 = ub.GBDTRegressor(model_type='catboost')
        regressor3 = ub.GBDTRegressor(model_type='lightgbm')

        # Fit the model
        regressor1.fit(X_train, y_train)
        #regressor2.fit(X_train, y_train)
        regressor3.fit(X_train, y_train)

        # Predict on the test set
        y_pred1 = regressor1.predict(X_test)
        #y_pred2 = regressor2.predict(X_test)
        y_pred3 = regressor3.predict(X_test)

        # Evaluate the model
        mse1 = mean_squared_error(y_test, y_pred1)
        #mse2 = mean_squared_error(y_test, y_pred2)
        mse3 = mean_squared_error(y_test, y_pred3)
        print(f"Regression Mean Squared Error xgboost: {mse1:.2f}")
        #print(f"Regression Mean Squared Error catboost: {mse2:.2f}")
        print(f"Regression Mean Squared Error lightgbm: {mse3:.2f}")
        ```
    """

    def __init__(
        self,
        model_type="xgboost",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        rowsample=1.0,
        colsample=1.0,
        level=None,
        pi_method="splitconformal",
        verbose=0,
        seed=123,
        **kwargs,
    ):

        self.type_fit = "regression"

        super().__init__(
            model_type=model_type,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            rowsample=rowsample,
            colsample=colsample,
            level=level,
            pi_method=pi_method,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )

        if self.level is not None:

            if model_type == "xgboost":
                self.model = PredictionInterval(
                    XGBRegressor(**self.params),
                    level=self.level,
                    method=self.pi_method,
                )
            elif model_type == "catboost":
                self.model = PredictionInterval(
                    CatBoostRegressor(**self.params),
                    level=self.level,
                    method=self.pi_method,
                )
            elif model_type == "lightgbm":
                self.model = PredictionInterval(
                    LGBMRegressor(**self.params),
                    level=self.level,
                    method=self.pi_method,
                )
            elif model_type == "gradientboosting":
                self.model = PredictionInterval(
                    GradientBoostingRegressor(**self.params),
                    level=self.level,
                    method=self.pi_method,
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

        else:

            if model_type == "xgboost":
                self.model = XGBRegressor(**self.params)
            elif model_type == "catboost":
                self.model = CatBoostRegressor(**self.params)
            elif model_type == "lightgbm":
                self.model = LGBMRegressor(**self.params)
            elif model_type == "gradientboosting":
                self.model = GradientBoostingRegressor(**self.params)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
