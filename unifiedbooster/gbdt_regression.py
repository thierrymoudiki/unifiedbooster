from .gbdt import GBDT
from sklearn.base import RegressorMixin
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


class GBDTRegressor(GBDT, RegressorMixin):
    """GBDT Regression model

    Attributes:

        n_estimators: int
            maximum number of trees that can be built 

        learning_rate: float
            shrinkage rate; used for reducing the gradient step

        rowsample: float
            subsample ratio of the training instances

        colsample: float
            percentage of features to use at each node split

        verbose: int
            controls verbosity (default=0)
        
        seed: int 
            reproducibility seed 

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

    def __init__(self, 
                 model_type='xgboost', 
                 n_estimators=100, 
                 learning_rate=0.1, 
                 max_depth=3, 
                 rowsample=1.0,
                 colsample=1.0,    
                 verbose=0,    
                 seed=123,          
                 **kwargs):
        
        self.type_fit = "regression"
                        
        super().__init__(
            model_type=model_type, 
            n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            max_depth=max_depth, 
            rowsample=rowsample,
            colsample=colsample,    
            verbose=verbose,    
            seed=seed,          
            **kwargs
        )

        if model_type == 'xgboost':
            self.model = XGBRegressor(**self.params)
        elif model_type == 'catboost':            
            self.model = CatBoostRegressor(**self.params)
        elif model_type == 'lightgbm':
            self.model = LGBMRegressor(**self.params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
