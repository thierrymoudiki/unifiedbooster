from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


class GBDTRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model_type='xgboost', 
                 n_estimators=100, 
                 learning_rate=0.1, 
                 max_depth=3, 
                 subsample=1.0,    
                 verbosity=0,              
                 **kwargs):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.verbosity = verbosity
        # xgboost -----
        # n_estimators        
        # learning_rate
        # subsample
        # max_depth
        # lightgbm -----
        # n_estimators
        # learning_rate
        # bagging_fraction
        # max_depth        
        # catboost -----
        # iterations
        # learning_rate        
        # rsm 
        # depth
        if self.model_type == "xgboost": 
            self.params = {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'max_depth': self.max_depth,
                'verbosity': self.verbosity,
                **kwargs
            }
        elif self.model_type == "lightgbm":
             verbose = self.verbosity - 1 if self.verbosity==0 else self.verbosity
             self.params = {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'bagging_fraction': self.subsample,
                'max_depth': self.max_depth,
                'verbose': verbose,
                **kwargs
            }
        elif self.model_type == "catboost":
             self.params = {
                'iterations': self.n_estimators,
                'learning_rate': self.learning_rate,
                'rsm': self.subsample,
                'depth': self.max_depth,
                'verbose': self.verbosity,
                **kwargs
            }           
        
        if model_type == 'xgboost':
            self.model = XGBRegressor(**self.params)
        elif model_type == 'catboost':            
            self.model = CatBoostRegressor(**self.params)
        elif model_type == 'lightgbm':
            self.model = LGBMRegressor(**self.params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)        