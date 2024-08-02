import numpy as np 
from sklearn.base import BaseEstimator


class GBDT(BaseEstimator):
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
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'subsample': self.rowsample, 
                'colsample_bynode': self.colsample,                
                'max_depth': self.max_depth,
                'verbosity': self.verbose, 
                'seed': self.seed,
                **kwargs
            }
        elif self.model_type == "lightgbm":
             verbose = self.verbose - 1 if self.verbose==0 else self.verbose
             self.params = {
                'n_estimators': self.n_estimators, 
                'learning_rate': self.learning_rate, 
                'subsample': self.rowsample, 
                'feature_fraction_bynode': self.colsample,                
                'max_depth': self.max_depth,
                'verbose': verbose, # keep this way
                'seed': self.seed,
                **kwargs
            }
        elif self.model_type == "catboost":
             self.params = {
                'iterations': self.n_estimators, 
                'learning_rate': self.learning_rate, 
                'subsample': self.rowsample, 
                'rsm': self.colsample, 
                'depth': self.max_depth, 
                'verbose': self.verbose,
                'random_seed': self.seed, 
                'bootstrap_type': 'Bernoulli',
                **kwargs
            }           
        
    def fit(self, X, y, **kwargs):
        if getattr(self, "type_fit") == "classification":
            self.classes_ = np.unique(y) # for compatibility with sklearn
            self.n_classes_ = len(self.classes_)  # for compatibility with sklearn        
        return getattr(self, "model").fit(X, y, **kwargs)
    
    def predict(self, X):
        return getattr(self, "model").predict(X)        