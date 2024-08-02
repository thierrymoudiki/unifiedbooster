from .gbdt import GBDT
from sklearn.base import ClassifierMixin
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


class GBDTClassifier(GBDT, ClassifierMixin):
    """GBDT Classification model

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
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the unified regressor (example with XGBoost)
    regressor1 = ub.GBDTClassifier(model_type='xgboost')
    #regressor2 = ub.GBDTClassifier(model_type='catboost')
    regressor3 = ub.GBDTClassifier(model_type='lightgbm')

    # Fit the model
    regressor1.fit(X_train, y_train)
    #regressor2.fit(X_train, y_train)
    regressor3.fit(X_train, y_train)

    # Predict on the test set
    y_pred1 = regressor1.predict(X_test)
    #y_pred2 = regressor2.predict(X_test)
    y_pred3 = regressor3.predict(X_test)

    # Evaluate the model
    accuracy1 = accuracy_score(y_test, y_pred1)
    #accuracy2 = accuracy_score(y_test, y_pred2)
    accuracy3 = accuracy_score(y_test, y_pred3)
    print(f"Classification Accuracy xgboost: {accuracy1:.2f}")
    #print(f"Classification Accuracy catboost: {accuracy2:.2f}")
    print(f"Classification Accuracy lightgbm: {accuracy3:.2f}")
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
        
        self.type_fit = "classification"
                        
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
            self.model = XGBClassifier(**self.params)
        elif model_type == 'catboost':            
            self.model = CatBoostClassifier(**self.params)
        elif model_type == 'lightgbm':
            self.model = LGBMClassifier(**self.params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
    def predict_proba(self, X):
        return self.model.predict_proba(X)