"""XGBoost regression model for tabular forecasting."""
import xgboost as xgb

from .base_model import BaseModel


class XGBModel(BaseModel):
    def __init__(self, **kwargs):
        if "random_state" not in kwargs:
            kwargs = {**kwargs, "random_state": 42}
        self._kwargs = kwargs
        self.model = xgb.XGBRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def reset(self):
        self.model = xgb.XGBRegressor(**self._kwargs)
