"""SARIMA forecast model."""
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMA_Model:
    def __init__(
        self,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model_fit = None

    def fit(self, series):
        series = pd.DataFrame(series).copy()
        if isinstance(series.index, pd.DatetimeIndex) and series.index.freq is None:
            inferred = pd.infer_freq(series.index)
            series.index.freq = inferred or "h"
        self.model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self.model_fit = self.model.fit(disp=False)
        return self.model_fit

    def predict(self, steps, **kwargs):
        if self.model_fit is None:
            raise ValueError("Model not fitted yet")
        return self.model_fit.forecast(steps=steps)

    def get_train_predictions(self):
        """In-sample fitted values (for train-set metrics). Returns Series with index aligned to training data."""
        if self.model_fit is None:
            raise ValueError("Model not fitted yet")
        return self.model_fit.fittedvalues
