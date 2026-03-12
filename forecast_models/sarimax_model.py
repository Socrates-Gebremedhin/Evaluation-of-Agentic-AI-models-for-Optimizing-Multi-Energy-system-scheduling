"""SARIMAX forecast model: SARIMA with exogenous regressors (e.g. temperature)."""
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAX_Model:
    """
    SARIMA with exogenous variables. Use when you have a DataFrame with
    target (e.g. electricity demand) and exog (e.g. temperature).
    Pipeline must provide exog_col and target_col; at predict time, future
    exog for the horizon must be passed.
    """

    def __init__(
        self,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        exog_col=None,
        target_col=None,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ):
        if not exog_col:
            raise ValueError("SARIMAX_Model requires exog_col (name of exogenous column).")
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_col = exog_col
        self.target_col = target_col
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model_fit = None

    def fit(self, df):
        df = pd.DataFrame(df).copy()
        if self.target_col is None:
            # infer target as first column that is not exog
            self.target_col = [c for c in df.columns if c != self.exog_col][0]
        endog = df[self.target_col]
        exog = df[[self.exog_col]]
        if isinstance(endog.index, pd.DatetimeIndex) and endog.index.freq is None:
            inferred = pd.infer_freq(endog.index)
            endog.index.freq = inferred or "h"
        self.model = SARIMAX(
            endog,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self.model_fit = self.model.fit(disp=False)
        return self.model_fit

    def predict(self, steps, exog=None):
        if self.model_fit is None:
            raise ValueError("Model not fitted yet")
        if exog is None:
            raise ValueError("SARIMAX_Model.predict requires exog (future exogenous for the horizon).")
        exog = pd.DataFrame(exog).reset_index(drop=True)
        if exog.shape[1] == 1 and self.exog_col not in exog.columns:
            exog.columns = [self.exog_col]
        if exog.shape[0] != steps:
            raise ValueError(f"exog must have {steps} rows (horizon); got {exog.shape[0]}.")
        return self.model_fit.forecast(steps=steps, exog=exog)

    def get_train_predictions(self):
        """In-sample fitted values (for train-set metrics). Returns Series with index aligned to training data."""
        if self.model_fit is None:
            raise ValueError("Model not fitted yet")
        return self.model_fit.fittedvalues
