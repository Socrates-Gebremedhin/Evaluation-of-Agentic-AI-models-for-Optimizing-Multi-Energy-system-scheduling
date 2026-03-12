"""Scaling and transform preprocessors."""
import numpy as np
import pandas as pd

from .base import Preprocessor


class LogTransform(Preprocessor):
    def fit(self, series):
        pass

    def transform(self, series):
        self.shift = 1 - series.min() if series.min() <= 0 else 0
        return np.log(series + self.shift)

    def inverse(self, series, **kwargs):
        return np.exp(series) - self.shift


class Standardize(Preprocessor):
    def __init__(self, target_col=None):
        """
        target_col: optional. When fit on a DataFrame and inverse() is called with
        a 1D array (e.g. predictions), use this column's mean/std. If None, uses
        first column (for backward compatibility with single-column/series fit).
        """
        self.target_col = target_col

    def fit(self, series):
        self.mean = series.mean()
        self.std = series.std()

    def transform(self, series):
        return (series - self.mean) / self.std

    def inverse(self, series, target_col=None, **kwargs):
        """
        Inverse standardization. Use target_col when fit was on a DataFrame and
        series is a 1D array (e.g. predictions) so we use the correct column's mean/std.
        """
        target_col = target_col or getattr(self, "target_col", None)
        if isinstance(series, np.ndarray):
            mean = self.mean
            std = self.std
            if isinstance(mean, (pd.Series, pd.DataFrame)):
                if target_col is not None and target_col in mean.index:
                    mean = float(mean[target_col])
                    std = float(std[target_col])
                else:
                    mean = float(mean.iloc[0])
                    std = float(std.iloc[0])
            return series * std + mean
        return series * self.std + self.mean
