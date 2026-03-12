"""Lag feature preprocessor."""
import pandas as pd

from .base import Preprocessor


class LagFeatures(Preprocessor):
    def __init__(self, target_col, lags, drop_na=True):
        self.target_col = target_col
        self.lags = lags
        self.drop_na = drop_na

    def fit(self, df):
        return self

    def transform(self, df):
        df = df.copy()
        for lag in self.lags:
            df[f"{self.target_col}_lag_{lag}"] = df[self.target_col].shift(lag)
        if self.drop_na:
            df = df.dropna()
        return df

    def inverse(self, df, **kwargs):
        return df
