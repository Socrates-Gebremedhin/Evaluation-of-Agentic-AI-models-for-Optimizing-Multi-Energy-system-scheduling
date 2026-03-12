"""Calendar / time-based feature preprocessor."""
import numpy as np
import pandas as pd


class CalendarFeatures:
    def __init__(self, features=None, cyclical=True):
        self.features = features or []
        self.cyclical = cyclical
        self.periods = {
            "hour": 24,
            "dayofweek": 7,
            "dayofmonth": 31,
            "dayofyear": 365,
            "weekofyear": 52,
            "month": 12,
        }

    def fit(self, df):
        return self

    def transform(self, df):
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("CalendarFeatures requires a DatetimeIndex")
        for feature in self.features:
            if feature == "hour":
                values = df.index.hour
            elif feature == "dayofweek":
                values = df.index.dayofweek
            elif feature == "dayofmonth":
                values = df.index.day
            elif feature == "dayofyear":
                values = df.index.dayofyear
            elif feature == "weekofyear":
                values = df.index.isocalendar().week.astype(int)
            elif feature == "month":
                values = df.index.month
            else:
                raise ValueError(f"Unsupported calendar feature: {feature}")
            if self.cyclical:
                period = self.periods[feature]
                df[f"{feature}_sin"] = np.sin(2 * np.pi * values / period)
                df[f"{feature}_cos"] = np.cos(2 * np.pi * values / period)
            else:
                df[feature] = values
        return df

    def inverse(self, df, **kwargs):
        return df
