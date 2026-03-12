"""Reshape tabular data to (X, y) sequences for LSTM-style models."""
import numpy as np

from .base import Preprocessor


class ReshapeToSequence(Preprocessor):
    def __init__(self, target_col, seq_len):
        self.target_col = target_col
        self.seq_len = seq_len

    def fit(self, df):
        return self

    def transform(self, df):
        df = df.copy()
        X, y = [], []
        for i in range(self.seq_len, len(df)):
            seq_x = df.iloc[i - self.seq_len : i].drop(columns=[self.target_col]).values
            seq_y = df.iloc[i][self.target_col]
            X.append(seq_x)
            y.append(seq_y)
        self.X = np.array(X)
        self.y = np.array(y)
        return self.X, self.y

    def inverse(self, df, **kwargs):
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
