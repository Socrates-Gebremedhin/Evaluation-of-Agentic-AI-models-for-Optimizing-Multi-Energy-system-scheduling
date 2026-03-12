"""Persistence (naive) forecast model."""
import numpy as np

from .base_model import BaseModel


class PersistenceModel(BaseModel):
    def __init__(self, period=24):
        self.period = period
        self._last = None

    def fit(self, train_data):
        arr = np.asarray(train_data).reshape(-1)
        self._last = arr[-self.period :] if len(arr) >= self.period else arr.copy()

    def predict(self, horizon):
        """Predict next `horizon` steps using last stored values (call fit first)."""
        if self._last is None or len(self._last) == 0:
            return np.full(horizon, np.nan)
        last = self._last
        if len(last) < horizon:
            last = np.tile(last, (horizon // len(last)) + 1)
        return last[:horizon].astype(float)
