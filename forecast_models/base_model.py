"""Base class for forecast models."""


class BaseModel:
    def fit(self, train_data):
        raise NotImplementedError

    def predict(self, input_data, horizon):
        raise NotImplementedError

    def reset(self):
        """Reinitialize model if needed (e.g. for expanding window)."""
        pass
