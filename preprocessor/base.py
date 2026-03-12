"""Base class for all preprocessors."""


class Preprocessor:
    def fit(self, series):
        """Fit any parameters (e.g., mean/std for standardization)."""
        raise NotImplementedError

    def transform(self, series):
        """Transform series."""
        raise NotImplementedError

    def inverse(self, series):
        """Inverse transform."""
        raise NotImplementedError
