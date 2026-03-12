"""German holiday feature using the holidays library."""
import pandas as pd

from .base import Preprocessor


class GermanHoliday(Preprocessor):
    """
    Add a binary column indicating German public holidays (using the holidays library).
    Requires: pip install holidays
    """

    def __init__(self, subdiv=None):
        """
        subdiv: optional state code for Germany (e.g. 'BY' for Bavaria).
               If None, uses federal holidays only.
        """
        self.subdiv = subdiv

    def fit(self, df):
        return self

    def transform(self, df):
        import holidays

        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("GermanHoliday requires a DatetimeIndex")

        years = range(int(df.index.year.min()), int(df.index.year.max()) + 1)
        kwargs = {"years": years}
        if self.subdiv is not None:
            kwargs["subdiv"] = self.subdiv
        de = holidays.country_holidays("DE", **kwargs)

        is_holiday = df.index.normalize().to_series().map(lambda d: d in de).values
        df["is_german_holiday"] = is_holiday.astype(float)
        return df

    def inverse(self, df, **kwargs):
        return df
