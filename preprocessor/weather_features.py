"""Weather feature preprocessor (Open-Meteo API)."""
import pandas as pd
import requests


class WeatherFeatures:
    """Fetch weather data from Open-Meteo API and align with DataFrame timestamps."""

    def __init__(self, latitude, longitude, variables=None):
        self.latitude = latitude
        self.longitude = longitude
        self.variables = variables or ["temperature_2m"]
        self.weather_df = None

    def fit(self, df):
        return self

    def transform(self, df):
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for WeatherFeatures")
        need_start = df.index.min()
        need_end = df.index.max()
        if self.weather_df is None:
            self.weather_df = self._fetch_weather(need_start, need_end)
        else:
            cache_start, cache_end = self.weather_df.index.min(), self.weather_df.index.max()
            if need_start < cache_start or need_end > cache_end:
                fetch_start = min(need_start, cache_start)
                fetch_end = max(need_end, cache_end)
                self.weather_df = self._fetch_weather(fetch_start, fetch_end)
        df = df.join(self.weather_df, how="left")
        return df

    def inverse(self, df, **kwargs):
        return df

    def _fetch_weather(self, start_time, end_time):
        start_str = start_time.strftime("%Y-%m-%d")
        end_str = end_time.strftime("%Y-%m-%d")
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={self.latitude}&longitude={self.longitude}"
            f"&start_date={start_str}&end_date={end_str}"
            f"&hourly={','.join(self.variables)}"
            f"&timezone=UTC"
        )
        r = requests.get(url)
        if r.status_code != 200:
            raise ValueError(f"Weather API failed: {r.status_code} {r.text}")
        data = r.json()["hourly"]
        weather_df = pd.DataFrame(data)
        weather_df["time"] = pd.to_datetime(weather_df["time"])
        weather_df = weather_df.set_index("time")
        weather_df = weather_df[self.variables]
        weather_df = weather_df.ffill().bfill()
        return weather_df
