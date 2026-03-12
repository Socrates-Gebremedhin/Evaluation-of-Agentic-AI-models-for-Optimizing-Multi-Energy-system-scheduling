from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .base import Preprocessor


class TimeSeriesSplitter(Preprocessor):
    """
    Ratio-based splitter (kept for backwards compatibility).
    """

    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def fit(self, df):
        return self

    def transform(self, df):
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        return train, val, test


# ----------------- Date-based splitter from central config -----------------

DEFAULT_SPLIT_CONFIG = (
    Path(__file__).resolve().parent.parent / "split_config.csv"
)


def get_date_splitter(
    dataset: str,
    config_path: Optional[str] = None,
) -> Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Return a splitter function that slices a time series into train/val/test
    based on explicit date ranges stored in a CSV file.

    The CSV (by default `split_config.csv` in the project root) should have
    columns:

        dataset,train_start,train_end,val_start,val_end,test_start,test_end

    Dates are parsed with `pd.to_datetime` and are assumed to align with
    the DataFrame's DatetimeIndex.

    Usage in a pipeline:

        from preprocessor import get_date_splitter
        splitter = get_date_splitter("electricity")
        pipeline = {
            "splitter": splitter,
            ...
        }
    """
    path = Path(config_path) if config_path is not None else DEFAULT_SPLIT_CONFIG
    if not path.exists():
        raise FileNotFoundError(
            f"Split config file not found at {path}. Create it with columns: "
            "dataset,train_start,train_end,val_start,val_end,test_start,test_end."
        )

    cfg = pd.read_csv(path)
    if "dataset" not in cfg.columns:
        raise ValueError("split_config.csv must contain a 'dataset' column.")

    row = cfg.loc[cfg["dataset"] == dataset]
    if row.empty:
        raise ValueError(
            f"No split configuration found for dataset='{dataset}' in {path}."
        )
    row = row.iloc[0]

    keys = [
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    ]
    for k in keys:
        if k not in cfg.columns:
            raise ValueError(f"split_config.csv must contain column '{k}'.")

    dates = {k: pd.to_datetime(row[k]) for k in keys}

    def splitter(df: pd.DataFrame):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                "Date-based splitting requires a DatetimeIndex on the input DataFrame."
            )
        train = df.loc[dates["train_start"] : dates["train_end"]]
        val = df.loc[dates["val_start"] : dates["val_end"]]
        test = df.loc[dates["test_start"] : dates["test_end"]]
        return train, val, test

    return splitter
