"""
Data loading and basic datetime reconstruction logic.
"""
import pandas as pd
from src import config

def make_ts(df: pd.DataFrame) -> pd.Series:
    """
    Reconstruct proper datetime: integer 'day' + 'H:MM' timestamp string.
    """
    ts_base = config.BASE_DATE + pd.to_timedelta(df["day"].astype(int), unit="D")
    parts = df["timestamp"].str.split(":", expand=True).astype(int)
    return (ts_base 
            + pd.to_timedelta(parts[0], unit="h") 
            + pd.to_timedelta(parts[1], unit="m"))

def load_data(train_path: str = config.TRAIN_PATH, test_path: str = config.TEST_PATH):
    """
    Load raw train and test files and inject reconstructed datetime `_ts`.
    Returns: train_raw, test_raw
    """
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    train_raw["_ts"] = make_ts(train_raw)
    test_raw["_ts"] = make_ts(test_raw)

    return train_raw, test_raw
