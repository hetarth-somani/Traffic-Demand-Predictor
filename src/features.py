"""
Feature engineering functions: temporal features, lag builders, and target encoders.
"""
import numpy as np
import pandas as pd
try:
    import category_encoders as ce
except ImportError:
    pass

from src import config

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers, cast categoricals, impute numeric NaN."""
    df = df.copy()
    df.drop(columns=["Index"], inplace=True, errors="ignore")
    df.drop(columns=["timestamp"], inplace=True, errors="ignore")
    
    for col in config.LOW_CARD_CATS:
        if col in df.columns:
            df[col] = df[col].fillna("Missing").astype("category")
            
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != config.TARGET]
    for col in num_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract slot-of-day, hour, cyclic encodings from '_ts'."""
    ts = df["_ts"]
    df = df.copy()
    df["slot_of_day"] = ts.dt.hour * 4 + ts.dt.minute // 15   # 0-95
    df["hour"]        = ts.dt.hour
    df["dayofweek"]   = ts.dt.dayofweek
    df["slot_sin"]    = np.sin(2 * np.pi * df["slot_of_day"] / 96)
    df["slot_cos"]    = np.cos(2 * np.pi * df["slot_of_day"] / 96)
    df["dow_sin"]     = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Add time features, impute numericals, and drop raw temporal columns."""
    df = add_time_features(df)
    df.drop(columns=["_ts", "timestamp", "day"], inplace=True, errors="ignore")
    
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != config.TARGET]
    for col in num_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df

def fit_encoder(X: pd.DataFrame, y: pd.Series) -> 'ce.TargetEncoder':
    """Fit geohash target encoder (smoothing=10) on training data."""
    enc = ce.TargetEncoder(cols=[config.GEOHASH_COL], smoothing=10)
    enc.fit(X[[config.GEOHASH_COL]], y)
    return enc

def apply_encoder(X: pd.DataFrame, enc: 'ce.TargetEncoder') -> pd.DataFrame:
    """Replace raw geohash strings with target-encoded float."""
    X = X.copy()
    if config.GEOHASH_COL not in X.columns:
        return X
    X[config.GH_ENC_COL] = enc.transform(X[[config.GEOHASH_COL]])[config.GEOHASH_COL]
    X.drop(columns=[config.GEOHASH_COL], inplace=True)
    return X

def build_train_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lag and rolling features for a training portion.
    Sorts by [geohash, _ts] internally; uses shift(k) within groups.
    LEAKAGE PREVENTION: no row ever sees its own or future demand.
    """
    df  = df.sort_values([config.GEOHASH_COL, "_ts"]).reset_index(drop=True)
    grp = df.groupby(config.GEOHASH_COL, sort=False)[config.TARGET]

    for k in config.LAG_STEPS:
        df[f"lag{k}"] = grp.shift(k)

    shifted = grp.shift(1)

    df[f"rolling_mean_{config.ROLLING_WINDOW}"] = (
        shifted.groupby(df[config.GEOHASH_COL], sort=False)
               .transform(lambda x: x.rolling(config.ROLLING_WINDOW, min_periods=1).mean())
    )
    df[f"rolling_std_{config.ROLLING_WINDOW}"] = (
        shifted.groupby(df[config.GEOHASH_COL], sort=False)
               .transform(lambda x: x.rolling(config.ROLLING_WINDOW, min_periods=1).std().fillna(0))
    )
    df["expanding_mean"] = (
        shifted.groupby(df[config.GEOHASH_COL], sort=False)
               .transform(lambda x: x.expanding().mean())
    )
    return df

def build_val_proxies(train_portion: pd.DataFrame, val_portion: pd.DataFrame) -> pd.DataFrame:
    """
    Build CONSTANT lag proxies for a validation/test set from the training tail.
    Used mainly for fast CV, not for final iterative prediction.
    """
    val = val_portion.copy()
    tr  = train_portion.sort_values([config.GEOHASH_COL, "_ts"])

    # Compute training lags to extract global medians
    tr_lags = build_train_lags(train_portion)
    global_meds = {}
    for col in config.LAG_COLS:
        if col in tr_lags.columns:
            global_meds[col] = float(tr_lags[col].median())
        else:
            global_meds[col] = 0.0

    for col in config.LAG_COLS:
        val[col] = np.nan

    for gh, grp in tr.groupby(config.GEOHASH_COL, sort=False):
        mask = val[config.GEOHASH_COL] == gh
        if not mask.any():
            continue
        demands = grp[config.TARGET].values
        n = len(demands)

        for k in config.LAG_STEPS:
            val.loc[mask, f"lag{k}"] = demands[-k] if n >= k else demands[0]

        tail_r = demands[-config.ROLLING_WINDOW:] if n >= config.ROLLING_WINDOW else demands
        val.loc[mask, f"rolling_mean_{config.ROLLING_WINDOW}"] = float(np.mean(tail_r))
        val.loc[mask, f"rolling_std_{config.ROLLING_WINDOW}"]  = (
            float(np.std(tail_r)) if len(tail_r) > 1 else 0.0
        )
        val.loc[mask, "expanding_mean"] = float(np.mean(demands))

    for col in config.LAG_COLS:
        val[col].fillna(global_meds.get(col, 0.0), inplace=True)

    return val
