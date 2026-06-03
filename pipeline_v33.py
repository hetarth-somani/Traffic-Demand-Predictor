"""
Traffic Demand Prediction - v3.3: Three-Model Ablation
=======================================================
Extends pipeline_v32.py with three experiments run in sequence:

  MODEL A: No-lag baseline
    - Drops all lag/rolling features
    - Keeps: geohash_te, slot_of_day, Temperature, NumberofLanes, categoricals
    - Diagnostic: if LB jumps to ~87, proxies were hurting. If stays ~85,
      the issue is in the feature set or distribution shift.

  MODEL B: Trend-adjusted proxies
    - Keeps lag features but adjusts each geohash's proxy with a
      short-window linear trend computed from the last few training rows.
    - proxy_lag1 = last_demand + slope * 1
    - proxy_lag4 = last_demand + slope * 4
    - This partially corrects for morning ramp-up in the test set.

  MODEL C: Iterative (one-step-ahead) forecasting
    - Predicts one time slot at a time in chronological order.
    - After predicting slot t, those predictions become the lag features
      for slot t+1. No proxies needed; this is the true lag structure.
    - Most accurate but O(n_slots) LightGBM inference calls.

All three use the same day-split validation (train=day48, val=day49-train)
and the same LGBM_PARAMS for fair comparison.

Outputs:
  submission_nolag.csv          (Model A)
  submission_trend.csv          (Model B)
  submission_iterative.csv      (Model C)
  experiment_log_v33.txt        (comparison table)
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import lightgbm as lgb
import category_encoders as ce


# =============================================================================
# CONFIG (identical to v3.2 for fair comparison)
# =============================================================================
TRAIN_PATH = "dataset/train.csv"
TEST_PATH  = "dataset/test.csv"

RANDOM_STATE = 42
EARLY_STOP   = 100

LAG_STEPS      = [1, 2, 4, 8, 12, 96]
ROLLING_WINDOW = 4   # 4 x 15min = 1 hour
LAG_COLS = (
    [f"lag{k}" for k in LAG_STEPS]
    + [f"rolling_mean_{ROLLING_WINDOW}", f"rolling_std_{ROLLING_WINDOW}",
       "expanding_mean"]
)

LOW_CARD_CATS = ["RoadType", "Weather", "LargeVehicles", "Landmarks"]
GEOHASH_COL   = "geohash"
GH_ENC_COL    = "geohash_te"
TARGET        = "demand"

# Trend proxy: number of recent training rows per geohash used to fit slope
TREND_WINDOW = 8   # last 8 slots (2 hours) for slope estimate

LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "n_estimators":      10_000,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      RANDOM_STATE,
    "verbosity":         -1,
    "n_jobs":            -1,
}

# Results accumulator for final comparison table
RESULTS = {}


# =============================================================================
# SECTION 1: LOAD + RECONSTRUCT DATETIME
# =============================================================================
print("\n" + "=" * 60)
print("SETUP: Loading data")
print("=" * 60)

train_raw  = pd.read_csv(TRAIN_PATH)
test_raw   = pd.read_csv(TEST_PATH)
test_index = test_raw["Index"].copy()

BASE_DATE = pd.Timestamp("2000-01-01")


def make_ts(df: pd.DataFrame) -> pd.Series:
    """Reconstruct proper datetime from integer 'day' + 'H:MM' timestamp."""
    ts_base = BASE_DATE + pd.to_timedelta(df["day"].astype(int), unit="D")
    parts   = df["timestamp"].str.split(":", expand=True).astype(int)
    return (ts_base
            + pd.to_timedelta(parts[0], unit="h")
            + pd.to_timedelta(parts[1], unit="m"))


train_raw["_ts"] = make_ts(train_raw)
test_raw["_ts"]  = make_ts(test_raw)

print(f"  Train: {len(train_raw)} rows | days={sorted(train_raw['day'].unique())}")
print(f"  Test : {len(test_raw)}  rows | days={sorted(test_raw['day'].unique())}")
print(f"  Train _ts: {train_raw['_ts'].min()} -> {train_raw['_ts'].max()}")
print(f"  Test  _ts: {test_raw['_ts'].min()}  -> {test_raw['_ts'].max()}")


# =============================================================================
# SECTION 2: CLEANING + SHARED HELPERS
# =============================================================================
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["Index"], inplace=True, errors="ignore")
    df.drop(columns=["timestamp"], inplace=True, errors="ignore")
    for col in LOW_CARD_CATS:
        if col in df.columns:
            df[col] = df[col].fillna("Missing").astype("category")
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != TARGET]
    for col in num_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["_ts"]
    df = df.copy()
    df["slot_of_day"] = ts.dt.hour * 4 + ts.dt.minute // 15  # 0-95
    df["hour"]        = ts.dt.hour
    df["dayofweek"]   = ts.dt.dayofweek
    df["slot_sin"]    = np.sin(2 * np.pi * df["slot_of_day"] / 96)
    df["slot_cos"]    = np.cos(2 * np.pi * df["slot_of_day"] / 96)
    df["dow_sin"]     = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Add time features and drop raw temporal columns."""
    df = add_time_features(df)
    df.drop(columns=["_ts", "timestamp"], inplace=True, errors="ignore")
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != TARGET]
    for col in num_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df


def fit_encoder(X: pd.DataFrame, y: pd.Series) -> ce.TargetEncoder:
    enc = ce.TargetEncoder(cols=[GEOHASH_COL], smoothing=10)
    enc.fit(X[[GEOHASH_COL]], y)
    return enc


def apply_encoder(X: pd.DataFrame, enc: ce.TargetEncoder) -> pd.DataFrame:
    X = X.copy()
    if GEOHASH_COL not in X.columns:
        return X
    X[GH_ENC_COL] = enc.transform(X[[GEOHASH_COL]])[GEOHASH_COL]
    X.drop(columns=[GEOHASH_COL], inplace=True)
    return X


def align_cols(X_tr: pd.DataFrame, X_other: pd.DataFrame):
    """Return both DataFrames restricted to their common columns (train order)."""
    common = [c for c in X_tr.columns if c in X_other.columns]
    return X_tr[common], X_other[common]


def train_lgbm(X_tr, y_tr, cat_cols):
    """
    Train LightGBM with chronological 80/20 inner split for early stopping.
    Returns (model, best_iteration).
    """
    es_cut = int(0.80 * len(X_tr))
    model  = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_tr.iloc[:es_cut], y_tr[:es_cut],
        eval_set=[(X_tr.iloc[es_cut:], y_tr[es_cut:])],
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return model, model.best_iteration_


def clip(preds, max_val):
    return np.clip(preds, 0.0, max_val * 1.10)


# Shared data splits
train_clean  = clean(train_raw)
test_clean   = clean(test_raw)
train_sorted = train_clean.sort_values("_ts").reset_index(drop=True)
d48          = train_sorted[train_sorted["day"] == 48].copy()
d49          = train_sorted[train_sorted["day"] == 49].copy()
max_demand   = train_raw[TARGET].max()

print(f"\n  Day 48: {len(d48)} rows | Day 49: {len(d49)} rows")
print(f"  Test  : {len(test_clean)} rows (day 49, continues after train)")


# =============================================================================
# SECTION 3: LAG FEATURE BUILDERS (shared across Model B and C)
# =============================================================================
def build_train_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all lag/rolling features for a training portion.
    Sorts by [geohash, _ts] internally. Uses shift(k) within groups (no leakage).
    """
    df  = df.sort_values([GEOHASH_COL, "_ts"]).reset_index(drop=True)
    grp = df.groupby(GEOHASH_COL, sort=False)[TARGET]

    for k in LAG_STEPS:
        df[f"lag{k}"] = grp.shift(k)

    shifted = grp.shift(1)

    df[f"rolling_mean_{ROLLING_WINDOW}"] = (
        shifted.groupby(df[GEOHASH_COL], sort=False)
               .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    df[f"rolling_std_{ROLLING_WINDOW}"] = (
        shifted.groupby(df[GEOHASH_COL], sort=False)
               .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1)
                                      .std().fillna(0))
    )
    df["expanding_mean"] = (
        shifted.groupby(df[GEOHASH_COL], sort=False)
               .transform(lambda x: x.expanding().mean())
    )
    return df


# =============================================================================
# MODEL A: NO-LAG BASELINE
# =============================================================================
# Purpose: diagnostic -- if LB score jumps to ~87 vs v3.2's ~89, then
# the lag features are helping and the proxy approach is not causing harm.
# If scores are similar, the limitation is in the feature set, not the lags.
#
# Features used: geohash_te, slot_of_day, slot_sin/cos, hour, dayofweek,
#                dow_sin/cos, day, NumberofLanes, Temperature, categoricals.

print("\n" + "=" * 60)
print("MODEL A: No-lag baseline (diagnostic)")
print("=" * 60)


def run_nolag(tr_df, val_df, te_df, label="nolag"):
    """
    Train and evaluate a model with ZERO lag/rolling features.
    All demand history is excluded; only static + time features remain.
    """
    y_val_true = val_df[TARGET].values

    # --- Training ---
    X_tr_raw = finalize(tr_df.drop(columns=[TARGET]))
    enc      = fit_encoder(X_tr_raw, tr_df[TARGET])
    X_tr     = apply_encoder(X_tr_raw, enc)
    y_tr     = tr_df[TARGET].values
    cat_cols = [c for c in LOW_CARD_CATS if c in X_tr.columns]

    model, best_iter = train_lgbm(X_tr, y_tr, cat_cols)

    # --- Validation ---
    X_val_raw = finalize(val_df.drop(columns=[TARGET]))
    X_val     = apply_encoder(X_val_raw, enc)
    X_tr, X_val = align_cols(X_tr, X_val)

    y_pred_val = clip(model.predict(X_val, num_iteration=best_iter), max_demand)
    r2         = r2_score(y_val_true, y_pred_val)
    score      = max(0.0, 100.0 * r2)
    print(f"  [{label}] Day-split R2={r2:.4f}  Score={score:.2f}  best_iter={best_iter}")

    # --- Final model on all train data ---
    X_tr_all = finalize(tr_df.drop(columns=[TARGET]))
    # Refit encoder on all training data for the final prediction
    enc_all   = fit_encoder(X_tr_all, tr_df[TARGET])
    X_tr_all  = apply_encoder(X_tr_all, enc_all)
    cat_all   = [c for c in LOW_CARD_CATS if c in X_tr_all.columns]
    model_all, bi_all = train_lgbm(X_tr_all, tr_df[TARGET].values, cat_all)

    X_te_raw  = finalize(te_df)
    X_te      = apply_encoder(X_te_raw, enc_all)
    X_tr_all, X_te = align_cols(X_tr_all, X_te)

    test_preds = clip(model_all.predict(X_te, num_iteration=bi_all), max_demand)
    return r2, score, test_preds


# Model A: day48 train, day49 val, no features from train target
r2_A, score_A, preds_A = run_nolag(
    tr_df  = d48.copy(),
    val_df = d49.copy(),
    te_df  = test_clean.copy(),
    label  = "A-nolag",
)
RESULTS["A_nolag"] = {"r2": r2_A, "score": score_A}

sub_A = pd.DataFrame({"Index": test_index.values, "demand": preds_A})
sub_A.to_csv("submission_nolag.csv", index=False)
print(f"  [OK] submission_nolag.csv saved  (mean_pred={preds_A.mean():.5f})")


# =============================================================================
# MODEL B: TREND-ADJUSTED PROXIES
# =============================================================================
# Instead of using a flat constant (last known demand) as the proxy,
# we fit a linear slope from the last TREND_WINDOW training rows per geohash
# and project forward by the step offset.
#
# For geohash g with training tail demands [d_{n-w}, ..., d_{n-1}, d_n]:
#   slope = linregress(x=[0..w], y=tail_demands).slope
#   adjusted_lag1   = d_n + slope * 1
#   adjusted_lag4   = d_n + slope * 4
#   adjusted_lag8   = d_n + slope * 8
#   adjusted_lag96  = d_n + slope * 96   (slope too uncertain; capped)
#
# RATIONALE: test set is a morning ramp-up (demand rises ~0.08 -> 0.12).
# The flat constant (d_n from 02:00) underestimates demand at, say, 09:00.
# The trend adjustment partially corrects this by extrapolating the recent
# direction of change observed in the training tail.
#
# CAVEATS:
#   - Only reliable for lags where the slope extrapolation is short (lag1, lag4)
#   - For lag96 (yesterday = 96 steps back), slope extrapolation is unreliable;
#     we keep the flat proxy with no trend adjustment for large offsets.
#   - Clamp adjusted values to [0, 1] (demand is bounded).

print("\n" + "=" * 60)
print("MODEL B: Trend-adjusted proxies")
print("=" * 60)
print(f"  Trend window: {TREND_WINDOW} slots ({TREND_WINDOW*15} min)")
print("  Method: fit linear slope on last TREND_WINDOW demands per geohash,")
print("  project forward by lag offset. Use flat proxy for lag96.")

# Max lag offset for which we trust the trend extrapolation
# Beyond this, the slope is too noisy to be useful
TREND_MAX_OFFSET = 12   # trust slope for lag1..lag12; use flat for lag96


def compute_trend_slope(demands: np.ndarray, window: int) -> float:
    """
    Fit a linear slope on the last 'window' demand values.
    Returns slope (demand units per slot). Returns 0 if insufficient data.
    """
    tail = demands[-window:] if len(demands) >= window else demands
    n    = len(tail)
    if n < 2:
        return 0.0
    x    = np.arange(n, dtype=float)
    # OLS slope: sum((x-x_mean)*(y-y_mean)) / sum((x-x_mean)^2)
    x_c  = x - x.mean()
    y_c  = tail - tail.mean()
    denom = (x_c * x_c).sum()
    return float((x_c * y_c).sum() / denom) if denom > 0 else 0.0


def build_trend_proxies(
    train_portion: pd.DataFrame,
    val_portion: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build lag proxies with linear trend adjustment for short offsets.
    Uses flat proxy for large offsets (lag96) where trend extrapolation
    is unreliable.

    LEAKAGE PREVENTION: reads only train_portion['demand'], never val demand.
    """
    val = val_portion.copy()
    tr  = train_portion.sort_values([GEOHASH_COL, "_ts"])

    # Build training lags to get global medians for unseen geohashes
    tr_lags     = build_train_lags(train_portion)
    global_meds = {
        col: float(tr_lags[col].median())
        for col in LAG_COLS
        if col in tr_lags.columns
    }

    for col in LAG_COLS:
        val[col] = np.nan

    for gh, grp in tr.groupby(GEOHASH_COL, sort=False):
        mask    = val[GEOHASH_COL] == gh
        if not mask.any():
            continue

        demands = grp[TARGET].values
        n       = len(demands)
        last    = demands[-1]   # most recent training demand for this geohash

        # Compute short-window slope from the recent training tail
        slope = compute_trend_slope(demands, TREND_WINDOW)
        # Clamp slope: don't let trend push adjusted proxy below 0 or above 1
        # We also dampen the slope for larger offsets to limit extrapolation error.
        # Dampening factor = 1 / (1 + offset / TREND_WINDOW) -- soft decay
        for k in LAG_STEPS:
            if k <= TREND_MAX_OFFSET:
                # Trend-adjusted: project 'k' steps forward from training tail
                # The proxy represents "what demand was k steps ago FROM the
                # perspective of the first test row". Since training ends right
                # before test begins, lag1 proxy = demand 1 step before test.
                # We adjust: last_demand + slope * k captures the upward ramp.
                dampening = 1.0 / (1.0 + k / TREND_WINDOW)
                adjusted  = last + slope * k * dampening
                adjusted  = float(np.clip(adjusted, 0.0, 1.0))
            else:
                # Flat proxy for large lags (e.g., lag96 = yesterday same slot)
                # Trend extrapolation 96 steps forward is meaningless
                adjusted = demands[-k] if n >= k else demands[0]
                adjusted = float(adjusted)
            val.loc[mask, f"lag{k}"] = adjusted

        # Rolling stats: use last ROLLING_WINDOW demands + trend
        tail_r = demands[-ROLLING_WINDOW:] if n >= ROLLING_WINDOW else demands
        # Trend-adjust the last rolling window values slightly
        adjusted_tail = np.array([
            float(np.clip(d + slope * (ROLLING_WINDOW - i), 0.0, 1.0))
            for i, d in enumerate(tail_r)
        ])
        val.loc[mask, f"rolling_mean_{ROLLING_WINDOW}"] = float(np.mean(adjusted_tail))
        val.loc[mask, f"rolling_std_{ROLLING_WINDOW}"]  = (
            float(np.std(adjusted_tail)) if len(adjusted_tail) > 1 else 0.0
        )
        val.loc[mask, "expanding_mean"] = float(np.mean(demands))

    for col in LAG_COLS:
        val[col].fillna(global_meds.get(col, 0.0), inplace=True)

    return val


def run_trend(tr_df, val_df, te_df, label="trend"):
    """
    Train with lag features where proxies are trend-adjusted.
    Training: standard build_train_lags (no change needed).
    Validation/Test: build_trend_proxies (slope-adjusted).
    """
    y_val_true = val_df[TARGET].values

    # Training lags (same as v3.2 -- no change)
    tr_lags = build_train_lags(tr_df)
    n_before = len(tr_lags)
    tr_lags  = tr_lags.dropna(subset=["lag1"]).reset_index(drop=True)
    print(f"  Rows dropped (lag1 NaN): {n_before - len(tr_lags)}")
    for col in LAG_COLS:
        if tr_lags[col].isna().any():
            tr_lags[col].fillna(tr_lags[col].median(), inplace=True)

    # Trend-adjusted proxies for validation
    val_lags = build_trend_proxies(tr_df, val_df)

    X_tr_raw  = finalize(tr_lags.drop(columns=[TARGET]))
    X_val_raw = finalize(val_lags.drop(columns=[TARGET]))

    enc      = fit_encoder(X_tr_raw, tr_lags[TARGET])
    X_tr     = apply_encoder(X_tr_raw,  enc)
    X_val    = apply_encoder(X_val_raw, enc)
    X_tr, X_val = align_cols(X_tr, X_val)
    cat_cols = [c for c in LOW_CARD_CATS if c in X_tr.columns]
    y_tr     = tr_lags[TARGET].values

    model, best_iter = train_lgbm(X_tr, y_tr, cat_cols)

    y_pred_val = clip(model.predict(X_val, num_iteration=best_iter), max_demand)
    r2         = r2_score(y_val_true, y_pred_val)
    score      = max(0.0, 100.0 * r2)
    print(f"  [{label}] Day-split R2={r2:.4f}  Score={score:.2f}  best_iter={best_iter}")

    # --- Final model on full training data + trend proxies for test ---
    tr_all_lags = build_train_lags(train_sorted)
    n_b = len(tr_all_lags)
    tr_all_lags = tr_all_lags.dropna(subset=["lag1"]).reset_index(drop=True)
    for col in LAG_COLS:
        if tr_all_lags[col].isna().any():
            tr_all_lags[col].fillna(tr_all_lags[col].median(), inplace=True)
    y_all = tr_all_lags[TARGET].values

    te_lags = build_trend_proxies(train_sorted, te_df)

    X_tr_all  = finalize(tr_all_lags.drop(columns=[TARGET]))
    X_te_raw  = finalize(te_lags)
    enc_all   = fit_encoder(X_tr_all, pd.Series(y_all))
    X_tr_all  = apply_encoder(X_tr_all, enc_all)
    X_te      = apply_encoder(X_te_raw, enc_all)
    X_tr_all, X_te = align_cols(X_tr_all, X_te)
    cat_all   = [c for c in LOW_CARD_CATS if c in X_tr_all.columns]

    model_all, bi_all = train_lgbm(X_tr_all, y_all, cat_all)
    test_preds = clip(model_all.predict(X_te, num_iteration=bi_all), max_demand)
    return r2, score, test_preds


r2_B, score_B, preds_B = run_trend(
    tr_df  = d48.copy(),
    val_df = d49.copy(),
    te_df  = test_clean.copy(),
    label  = "B-trend",
)
RESULTS["B_trend"] = {"r2": r2_B, "score": score_B}

sub_B = pd.DataFrame({"Index": test_index.values, "demand": preds_B})
sub_B.to_csv("submission_trend.csv", index=False)
print(f"  [OK] submission_trend.csv saved  (mean_pred={preds_B.mean():.5f})")


# =============================================================================
# MODEL C: ITERATIVE ONE-STEP-AHEAD FORECASTING
# =============================================================================
# True lag features require knowing demand at t-1, t-2, etc.
# For validation/test, we don't have future demand -- so we predict one
# time slot at a time, use those predictions as lag inputs for the next slot.
#
# ALGORITHM:
#   1. Train model on day48 (with true lags, same as before).
#   2. Sort test slots chronologically (ascending _ts).
#   3. Maintain a "rolling buffer" per geohash: a dict of recent demand values.
#      Initialize from the training tail.
#   4. For each test time slot t (in order):
#      a. For each geohash in this slot, read lag values from its buffer.
#      b. Predict demand for all rows in slot t simultaneously.
#      c. Append predicted demand to each geohash's buffer.
#      d. Proceed to slot t+1.
#
# LEAKAGE: None. Each slot only uses predictions from previous slots.
# ACCURACY: Far better than flat/trend proxies for long horizons, because
#   the model's own predictions are used iteratively to propagate recent state.
# COST: O(n_test_slots) inference calls instead of 1 (fast for <50 slots).
#
# IMPLEMENTATION NOTE: we predict all geohashes in a slot at once (batch),
# so we don't need to iterate over individual rows -- just over time slots.

print("\n" + "=" * 60)
print("MODEL C: Iterative one-step-ahead forecasting")
print("=" * 60)


def build_iterative_proxies(
    test_df: pd.DataFrame,
    train_portion: pd.DataFrame,
    model: lgb.LGBMRegressor,
    enc: ce.TargetEncoder,
    feature_cols: list,
    cat_cols: list,
) -> np.ndarray:
    """
    Predict demand for test rows slot by slot, feeding each slot's
    predictions back as lag inputs for the next slot.

    Args:
        test_df      : test rows with _ts, geohash, and all static features.
                       Must be sorted by _ts (chronological).
        train_portion: full training data (for initializing geohash buffers).
        model        : fitted LightGBM model.
        enc          : fitted geohash TargetEncoder.
        feature_cols : ordered list of feature columns the model expects.
        cat_cols     : categorical feature column names.

    Returns:
        numpy array of predictions for all test rows, in the ORIGINAL
        (unsorted) test_df row order.
    """
    # Sort test chronologically for slot-by-slot iteration
    test_sorted = test_df.sort_values("_ts").reset_index(drop=True)
    original_order = test_df.sort_values("_ts").index  # track original row idx

    # Initialize geohash demand buffers from training tail
    # Buffer holds the last max(LAG_STEPS) demand values per geohash,
    # in chronological order (oldest first).
    max_lag    = max(LAG_STEPS)
    tr_sorted  = train_portion.sort_values([GEOHASH_COL, "_ts"])
    buffers    = {}   # {geohash: deque-like list of demands, length >= max_lag}
    global_med = float(train_portion[TARGET].median())

    for gh, grp in tr_sorted.groupby(GEOHASH_COL, sort=False):
        demands = list(grp[TARGET].values)
        # Keep last max_lag values (sufficient for all lag lookups)
        buffers[gh] = demands[-max_lag:] if len(demands) >= max_lag else demands

    # Get all unique test time slots in order
    unique_test_slots = sorted(test_sorted["_ts"].unique())
    n_test_slots = len(unique_test_slots)
    print(f"    Iterating over {n_test_slots} test time slots ...")

    # Store predictions indexed by sorted test row index
    preds_sorted = np.zeros(len(test_sorted))

    for slot_ts in unique_test_slots:
        slot_mask = test_sorted["_ts"] == slot_ts
        slot_rows = test_sorted[slot_mask].copy()

        # ---------------------------------------------------------------
        # Build lag features for this slot from buffers
        # ---------------------------------------------------------------
        for col in LAG_COLS:
            slot_rows[col] = np.nan

        for idx, row in slot_rows.iterrows():
            gh  = row[GEOHASH_COL]
            buf = buffers.get(gh, None)

            if buf is None or len(buf) == 0:
                # Unseen geohash -- use global median for all lags
                for k in LAG_STEPS:
                    slot_rows.at[idx, f"lag{k}"] = global_med
                slot_rows.at[idx, f"rolling_mean_{ROLLING_WINDOW}"] = global_med
                slot_rows.at[idx, f"rolling_std_{ROLLING_WINDOW}"]  = 0.0
                slot_rows.at[idx, "expanding_mean"] = global_med
                continue

            n = len(buf)
            for k in LAG_STEPS:
                # lag k = demand k steps before the CURRENT slot
                # In the buffer, index -1 is the most recent (t-1), -2 is t-2, etc.
                slot_rows.at[idx, f"lag{k}"] = buf[-k] if n >= k else buf[0]

            tail_r = buf[-ROLLING_WINDOW:] if n >= ROLLING_WINDOW else buf
            slot_rows.at[idx, f"rolling_mean_{ROLLING_WINDOW}"] = float(np.mean(tail_r))
            slot_rows.at[idx, f"rolling_std_{ROLLING_WINDOW}"]  = (
                float(np.std(tail_r)) if len(tail_r) > 1 else 0.0
            )
            slot_rows.at[idx, "expanding_mean"] = float(np.mean(buf))

        # ---------------------------------------------------------------
        # Finalize features + predict for this slot
        # ---------------------------------------------------------------
        slot_feat = finalize(slot_rows.drop(columns=[TARGET], errors="ignore"))
        slot_enc  = apply_encoder(slot_feat, enc)

        # Align to model's expected feature columns
        for fc in feature_cols:
            if fc not in slot_enc.columns:
                slot_enc[fc] = 0.0
        slot_enc = slot_enc[feature_cols]

        # Cast categoricals
        for col in cat_cols:
            if col in slot_enc.columns:
                slot_enc[col] = slot_enc[col].astype("category")

        slot_preds = model.predict(slot_enc, num_iteration=model.best_iteration_)
        slot_preds = np.clip(slot_preds, 0.0, max_demand * 1.1)

        # Store predictions
        slot_indices  = test_sorted.index[slot_mask].tolist()
        preds_sorted[slot_indices] = slot_preds

        # ---------------------------------------------------------------
        # Update buffers with this slot's predictions for all geohashes
        # ---------------------------------------------------------------
        for (idx2, row2), pred in zip(slot_rows.iterrows(), slot_preds):
            gh = row2[GEOHASH_COL]
            if gh not in buffers:
                buffers[gh] = []
            buffers[gh].append(float(pred))
            # Keep buffer bounded to max_lag (memory efficiency)
            if len(buffers[gh]) > max_lag:
                buffers[gh] = buffers[gh][-max_lag:]

    # Re-index predictions back to original test_df order
    # test_sorted has same rows as test_df, just reindexed 0..N
    # We need to map sorted predictions back to original Index
    result = np.zeros(len(test_df))
    sorted_to_orig = test_df.sort_values("_ts").index.tolist()
    for i, orig_idx in enumerate(sorted_to_orig):
        result[test_df.index.get_loc(orig_idx)] = preds_sorted[i]

    return result


def run_iterative(tr_df, val_df, te_df, label="iterative"):
    """
    Train on tr_df with true lags, then predict val_df and te_df iteratively.
    """
    y_val_true = val_df[TARGET].values

    # --- Training: build true lags ---
    tr_lags = build_train_lags(tr_df)
    n_b     = len(tr_lags)
    tr_lags = tr_lags.dropna(subset=["lag1"]).reset_index(drop=True)
    print(f"  Rows dropped (lag1 NaN): {n_b - len(tr_lags)}")
    for col in LAG_COLS:
        if tr_lags[col].isna().any():
            tr_lags[col].fillna(tr_lags[col].median(), inplace=True)

    X_tr_raw = finalize(tr_lags.drop(columns=[TARGET]))
    enc      = fit_encoder(X_tr_raw, tr_lags[TARGET])
    X_tr     = apply_encoder(X_tr_raw, enc)
    cat_cols = [c for c in LOW_CARD_CATS if c in X_tr.columns]
    y_tr     = tr_lags[TARGET].values

    model, best_iter = train_lgbm(X_tr, y_tr, cat_cols)
    feature_cols = list(X_tr.columns)

    # --- Validation: iterative prediction ---
    print(f"  Running iterative prediction on validation set ...")
    t0_val = time.time()
    val_preds = build_iterative_proxies(
        test_df       = val_df.copy(),
        train_portion = tr_df,
        model         = model,
        enc           = enc,
        feature_cols  = feature_cols,
        cat_cols      = cat_cols,
    )
    val_preds = clip(val_preds, max_demand)
    r2    = r2_score(y_val_true, val_preds)
    score = max(0.0, 100.0 * r2)
    print(f"  [{label}] Val inference: {time.time()-t0_val:.1f}s")
    print(f"  [{label}] Day-split R2={r2:.4f}  Score={score:.2f}  best_iter={best_iter}")

    # --- Final model on full training data ---
    tr_all_lags = build_train_lags(train_sorted)
    n_b2 = len(tr_all_lags)
    tr_all_lags = tr_all_lags.dropna(subset=["lag1"]).reset_index(drop=True)
    for col in LAG_COLS:
        if tr_all_lags[col].isna().any():
            tr_all_lags[col].fillna(tr_all_lags[col].median(), inplace=True)

    X_tr_all  = finalize(tr_all_lags.drop(columns=[TARGET]))
    enc_all   = fit_encoder(X_tr_all, tr_all_lags[TARGET])
    X_tr_all  = apply_encoder(X_tr_all, enc_all)
    cat_all   = [c for c in LOW_CARD_CATS if c in X_tr_all.columns]
    y_all     = tr_all_lags[TARGET].values

    model_all, bi_all  = train_lgbm(X_tr_all, y_all, cat_all)
    feat_cols_all      = list(X_tr_all.columns)

    print(f"  Running iterative prediction on test set ...")
    t0_te = time.time()
    test_preds = build_iterative_proxies(
        test_df       = te_df.copy(),
        train_portion = train_sorted,
        model         = model_all,
        enc           = enc_all,
        feature_cols  = feat_cols_all,
        cat_cols      = cat_all,
    )
    test_preds = clip(test_preds, max_demand)
    print(f"  [{label}] Test inference: {time.time()-t0_te:.1f}s")
    return r2, score, test_preds


t_iter_start = time.time()
r2_C, score_C, preds_C = run_iterative(
    tr_df  = d48.copy(),
    val_df = d49.copy(),
    te_df  = test_clean.copy(),
    label  = "C-iter",
)
print(f"  Total iterative time: {time.time()-t_iter_start:.1f}s")
RESULTS["C_iterative"] = {"r2": r2_C, "score": score_C}

sub_C = pd.DataFrame({"Index": test_index.values, "demand": preds_C})
sub_C.to_csv("submission_iterative.csv", index=False)
print(f"  [OK] submission_iterative.csv saved  (mean_pred={preds_C.mean():.5f})")


# =============================================================================
# COMPARISON TABLE + EXPERIMENT LOG
# =============================================================================
v32_day_score = 89.06  # from previous run for reference

print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)
header = f"  {'Model':<30} {'Day-split R2':>12} {'Day-split Score':>15}"
print(header)
print("  " + "-" * (len(header) - 2))

model_labels = {
    "A_nolag":     "A: No-lag baseline",
    "B_trend":     "B: Trend-adjusted proxies",
    "C_iterative": "C: Iterative forecasting",
}

for key, info in RESULTS.items():
    label = model_labels.get(key, key)
    print(f"  {label:<30} {info['r2']:>12.4f} {info['score']:>15.2f}")

print(f"\n  Reference v3.2 (flat proxies) : {v32_day_score:.2f}")
print()
print("  INTERPRETATION:")
winner_score = max(v["score"] for v in RESULTS.values())
winner_key   = max(RESULTS, key=lambda k: RESULTS[k]["score"])
winner_label = model_labels.get(winner_key, winner_key)
print(f"  Best model: {winner_label}  (Score={winner_score:.2f})")
print()
print("  DIAGNOSTIC GUIDE:")
print("  If A_nolag >> v3.2: lag proxies were hurting -> use no-lag or iterative")
print("  If A_nolag ~= v3.2: feature/distribution issue -> focus on iterative (C)")
print("  If C_iterative >> B: iterative forecasting is worth the complexity")
print("  If B_trend ~= v3.2: slope correction is not capturing the trend well")

# Write log
with open("experiment_log_v33.txt", "w") as f:
    f.write("Traffic Demand Prediction - Experiment Log v3.3\n")
    f.write("=" * 60 + "\n\n")
    f.write("Three-model ablation study (day-split validation):\n\n")
    f.write(f"  v3.2 reference (flat proxies) : {v32_day_score:.2f}\n\n")
    for key, info in RESULTS.items():
        label = model_labels.get(key, key)
        f.write(f"  {label:<30} R2={info['r2']:.4f}  Score={info['score']:.2f}\n")
    f.write("\nSubmission files:\n")
    f.write("  submission_nolag.csv       (Model A)\n")
    f.write("  submission_trend.csv       (Model B)\n")
    f.write("  submission_iterative.csv   (Model C)\n")
    f.write("\nConfig:\n")
    f.write(f"  LAG_STEPS={LAG_STEPS}\n")
    f.write(f"  ROLLING_WINDOW={ROLLING_WINDOW}\n")
    f.write(f"  TREND_WINDOW={TREND_WINDOW}\n")
    f.write(f"  TREND_MAX_OFFSET={TREND_MAX_OFFSET}\n")
    f.write(f"  LGBM_PARAMS={LGBM_PARAMS}\n")

print("\n  [OK] experiment_log_v33.txt saved")
print("\n" + "=" * 60)
print("SUBMISSIONS READY")
print("=" * 60)
print("  submission_nolag.csv       -> upload first (diagnostic)")
print("  submission_trend.csv       -> if no-lag hurts, this might help")
print("  submission_iterative.csv   -> most likely the best (true lags)")
print("  submission_v32.csv         -> previous best (flat proxies)")
