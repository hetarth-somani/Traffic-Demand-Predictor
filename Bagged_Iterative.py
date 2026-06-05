"""
Traffic Demand Prediction – Bagged Iterative LightGBM (Model C + 5‑seed bagging)
================================================================================
Starting from pipeline_v33 Model C (iterative forecasting), this script
adds bagging: trains N LightGBM models with different random seeds and
averages their predictions at each iterative step.

All other settings (features, lags, encoder, hyperparameters) are unchanged.
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
# CONFIG
# =============================================================================
TRAIN_PATH = "dataset/train.csv"
TEST_PATH  = "dataset/test.csv"
BAGGING_SEEDS = [42, 101, 777, 2023, 1111, 2222, 3333, 4444, 5555, 9999]      # 5 models
EARLY_STOP = 100

LAG_STEPS      = [1, 2, 4, 8, 12, 96]
ROLLING_WINDOW = 4
LAG_COLS = (
    [f"lag{k}" for k in LAG_STEPS]
    + [f"rolling_mean_{ROLLING_WINDOW}", f"rolling_std_{ROLLING_WINDOW}",
       "expanding_mean"]
)

LOW_CARD_CATS = ["RoadType", "Weather", "LargeVehicles", "Landmarks"]
GEOHASH_COL   = "geohash"
GH_ENC_COL    = "geohash_te"
TARGET        = "demand"

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
    "verbosity":         -1,
    "n_jobs":            -1,
}

BASE_DATE = pd.Timestamp("2000-01-01")

# =============================================================================
# SECTION 1: LOAD DATA + RECONSTRUCT DATETIME
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 1: Loading data")
print("=" * 60)

train_raw = pd.read_csv("/kaggle/input/datasets/hetarthsomani/datasett/train.csv")
test_raw  = pd.read_csv("/kaggle/input/datasets/hetarthsomani/datasett/test.csv")
test_index = test_raw["Index"].copy()

def make_ts(df: pd.DataFrame) -> pd.Series:
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
# SECTION 2: CLEANING + FEATURE HELPERS
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
    df["slot_of_day"] = ts.dt.hour * 4 + ts.dt.minute // 15
    df["hour"]        = ts.dt.hour
    df["dayofweek"]   = ts.dt.dayofweek
    df["slot_sin"]    = np.sin(2 * np.pi * df["slot_of_day"] / 96)
    df["slot_cos"]    = np.cos(2 * np.pi * df["slot_of_day"] / 96)
    df["dow_sin"]     = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df

def finalize(df: pd.DataFrame) -> pd.DataFrame:
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
    common = [c for c in X_tr.columns if c in X_other.columns]
    return X_tr[common], X_other[common]

def train_lgbm(X_tr, y_tr, cat_cols, random_state):
    """Train a single LightGBM model with given seed."""
    params = LGBM_PARAMS.copy()
    params["random_state"] = random_state
    es_cut = int(0.80 * len(X_tr))
    model = lgb.LGBMRegressor(**params)
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

# =============================================================================
# LAG BUILDERS (unchanged)
# =============================================================================
def build_train_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([GEOHASH_COL, "_ts"]).reset_index(drop=True)
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
               .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0))
    )
    df["expanding_mean"] = (
        shifted.groupby(df[GEOHASH_COL], sort=False)
               .transform(lambda x: x.expanding().mean())
    )
    return df

# =============================================================================
# BAGGED ITERATIVE PREDICTOR (modified to accept multiple models)
# =============================================================================
def build_bagged_iterative_predictions(
    test_df: pd.DataFrame,
    train_portion: pd.DataFrame,
    models: list,          # list of (model, encoder, feature_cols, cat_cols, best_iter)
    max_demand: float,
    n_models: int
) -> np.ndarray:
    """
    Iterative one-step-ahead prediction using an ENSEMBLE of models.
    At each time slot:
        - build lag features from buffer
        - predict with every model
        - average predictions -> final demand for that slot
        - use the averaged predictions to update the buffer for next slot
    """
    test_sorted = test_df.sort_values("_ts").reset_index(drop=True)

    max_lag    = max(LAG_STEPS)
    tr_sorted  = train_portion.sort_values([GEOHASH_COL, "_ts"])
    buffers    = {}
    global_med = float(train_portion[TARGET].median())

    # Initialize buffers from training tail
    for gh, grp in tr_sorted.groupby(GEOHASH_COL, sort=False):
        demands = list(grp[TARGET].values)
        buffers[gh] = demands[-max_lag:] if len(demands) >= max_lag else demands

    unique_test_slots = sorted(test_sorted["_ts"].unique())
    n_test_slots = len(unique_test_slots)
    print(f"    Iterating over {n_test_slots} test time slots ({n_models} models) ...")

    preds_sorted = np.zeros(len(test_sorted))

    for slot_ts in unique_test_slots:
        slot_mask = test_sorted["_ts"] == slot_ts
        slot_rows = test_sorted[slot_mask].copy()

        # 1) Build lag features from buffers
        for col in LAG_COLS:
            slot_rows[col] = np.nan

        for idx, row in slot_rows.iterrows():
            gh  = row[GEOHASH_COL]
            buf = buffers.get(gh, None)

            if buf is None or len(buf) == 0:
                for k in LAG_STEPS:
                    slot_rows.at[idx, f"lag{k}"] = global_med
                slot_rows.at[idx, f"rolling_mean_{ROLLING_WINDOW}"] = global_med
                slot_rows.at[idx, f"rolling_std_{ROLLING_WINDOW}"]  = 0.0
                slot_rows.at[idx, "expanding_mean"] = global_med
                continue

            n = len(buf)
            for k in LAG_STEPS:
                slot_rows.at[idx, f"lag{k}"] = buf[-k] if n >= k else buf[0]

            tail_r = buf[-ROLLING_WINDOW:] if n >= ROLLING_WINDOW else buf
            slot_rows.at[idx, f"rolling_mean_{ROLLING_WINDOW}"] = float(np.mean(tail_r))
            slot_rows.at[idx, f"rolling_std_{ROLLING_WINDOW}"]  = (
                float(np.std(tail_r)) if len(tail_r) > 1 else 0.0
            )
            slot_rows.at[idx, "expanding_mean"] = float(np.mean(buf))

        # 2) Finalize features
        slot_feat = finalize(slot_rows.drop(columns=[TARGET], errors="ignore"))

        # 3) Predict with each model and average
        all_preds = np.zeros((len(slot_rows), n_models))
        for m_idx, (model, enc, feat_cols, cat_cols, best_iter) in enumerate(models):
            slot_enc = apply_encoder(slot_feat, enc)
            for fc in feat_cols:
                if fc not in slot_enc.columns:
                    slot_enc[fc] = 0.0
            slot_enc = slot_enc[feat_cols]
            for col in cat_cols:
                if col in slot_enc.columns:
                    slot_enc[col] = slot_enc[col].astype("category")
            preds_m = model.predict(slot_enc, num_iteration=best_iter)
            all_preds[:, m_idx] = preds_m

        slot_preds = np.mean(all_preds, axis=1)
        slot_preds = np.clip(slot_preds, 0.0, max_demand * 1.1)

        # Store
        slot_indices = test_sorted.index[slot_mask].tolist()
        preds_sorted[slot_indices] = slot_preds

        # 4) Update buffers with averaged predictions
        for (idx2, row2), pred in zip(slot_rows.iterrows(), slot_preds):
            gh = row2[GEOHASH_COL]
            if gh not in buffers:
                buffers[gh] = []
            buffers[gh].append(float(pred))
            if len(buffers[gh]) > max_lag:
                buffers[gh] = buffers[gh][-max_lag:]

    # Reorder to original test_df index
    result = np.zeros(len(test_df))
    sorted_to_orig = test_df.sort_values("_ts").index.tolist()
    for i, orig_idx in enumerate(sorted_to_orig):
        result[test_df.index.get_loc(orig_idx)] = preds_sorted[i]
    return result

# =============================================================================
# MAIN PIPELINE
# =============================================================================
# Clean data
train_clean = clean(train_raw)
test_clean  = clean(test_raw)
train_sorted = train_clean.sort_values("_ts").reset_index(drop=True)
d48 = train_sorted[train_sorted["day"] == 48].copy()
d48 = add_time_features(d48)   # adds slot_of_day
d48_slots = d48["slot_of_day"].values
mask_holdout = (d48_slots >= 9) & (d48_slots <= 55)
d48_train = d48[~mask_holdout].copy()
d48_test  = d48[mask_holdout].copy()

print(f"  Day48 morning‑holdout: train={len(d48_train)} rows, test={len(d48_test)} rows")
d49 = train_sorted[train_sorted["day"] == 49].copy()
max_demand = train_raw[TARGET].max()

print(f"\n  Day 48: {len(d48)} rows | Day 49: {len(d49)} rows")
print(f"  Test  : {len(test_clean)} rows")

# ---------------------------------------------------------------------------
# 1. Train bagged models on day48 (with true lags)
# ---------------------------------------------------------------------------
# Train new models on day48 training portion only
tr_lags_h = build_train_lags(d48_train)
tr_lags_h = tr_lags_h.dropna(subset=["lag1"]).reset_index(drop=True)
for col in LAG_COLS:
    if tr_lags_h[col].isna().any():
        tr_lags_h[col].fillna(tr_lags_h[col].median(), inplace=True)

X_tr_h = finalize(tr_lags_h.drop(columns=[TARGET]))
enc_h  = fit_encoder(X_tr_h, tr_lags_h[TARGET])
X_tr_h = apply_encoder(X_tr_h, enc_h)
cat_h  = [c for c in LOW_CARD_CATS if c in X_tr_h.columns]
y_tr_h = tr_lags_h[TARGET].values

models_val_long = []
print(f"\n  Training {len(BAGGING_SEEDS)} models for long‑horizon validation ...")
for seed in BAGGING_SEEDS:
    print(f"    Long‑val seed={seed} ...")
    model, best_iter = train_lgbm(X_tr_h, y_tr_h, cat_h, random_state=seed)
    models_val_long.append((model, enc_h, list(X_tr_h.columns), cat_h, best_iter))
print("  All models trained.\n")

# ---------------------------------------------------------------------------
# 2. Validate on day49 (iterative, using bagged ensemble)
# ---------------------------------------------------------------------------
y_val_long_true = d48_test[TARGET].values
print("  Running bagged iterative prediction on 47‑step morning holdout ...")
t0 = time.time()
val_long_preds = build_bagged_iterative_predictions(
    test_df = d48_test,
    train_portion = d48_train,
    models = models_val_long,
    max_demand = max_demand,
    n_models = len(models_val_long)
)
val_long_preds = clip(val_long_preds, max_demand)
r2_val_long = r2_score(y_val_long_true, val_long_preds)
score_val_long = max(0.0, 100.0 * r2_val_long)
print(f"  Long‑horizon validation (47 steps): R² = {r2_val_long:.4f}, Score = {score_val_long:.2f}")

# ---------------------------------------------------------------------------
# 3. Retrain final models on full training data (day48 + day49 first 9 slots)
# ---------------------------------------------------------------------------
print("\n  Training final bagged models on full training set ...")
tr_all_lags = build_train_lags(train_sorted)
n_b2 = len(tr_all_lags)
tr_all_lags = tr_all_lags.dropna(subset=["lag1"]).reset_index(drop=True)
for col in LAG_COLS:
    if tr_all_lags[col].isna().any():
        tr_all_lags[col].fillna(tr_all_lags[col].median(), inplace=True)

X_all_raw = finalize(tr_all_lags.drop(columns=[TARGET]))
enc_all = fit_encoder(X_all_raw, tr_all_lags[TARGET])
X_all = apply_encoder(X_all_raw, enc_all)
cat_all = [c for c in LOW_CARD_CATS if c in X_all.columns]
y_all = tr_all_lags[TARGET].values

models_final = []
for seed in BAGGING_SEEDS:
    print(f"    Training seed={seed} on all data ...")
    model, best_iter = train_lgbm(X_all, y_all, cat_all, random_state=seed)
    models_final.append((model, enc_all, list(X_all.columns), cat_all, best_iter))
print("  Final models trained.\n")

# ---------------------------------------------------------------------------
# 4. Iterative inference on test set
# ---------------------------------------------------------------------------
print("  Running bagged iterative prediction on test set ...")
t0 = time.time()
test_preds = build_bagged_iterative_predictions(
    test_df = test_clean.copy(),
    train_portion = train_sorted,
    models = models_final,
    max_demand = max_demand,
    n_models = len(models_final)
)
test_preds = clip(test_preds, max_demand)
print(f"  Test inference: {time.time()-t0:.1f}s")
print(f"  Predictions — min={test_preds.min():.5f}  max={test_preds.max():.5f}  mean={test_preds.mean():.5f}")

# ---------------------------------------------------------------------------
# 5. Save submission
# ---------------------------------------------------------------------------
sub = pd.DataFrame({"Index": test_index.values, "demand": test_preds})
sub.to_csv("submission_bagged.csv", index=False)
print(f"  [OK] submission_bagged.csv saved  shape={sub.shape}\n")

# ---------------------------------------------------------------------------
# 6. Final comparison
# ---------------------------------------------------------------------------
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(f"  Bagged iterative (this run) : {score_val:.2f}")
print(f"  Baseline Model C (v33)     : 91.68")
print(f"  Delta                       : {score_val - 91.68:+.2f}")
print(f"\n  Submit submission_bagged.csv.")
