"""
Traffic Demand Prediction - Baseline Model
==========================================
Metric : score = max(0, 100 * R2(actual, predicted))
Target : demand  (continuous, small floats ~ 0-1 range)
Author : Baseline v1.0
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

try:
    import category_encoders as ce
    HAS_CE = True
except ImportError:
    HAS_CE = False
    print("[WARN] category_encoders not installed. Run: pip install category_encoders")


# =============================================================================
# Config
# =============================================================================
TRAIN_PATH      = "dataset/train.csv"
TEST_PATH       = "dataset/test.csv"
SUBMISSION_PATH = "submission.csv"

RANDOM_STATE  = 42
N_SPLITS      = 5       # TimeSeriesSplit folds
EARLY_STOP    = 50      # LightGBM early stopping rounds
LOG_TRANSFORM = True    # Apply log1p to demand before training

LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "n_estimators":      10_000,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 30,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      RANDOM_STATE,
    "verbosity":         -1,
    "n_jobs":            -1,
}

# Low-cardinality categoricals LightGBM handles natively as 'category' dtype
LOW_CARD_CATS = ["RoadType", "Weather", "LargeVehicles", "Landmarks"]
GEOHASH_COL   = "geohash"
GH_ENC_COL    = "geohash_te"


# =============================================================================
# 1. Load data
# =============================================================================
print("\n[1/7] Loading data ...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"  train : {train.shape}  |  test : {test.shape}")
print(f"  demand -> min={train['demand'].min():.4f}  "
      f"max={train['demand'].max():.4f}  "
      f"mean={train['demand'].mean():.4f}  "
      f"skew={train['demand'].skew():.2f}")

# Keep Index for submission construction; never use as a feature
test_index = test["Index"].copy()


# =============================================================================
# 2. Feature engineering helpers
# =============================================================================

def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar + cyclic features from the timestamp column."""
    df = df.copy()
    try:
        ts = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
    except Exception:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")

    df["hour"]       = ts.dt.hour
    df["dayofweek"]  = ts.dt.dayofweek   # 0=Mon, 6=Sun
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"]      = ts.dt.month

    # Cyclic encoding preserves periodicity (e.g. hour 23 is close to hour 0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def preprocess(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Drop identifiers, extract features, encode categoricals."""
    df = df.copy()

    # Drop row identifier -- never a predictive feature
    df.drop(columns=["Index"], inplace=True, errors="ignore")

    # --- Timestamp features ---
    df = parse_timestamp(df)

    # Check whether 'day' duplicates timestamp date; drop if >95% overlap
    if "day" in df.columns and "timestamp" in df.columns:
        try:
            ts_dates  = pd.to_datetime(df["timestamp"], infer_datetime_format=True).dt.date
            day_dates = pd.to_datetime(df["day"], errors="coerce").dt.date
            overlap   = (ts_dates == day_dates).mean()
            if overlap > 0.95:
                df.drop(columns=["day"], inplace=True)
                if is_train:
                    print("    'day' dropped (>95% overlap with timestamp)")
            else:
                # Keep day as ordinal day-of-year
                df["day_ordinal"] = day_dates.apply(
                    lambda d: d.timetuple().tm_yday if pd.notna(d) else np.nan
                )
                df.drop(columns=["day"], inplace=True)
        except Exception:
            df.drop(columns=["day"], inplace=True, errors="ignore")

    # Drop raw string timestamp -- features already extracted
    df.drop(columns=["timestamp"], inplace=True, errors="ignore")

    # --- Numeric imputation (LightGBM handles NaN natively, but be safe) ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if is_train and "demand" in num_cols:
        num_cols.remove("demand")
    for col in num_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    # --- Low-cardinality categoricals -> category dtype for LightGBM ---
    for col in LOW_CARD_CATS:
        if col in df.columns:
            df[col] = df[col].fillna("Missing").astype("category")

    return df


def fit_geohash_encoder(X_tr: pd.DataFrame, y_tr: pd.Series):
    """Fit a smoothed target encoder on the geohash column."""
    if not HAS_CE or GEOHASH_COL not in X_tr.columns:
        return None
    enc = ce.TargetEncoder(cols=[GEOHASH_COL], smoothing=10)
    enc.fit(X_tr[[GEOHASH_COL]], y_tr)
    return enc


def apply_geohash_encoding(X: pd.DataFrame, enc) -> pd.DataFrame:
    """Apply target encoding; drop raw geohash string."""
    X = X.copy()
    if enc is None or GEOHASH_COL not in X.columns:
        X.drop(columns=[GEOHASH_COL], inplace=True, errors="ignore")
        return X
    X[GH_ENC_COL] = enc.transform(X[[GEOHASH_COL]])[GEOHASH_COL]
    X.drop(columns=[GEOHASH_COL], inplace=True)
    return X


# =============================================================================
# 3. Preprocessing
# =============================================================================
print("\n[2/7] Preprocessing ...")
train = preprocess(train, is_train=True)
test  = preprocess(test,  is_train=False)
print(f"  Processed train : {train.shape}  |  test : {test.shape}")
print(f"  Categorical cols: {[c for c in LOW_CARD_CATS if c in train.columns]}")


# =============================================================================
# 4. Sort by time proxy (enables TimeSeriesSplit)
# =============================================================================
print("\n[3/7] Sorting by time proxy for time-series split ...")
# We sort by the extracted calendar features as a proxy for chronological order.
# If you have an absolute date column, sort on that instead.
sort_cols = [c for c in ["month", "dayofweek", "hour"] if c in train.columns]
train = train.sort_values(sort_cols).reset_index(drop=True)
print(f"  Sorted on: {sort_cols}")


# =============================================================================
# 5. Target transformation
# =============================================================================
print("\n[4/7] Target transformation ...")
TARGET = "demand"
y_raw  = train[TARGET].copy()

if LOG_TRANSFORM:
    y_train = np.log1p(y_raw)
    print(f"  log1p applied: skew {y_raw.skew():.3f} -> {y_train.skew():.3f}")
else:
    y_train = y_raw.copy()
    print("  No log transform.")

X_all = train.drop(columns=[TARGET])


# =============================================================================
# 6. Cross-validation  (TimeSeriesSplit -- 5 folds)
# =============================================================================
print("\n[5/7] Cross-validation (TimeSeriesSplit, 5 folds) ...")
# NOTE: TimeSeriesSplit trains on earlier time blocks and validates on later ones.
# This mimics production: the model never sees future data during training.
# If you find the data is purely shuffled with no temporal signal, swap to
# sklearn.model_selection.KFold instead.

tscv           = TimeSeriesSplit(n_splits=N_SPLITS)
fold_r2_scores = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_all), start=1):
    X_tr_raw = X_all.iloc[tr_idx].copy()
    X_val_raw = X_all.iloc[val_idx].copy()
    y_tr_raw  = y_raw.iloc[tr_idx]
    y_val_raw = y_raw.iloc[val_idx]
    y_tr_t    = y_train.iloc[tr_idx]

    # Geohash encoder fitted ONLY on training fold to prevent target leakage
    gh_enc = fit_geohash_encoder(X_tr_raw, y_tr_raw)
    X_tr   = apply_geohash_encoding(X_tr_raw,  gh_enc)
    X_val  = apply_geohash_encoding(X_val_raw, gh_enc)

    cat_cols = [c for c in LOW_CARD_CATS if c in X_tr.columns]

    # Use last 20% of the training fold as early-stopping holdout
    es_split    = int(0.8 * len(X_tr))
    X_es_tr     = X_tr.iloc[:es_split]
    X_es_val    = X_tr.iloc[es_split:]
    y_es_tr     = y_tr_t.iloc[:es_split]
    y_es_val    = y_tr_t.iloc[es_split:]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_es_tr, y_es_tr,
        eval_set=[(X_es_val, y_es_val)],
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    y_pred_t   = model.predict(X_val, num_iteration=model.best_iteration_)
    y_pred_raw = np.expm1(y_pred_t) if LOG_TRANSFORM else y_pred_t
    y_pred_raw = np.clip(y_pred_raw, 0.0, y_raw.max() * 1.1)

    fold_r2 = r2_score(y_val_raw, y_pred_raw)
    comp_score = max(0.0, 100.0 * fold_r2)
    fold_r2_scores.append(fold_r2)
    print(f"  Fold {fold} | best_iter={model.best_iteration_:>5} "
          f"| R2={fold_r2:.4f} | Score={comp_score:.2f}")

cv_r2_mean = float(np.mean(fold_r2_scores))
cv_r2_std  = float(np.std(fold_r2_scores))
cv_score   = max(0.0, 100.0 * cv_r2_mean)
print(f"\n  CV Mean R2 : {cv_r2_mean:.4f} +/- {cv_r2_std:.4f}")
print(f"  CV Score   : {cv_score:.2f} (max 100)")


# =============================================================================
# 7. Final model on all training data
# =============================================================================
print("\n[6/7] Training final model on full training set ...")
# NOTE: Geohash encoder is now fitted on the entire training set.
# This is acceptable for the final model (no validation fold to leak into).
gh_enc_final = fit_geohash_encoder(X_all, y_raw)
X_final      = apply_geohash_encoding(X_all.copy(), gh_enc_final)
X_test_final = apply_geohash_encoding(test.copy(),  gh_enc_final)

cat_cols_final = [c for c in LOW_CARD_CATS if c in X_final.columns]

# Last 15% used as early-stopping holdout for the final model
val_cut = int(0.85 * len(X_final))
final_model = lgb.LGBMRegressor(**LGBM_PARAMS)
final_model.fit(
    X_final.iloc[:val_cut], y_train.iloc[:val_cut],
    eval_set=[(X_final.iloc[val_cut:], y_train.iloc[val_cut:])],
    categorical_feature=cat_cols_final,
    callbacks=[
        lgb.early_stopping(EARLY_STOP, verbose=False),
        lgb.log_evaluation(period=200),
    ],
)
print(f"  Best iteration: {final_model.best_iteration_}")


# =============================================================================
# 8. Submission
# =============================================================================
print("\n[7/7] Generating submission ...")
test_preds_t = final_model.predict(
    X_test_final, num_iteration=final_model.best_iteration_
)
test_preds = np.expm1(test_preds_t) if LOG_TRANSFORM else test_preds_t

# Clip: demand must be non-negative; cap at 105% of max observed train demand
test_preds = np.clip(test_preds, 0.0, y_raw.max() * 1.05)

submission = pd.DataFrame({"Index": test_index.values, "demand": test_preds})
assert len(submission) == 41_778, f"Row count mismatch: {len(submission)}"
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"  [OK] {SUBMISSION_PATH} saved -- shape: {submission.shape}")
print(f"  demand -> min={submission['demand'].min():.5f}  "
      f"max={submission['demand'].max():.5f}  "
      f"mean={submission['demand'].mean():.5f}")


# =============================================================================
# Feature importance (top 15 by gain)
# =============================================================================
print("\n-- Top-15 Feature Importances (gain) --")
fi = (
    pd.DataFrame({
        "feature":    X_final.columns,
        "importance": final_model.feature_importances_,
    })
    .sort_values("importance", ascending=False)
)
print(fi.head(15).to_string(index=False))


# =============================================================================
# Next steps
# =============================================================================
print("""
+--------------------------------------------------------------+
|              NEXT ITERATION IMPROVEMENTS                     |
+--------------------------------------------------------------+
|  1. Lag features: lag1, rolling_mean_3h, rolling_mean_6h    |
|     -> groupby geohash, sort by time, then .shift(1)        |
|  2. Hyperparameter tuning with Optuna (30 trials, ~1 hour)  |
|  3. Add CatBoost as second model, blend OOF predictions     |
|  4. Residual analysis: inspect 20 worst errors per segment  |
|  5. Decode geohash -> lat/lon for spatial cluster features  |
|  6. Interaction: NumberofLanes x hour, Temp x Weather       |
|  7. Seed averaging: 3 seeds, average final test predictions |
+--------------------------------------------------------------+
""")
