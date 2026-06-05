"""
tune_optuna.py
==============
Optuna hyperparameter tuning for the iterative-forecasting LightGBM pipeline.

Inherits all core helpers from pipeline_v33.py via selective import.
Since pipeline_v33.py is a script (not a module), we guard its side-effects
by monkey-patching the module-level execution block using importlib tricks.
In practice we simply re-define the shared helpers here and call them --
the logic is identical to pipeline_v33.py, just reorganised for import safety.

STRUCTURE
---------
  SECTION 0 : Imports and config
  SECTION 1 : Data loading + datetime reconstruction (same as v33)
  SECTION 2 : Shared feature-engineering helpers (from v33)
  SECTION 3 : Fold-aware lag construction (adapted from v33)
  SECTION 4 : Optuna objective (3-fold expanding-window on day48 slots)
  SECTION 5 : Run Optuna study (TPESampler + MedianPruner, 40 trials)
  SECTION 6 : Retrain best params on full day48
  SECTION 7 : Iterative forecasting on day49 (final honest evaluation)
  SECTION 8 : Submission + experiment log

Run:  python tune_optuna.py
"""

import warnings
warnings.filterwarnings("ignore")

import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import lightgbm as lgb
import category_encoders as ce
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# SECTION 0: CONFIG
# =============================================================================
TRAIN_PATH  = "/kaggle/input/datasets/hetarthsomani/datasett/train.csv"
TEST_PATH   = "/kaggle/input/datasets/hetarthsomani/datasett/test.csv"

RANDOM_STATE = 42

# Lag structure -- identical to pipeline_v33.py (Model C)
LAG_STEPS      = [1, 2, 4, 8, 12, 96]
ROLLING_WINDOW = 4          # 4 x 15min = 1 hour

LAG_COLS = (
    [f"lag{k}" for k in LAG_STEPS]
    + [f"rolling_mean_{ROLLING_WINDOW}", f"rolling_std_{ROLLING_WINDOW}",
       "expanding_mean"]
)

LOW_CARD_CATS = ["RoadType", "Weather", "LargeVehicles", "Landmarks"]
GEOHASH_COL   = "geohash"
GH_ENC_COL    = "geohash_te"
TARGET        = "demand"

BASE_DATE = pd.Timestamp("2000-01-01")

# CV setup: 3 expanding-window folds on day48 slots
# Each validation window = 16 consecutive slots (= 4 hours at 15-min resolution)
N_CV_FOLDS       = 3
VAL_WINDOW_SLOTS = 16       # 16 slots = 4 hours

# Optuna config
N_TRIALS         = 40       # total Optuna trials (MedianPruner will kill bad ones early)
N_STARTUP_TRIALS = 5        # pruner waits for this many completed trials before pruning
N_WARMUP_STEPS   = 1        # pruner waits this many folds before reporting to prune

# Fixed LightGBM params (not tuned)
LGBM_FIXED = {
    "objective":     "regression",
    "metric":        "rmse",
    "boosting_type": "gbdt",
    "n_estimators":  2000,      # early stopping will cut this short
    "verbosity":     -1,
    "n_jobs":        -1,
    "random_state":  RANDOM_STATE,
}

# Early stopping patience for each fold's inner sub-split
EARLY_STOP = 50

# Baseline day-split score from pipeline_v33.py Model C
BASELINE_SCORE = 91.68


# =============================================================================
# SECTION 1: DATA LOADING + DATETIME RECONSTRUCTION
# (same logic as pipeline_v33.py -- duplicated here for import safety)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 1: Loading data")
print("=" * 60)

train_raw  = pd.read_csv(TRAIN_PATH)
test_raw   = pd.read_csv(TEST_PATH)
test_index = test_raw["Index"].copy()


def make_ts(df: pd.DataFrame) -> pd.Series:
    """Reconstruct proper datetime: integer 'day' + 'H:MM' timestamp string."""
    ts_base = BASE_DATE + pd.to_timedelta(df["day"].astype(int), unit="D")
    parts   = df["timestamp"].str.split(":", expand=True).astype(int)
    return (ts_base
            + pd.to_timedelta(parts[0], unit="h")
            + pd.to_timedelta(parts[1], unit="m"))


train_raw["_ts"] = make_ts(train_raw)
test_raw["_ts"]  = make_ts(test_raw)

print(f"  Train: {len(train_raw)} rows | days={sorted(train_raw['day'].unique())}")
print(f"  Test : {len(test_raw)} rows  | days={sorted(test_raw['day'].unique())}")
print(f"  Train _ts: {train_raw['_ts'].min()} -> {train_raw['_ts'].max()}")
print(f"  Test  _ts: {test_raw['_ts'].min()}  -> {test_raw['_ts'].max()}")


# =============================================================================
# SECTION 2: SHARED FEATURE-ENGINEERING HELPERS
# (identical to pipeline_v33.py)
# =============================================================================
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers, cast categoricals, impute numeric NaN."""
    df = df.copy()
    df.drop(columns=["Index"],     inplace=True, errors="ignore")
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
    """Fit geohash target encoder (smoothing=10) on training data."""
    enc = ce.TargetEncoder(cols=[GEOHASH_COL], smoothing=10)
    enc.fit(X[[GEOHASH_COL]], y)
    return enc


def apply_encoder(X: pd.DataFrame, enc: ce.TargetEncoder) -> pd.DataFrame:
    """Replace raw geohash strings with target-encoded float."""
    X = X.copy()
    if GEOHASH_COL not in X.columns:
        return X
    X[GH_ENC_COL] = enc.transform(X[[GEOHASH_COL]])[GEOHASH_COL]
    X.drop(columns=[GEOHASH_COL], inplace=True)
    return X


def align_cols(X_ref: pd.DataFrame, X_other: pd.DataFrame):
    """Restrict both DataFrames to their common columns in ref's order."""
    common = [c for c in X_ref.columns if c in X_other.columns]
    return X_ref[common], X_other[common]


def clip_preds(preds: np.ndarray, max_val: float) -> np.ndarray:
    return np.clip(preds, 0.0, max_val * 1.10)


# =============================================================================
# SECTION 3: LAG FEATURE BUILDERS
# (identical to pipeline_v33.py)
# =============================================================================
def build_train_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lag and rolling features for a training portion.
    Sorts by [geohash, _ts] internally; uses shift(k) within groups.
    LEAKAGE PREVENTION: no row ever sees its own or future demand.
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


def build_val_proxies(
    train_portion: pd.DataFrame,
    val_portion: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build CONSTANT lag proxies for a validation/test set from the training tail.
    LEAKAGE PREVENTION: only reads train_portion['demand'], never val demand.

    For each geohash, uses the last k demand values from training as
    constant proxies for lag_k across ALL validation rows of that geohash.
    This is the same approach used for CV in pipeline_v33.py (not the
    iterative version, which is reserved for the final evaluation).
    """
    val = val_portion.copy()
    tr  = train_portion.sort_values([GEOHASH_COL, "_ts"])

    # Compute training lags to extract global medians
    tr_lags = build_train_lags(train_portion)
    global_meds = {}
    for col in LAG_COLS:
        if col in tr_lags.columns:
            global_meds[col] = float(tr_lags[col].median())
        else:
            global_meds[col] = 0.0

    for col in LAG_COLS:
        val[col] = np.nan

    for gh, grp in tr.groupby(GEOHASH_COL, sort=False):
        mask    = val[GEOHASH_COL] == gh
        if not mask.any():
            continue
        demands = grp[TARGET].values
        n       = len(demands)

        for k in LAG_STEPS:
            val.loc[mask, f"lag{k}"] = demands[-k] if n >= k else demands[0]

        tail_r = demands[-ROLLING_WINDOW:] if n >= ROLLING_WINDOW else demands
        val.loc[mask, f"rolling_mean_{ROLLING_WINDOW}"] = float(np.mean(tail_r))
        val.loc[mask, f"rolling_std_{ROLLING_WINDOW}"]  = (
            float(np.std(tail_r)) if len(tail_r) > 1 else 0.0
        )
        val.loc[mask, "expanding_mean"] = float(np.mean(demands))

    for col in LAG_COLS:
        val[col].fillna(global_meds.get(col, 0.0), inplace=True)

    return val


def build_iterative_proxies(
    test_df: pd.DataFrame,
    train_portion: pd.DataFrame,
    model: lgb.LGBMRegressor,
    enc: ce.TargetEncoder,
    feature_cols: list,
    cat_cols: list,
    max_demand: float,
    verbose: bool = True,
) -> np.ndarray:
    """
    Slot-by-slot iterative forecasting (Model C from pipeline_v33.py).
    Predicts one time slot at a time; each slot's predictions become
    lag inputs for the next slot via in-memory geohash buffers.

    LEAKAGE PREVENTION: each slot only reads buffer values from
    slots that have already been predicted.
    """
    max_lag     = max(LAG_STEPS)
    tr_sorted   = train_portion.sort_values([GEOHASH_COL, "_ts"])
    buffers     = {}
    global_med  = float(train_portion[TARGET].median())

    for gh, grp in tr_sorted.groupby(GEOHASH_COL, sort=False):
        demands = list(grp[TARGET].values)
        buffers[gh] = demands[-max_lag:] if len(demands) >= max_lag else demands

    test_sorted  = test_df.sort_values("_ts").reset_index(drop=True)
    unique_slots = sorted(test_sorted["_ts"].unique())
    preds_sorted = np.zeros(len(test_sorted))

    if verbose:
        print(f"      Iterating over {len(unique_slots)} test time slots ...")

    for slot_ts in unique_slots:
        slot_mask = test_sorted["_ts"] == slot_ts
        slot_rows = test_sorted[slot_mask].copy()

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

        slot_feat = finalize(slot_rows.drop(columns=[TARGET], errors="ignore"))
        slot_enc  = apply_encoder(slot_feat, enc)

        for fc in feature_cols:
            if fc not in slot_enc.columns:
                slot_enc[fc] = 0.0
        slot_enc = slot_enc[feature_cols]

        for col in cat_cols:
            if col in slot_enc.columns:
                slot_enc[col] = slot_enc[col].astype("category")

        slot_preds = model.predict(slot_enc, num_iteration=model.best_iteration_)
        slot_preds = np.clip(slot_preds, 0.0, max_demand * 1.1)

        slot_indices = test_sorted.index[slot_mask].tolist()
        preds_sorted[slot_indices] = slot_preds

        for (idx2, row2), pred in zip(slot_rows.iterrows(), slot_preds):
            gh = row2[GEOHASH_COL]
            if gh not in buffers:
                buffers[gh] = []
            buffers[gh].append(float(pred))
            if len(buffers[gh]) > max_lag:
                buffers[gh] = buffers[gh][-max_lag:]

    # Map back to original test_df row order
    result = np.zeros(len(test_df))
    for i, orig_idx in enumerate(test_df.sort_values("_ts").index.tolist()):
        result[test_df.index.get_loc(orig_idx)] = preds_sorted[i]

    return result


# =============================================================================
# DATA PREPARATION
# =============================================================================
train_clean  = clean(train_raw)
test_clean   = clean(test_raw)
train_sorted = train_clean.sort_values("_ts").reset_index(drop=True)

# Day 48 (full day) used exclusively for Optuna CV
# Day 49 train portion (00:00-02:00) used for final honest evaluation
d48 = train_sorted[train_sorted["day"] == 48].copy().reset_index(drop=True)
d49 = train_sorted[train_sorted["day"] == 49].copy().reset_index(drop=True)

max_demand = float(train_raw[TARGET].max())

# Ordered list of unique day48 time slots (96 total, 0:00-23:45)
unique_slots_d48 = sorted(d48["_ts"].unique())
n_slots_d48      = len(unique_slots_d48)   # should be 96

print(f"\n  Day 48: {len(d48)} rows | {n_slots_d48} unique slots")
print(f"  Day 49: {len(d49)} rows  | {len(sorted(d49['_ts'].unique()))} unique slots")


# =============================================================================
# SECTION 4: OPTUNA OBJECTIVE
# =============================================================================
# CV strategy: 3 expanding-window folds on day48 slot indices.
#
# fold 0: train = slots[0 .. split0-1],   val = slots[split0 .. split0+15]
# fold 1: train = slots[0 .. split1-1],   val = slots[split1 .. split1+15]
# fold 2: train = slots[0 .. split2-1],   val = slots[split2 .. split2+15]
#
# We space the val windows evenly across the second half of day48
# so that each fold's val covers a different portion of the day.
# Concretely:
#   split points: 48, 64, 80
#   fold 0 val: slots 48-63  (12:00 - 15:45)
#   fold 1 val: slots 64-79  (16:00 - 19:45)
#   fold 2 val: slots 80-95  (20:00 - 23:45)
#
# Each fold trains on ALL slots preceding its val window (expanding window).
# The val set uses CONSTANT PROXIES (same as build_val_proxies) -- NOT
# iterative forecasting, which would be too slow for 40 * 3 = 120 fold runs.
# Iterative forecasting is reserved for the final evaluation only.
#
# LEAKAGE PREVENTION (inside objective):
#   - build_train_lags called ONLY on the training portion of each fold.
#   - Target encoder fitted ONLY on the training portion of each fold.
#   - Val set receives only constant proxies derived from the training tail.
#   - Early stopping uses a chronological 80/20 sub-split of the training fold.
#
# PRUNING:
#   MedianPruner uses the intermediate R2 value reported after each fold.
#   A trial is pruned if its running mean R2 is below the median of
#   completed trials at the same step (fold index).

def make_cv_folds(unique_slots: list, n_folds: int, val_window: int) -> list:
    """
    Build expanding-window fold definitions from a sorted list of time slots.

    Returns a list of (train_slots, val_slots) tuples.
    Train slots for fold i = all slots before the val window.
    Val window is spaced evenly in the second half of the slot list.
    """
    n = len(unique_slots)
    # Place the first val window start at 50% of slots to guarantee
    # enough training data for each fold.
    first_val_start = n // 2
    step = (n - first_val_start - val_window) // max(n_folds - 1, 1)

    folds = []
    for i in range(n_folds):
        val_start = first_val_start + i * step
        val_end   = val_start + val_window
        if val_end > n:
            break
        tr_slots  = unique_slots[:val_start]
        val_slots = unique_slots[val_start:val_end]
        folds.append((tr_slots, val_slots))
    return folds


CV_FOLDS = make_cv_folds(unique_slots_d48, N_CV_FOLDS, VAL_WINDOW_SLOTS)

print(f"\n  CV folds (expanding window):")
for i, (tr_s, val_s) in enumerate(CV_FOLDS):
    t0 = pd.Timestamp(tr_s[0]).strftime("%H:%M")
    t1 = pd.Timestamp(tr_s[-1]).strftime("%H:%M")
    v0 = pd.Timestamp(val_s[0]).strftime("%H:%M")
    v1 = pd.Timestamp(val_s[-1]).strftime("%H:%M")
    print(f"    Fold {i}: train {t0}-{t1} ({len(tr_s)} slots) | "
          f"val {v0}-{v1} ({len(val_s)} slots)")


def run_single_fold(
    tr_slots: list,
    val_slots: list,
    params: dict,
) -> float:
    """
    Train and evaluate one CV fold. Returns R2 on original demand scale.

    NO global data used inside this function -- all features are built
    strictly from the training portion of this fold.
    """
    # Subset rows belonging to this fold's slot windows
    tr_fold  = d48[d48["_ts"].isin(tr_slots)].copy()
    val_fold = d48[d48["_ts"].isin(val_slots)].copy()
    y_val    = val_fold[TARGET].values

    if len(tr_fold) < 100 or len(val_fold) < 10:
        return 0.0

    # --- Build lag features on TRAINING portion only ---
    tr_lags = build_train_lags(tr_fold)
    tr_lags = tr_lags.dropna(subset=["lag1"]).reset_index(drop=True)
    for col in LAG_COLS:
        if tr_lags[col].isna().any():
            tr_lags[col].fillna(tr_lags[col].median(), inplace=True)

    if len(tr_lags) < 50:
        return 0.0

    # --- Build CONSTANT lag proxies for validation ---
    # (fast; iterative forecasting only used in final evaluation)
    val_lags = build_val_proxies(
        train_portion = tr_fold,
        val_portion   = val_fold,
    )

    # --- Feature matrices ---
    X_tr_raw  = finalize(tr_lags.drop(columns=[TARGET]))
    X_val_raw = finalize(val_lags.drop(columns=[TARGET]))

    # Target encoder fitted on TRAINING fold only
    enc   = fit_encoder(X_tr_raw, tr_lags[TARGET])
    X_tr  = apply_encoder(X_tr_raw,  enc)
    X_val = apply_encoder(X_val_raw, enc)
    X_tr, X_val = align_cols(X_tr, X_val)

    cat_cols = [c for c in LOW_CARD_CATS if c in X_tr.columns]
    y_tr     = tr_lags[TARGET].values

    # --- Train LightGBM with trial's hyperparameters ---
    # Early stopping: chronological 80/20 split INSIDE the training fold.
    # We NEVER use the val_fold rows for early stopping.
    es_cut = int(0.80 * len(X_tr))
    if es_cut < 20 or (len(X_tr) - es_cut) < 10:
        return 0.0

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

    y_pred = clip_preds(
        model.predict(X_val, num_iteration=model.best_iteration_),
        max_demand,
    )
    return float(r2_score(y_val, y_pred))


def optuna_objective(trial: optuna.Trial) -> float:
    """
    Optuna objective: mean CV R2 over N_CV_FOLDS folds on day48 slots.

    PRUNING: reports intermediate R2 after each fold so MedianPruner
    can kill unpromising trials after fold 0 or 1 without waiting for
    all 3 folds to complete.

    Returns: mean R2 (higher is better; direction='maximize').
    """
    # --- Hyperparameter search space ---
    params = {
        **LGBM_FIXED,
        "num_leaves":        trial.suggest_int(   "num_leaves",        31,   255, step=10),
        "min_child_samples": trial.suggest_int(   "min_child_samples",  5,   100),
        "learning_rate":     trial.suggest_float(  "learning_rate",     0.01, 0.10, log=True),
        "subsample":         trial.suggest_float(  "subsample",         0.60, 1.00),
        "colsample_bytree":  trial.suggest_float(  "colsample_bytree",  0.60, 1.00),
        "reg_alpha":         trial.suggest_float(  "reg_alpha",         1e-5, 10.0, log=True),
        "reg_lambda":        trial.suggest_float(  "reg_lambda",        1e-5, 10.0, log=True),
        "min_split_gain":    trial.suggest_float(  "min_split_gain",    0.00, 1.00),
    }

    fold_r2s = []

    for fold_idx, (tr_slots, val_slots) in enumerate(CV_FOLDS):
        r2 = run_single_fold(tr_slots, val_slots, params)
        fold_r2s.append(r2)

        # Report running mean to Optuna for MedianPruner
        running_mean = float(np.mean(fold_r2s))
        trial.report(running_mean, step=fold_idx)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_r2s))


# =============================================================================
# SECTION 5: RUN OPTUNA STUDY
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5: Optuna hyperparameter search")
print("=" * 60)
print(f"  Trials    : {N_TRIALS}")
print(f"  CV folds  : {N_CV_FOLDS} (expanding window, 16 slots each)")
print(f"  Sampler   : TPESampler(seed={RANDOM_STATE})")
print(f"  Pruner    : MedianPruner(startup={N_STARTUP_TRIALS}, warmup={N_WARMUP_STEPS})")
print(f"  Objective : maximise mean R2 across folds (original demand scale)")
print()

sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
pruner  = optuna.pruners.MedianPruner(
    n_startup_trials = N_STARTUP_TRIALS,
    n_warmup_steps   = N_WARMUP_STEPS,
    interval_steps   = 1,
)
study = optuna.create_study(
    direction = "maximize",
    sampler   = sampler,
    pruner    = pruner,
)

t_optuna = time.time()
study.optimize(optuna_objective, n_trials=N_TRIALS, show_progress_bar=False)
t_optuna_elapsed = time.time() - t_optuna

# Summary
completed = [t for t in study.trials
             if t.state == optuna.trial.TrialState.COMPLETE]
pruned    = [t for t in study.trials
             if t.state == optuna.trial.TrialState.PRUNED]

best_trial  = study.best_trial
best_cv_r2  = best_trial.value
best_cv_score = max(0.0, 100.0 * best_cv_r2)
best_params = best_trial.params

print(f"  Optuna finished in {t_optuna_elapsed:.1f}s")
print(f"  Total trials   : {len(study.trials)}")
print(f"  Completed      : {len(completed)}")
print(f"  Pruned         : {len(pruned)}")
print(f"  Best trial #   : {best_trial.number}")
print(f"  Best CV R2     : {best_cv_r2:.4f}  (Score: {best_cv_score:.2f})")
print()
print("  Best hyperparameters:")
for k, v in best_params.items():
    print(f"    {k:25s} = {v}")


# =============================================================================
# SECTION 6: RETRAIN BEST PARAMS ON FULL DAY48
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 6: Final model -- retrain on full day48")
print("=" * 60)

# Build true lag features on all of day48 (no CV split needed here)
tr_final = build_train_lags(d48)
n_before = len(tr_final)
tr_final = tr_final.dropna(subset=["lag1"]).reset_index(drop=True)
print(f"  Rows dropped (lag1 NaN): {n_before - len(tr_final)}")

for col in LAG_COLS:
    if tr_final[col].isna().any():
        tr_final[col].fillna(tr_final[col].median(), inplace=True)

y_final = tr_final[TARGET].values

X_tr_raw = finalize(tr_final.drop(columns=[TARGET]))
enc_final = fit_encoder(X_tr_raw, tr_final[TARGET])
X_tr_enc  = apply_encoder(X_tr_raw, enc_final)
cat_final = [c for c in LOW_CARD_CATS if c in X_tr_enc.columns]

# Compose final parameters: best Optuna params + fixed params
final_params = {**LGBM_FIXED, **best_params}

# Lower the learning rate slightly for the final model and scale up
# n_estimators proportionally (common practice: lr/2 -> n_est*2).
# This uses a softer shrinkage to squeeze a bit more from the full data.
FINAL_LR   = min(best_params.get("learning_rate", 0.05), 0.03)
LR_SCALE   = best_params.get("learning_rate", 0.05) / FINAL_LR
final_params["learning_rate"] = FINAL_LR
final_params["n_estimators"]  = int(2000 * LR_SCALE)
print(f"  Final LR={FINAL_LR} | n_estimators={final_params['n_estimators']}")

# Chronological 80/20 early-stopping split on full day48 training
es_cut = int(0.80 * len(X_tr_enc))
t0 = time.time()
final_model = lgb.LGBMRegressor(**final_params)
final_model.fit(
    X_tr_enc.iloc[:es_cut], y_final[:es_cut],
    eval_set=[(X_tr_enc.iloc[es_cut:], y_final[es_cut:])],
    categorical_feature=cat_final,
    callbacks=[
        lgb.early_stopping(EARLY_STOP, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)
feat_cols = list(X_tr_enc.columns)
print(f"  Final model trained in {time.time()-t0:.1f}s | "
      f"best_iter={final_model.best_iteration_}")


# =============================================================================
# SECTION 7: ITERATIVE FORECASTING ON DAY49 (FINAL HONEST VALIDATION)
# =============================================================================
# Use Model C (iterative slot-by-slot forecasting) so that lag features
# for slot t use the model's own predictions from slot t-1, not flat proxies.
# This is the same evaluation used in pipeline_v33.py's Model C.

print("\n" + "=" * 60)
print("SECTION 7: Iterative forecasting on day49 (honest validation)")
print("=" * 60)

y_val_true = d49[TARGET].values
t0_val = time.time()

val_preds = build_iterative_proxies(
    test_df       = d49.copy(),
    train_portion = d48,
    model         = final_model,
    enc           = enc_final,
    feature_cols  = feat_cols,
    cat_cols      = cat_final,
    max_demand    = max_demand,
    verbose       = True,
)
val_preds = clip_preds(val_preds, max_demand)
val_r2    = r2_score(y_val_true, val_preds)
val_score = max(0.0, 100.0 * val_r2)

print(f"\n  Val inference time : {time.time()-t0_val:.1f}s")
print(f"  Tuned day-split R2 : {val_r2:.4f}")
print(f"  Tuned day-split Score : {val_score:.2f}")
print(f"  Baseline (v33 Model C): {BASELINE_SCORE:.2f}")
print(f"  Delta vs baseline     : {val_score - BASELINE_SCORE:+.2f}")


# =============================================================================
# SECTION 8: TEST SET SUBMISSION
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 8: Generating test-set submission")
print("=" * 60)

# For test submission, retrain on all training data (day48 + day49 in-train)
print("  Retraining final model on full train (day48 + day49-train) ...")
tr_all_lags = build_train_lags(train_sorted)
n_b2 = len(tr_all_lags)
tr_all_lags = tr_all_lags.dropna(subset=["lag1"]).reset_index(drop=True)
print(f"  Rows dropped (lag1 NaN): {n_b2 - len(tr_all_lags)}")

for col in LAG_COLS:
    if tr_all_lags[col].isna().any():
        tr_all_lags[col].fillna(tr_all_lags[col].median(), inplace=True)

y_all = tr_all_lags[TARGET].values
X_tr_all  = finalize(tr_all_lags.drop(columns=[TARGET]))
enc_all   = fit_encoder(X_tr_all, tr_all_lags[TARGET])
X_tr_all  = apply_encoder(X_tr_all, enc_all)
cat_all   = [c for c in LOW_CARD_CATS if c in X_tr_all.columns]

# Use best Optuna params for the all-data model too
all_data_params = dict(final_params)
es_cut2 = int(0.80 * len(X_tr_all))
t0_all = time.time()
model_all = lgb.LGBMRegressor(**all_data_params)
model_all.fit(
    X_tr_all.iloc[:es_cut2], y_all[:es_cut2],
    eval_set=[(X_tr_all.iloc[es_cut2:], y_all[es_cut2:])],
    categorical_feature=cat_all,
    callbacks=[
        lgb.early_stopping(EARLY_STOP, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)
feat_cols_all = list(X_tr_all.columns)
print(f"  All-data model trained in {time.time()-t0_all:.1f}s | "
      f"best_iter={model_all.best_iteration_}")

# Iterative prediction on test set
print("\n  Iterative prediction on test set ...")
t0_te = time.time()
test_preds = build_iterative_proxies(
    test_df       = test_clean.copy(),
    train_portion = train_sorted,
    model         = model_all,
    enc           = enc_all,
    feature_cols  = feat_cols_all,
    cat_cols      = cat_all,
    max_demand    = max_demand,
    verbose       = True,
)
test_preds = clip_preds(test_preds, max_demand)
print(f"  Test inference time: {time.time()-t0_te:.1f}s")

submission = pd.DataFrame({
    "Index":  test_index.values,
    "demand": test_preds,
})
assert len(submission) == 41_778, f"Row count mismatch: {len(submission)}"
submission.to_csv("submission_tuned.csv", index=False)
print(f"\n  [OK] submission_tuned.csv saved | shape={submission.shape}")
print(f"  demand -> min={test_preds.min():.5f}  "
      f"max={test_preds.max():.5f}  mean={test_preds.mean():.5f}")


# =============================================================================
# SECTION 9: EXPERIMENT LOG
# =============================================================================
log = {
    "baseline_model_c_score":  BASELINE_SCORE,
    "optuna_best_cv_r2":       round(best_cv_r2, 4),
    "optuna_best_cv_score":    round(best_cv_score, 2),
    "tuned_day_split_r2":      round(val_r2, 4),
    "tuned_day_split_score":   round(val_score, 2),
    "delta_vs_baseline":       round(val_score - BASELINE_SCORE, 2),
    "optuna_trials_total":     len(study.trials),
    "optuna_trials_completed": len(completed),
    "optuna_trials_pruned":    len(pruned),
    "optuna_time_seconds":     round(t_optuna_elapsed, 1),
    "best_trial_number":       best_trial.number,
    "best_hyperparams":        best_params,
    "final_lr":                FINAL_LR,
    "final_n_estimators":      final_params["n_estimators"],
    "final_best_iter_d48":     final_model.best_iteration_,
    "final_best_iter_all":     model_all.best_iteration_,
}

with open("experiment_log_optuna.txt", "w") as f:
    f.write("Traffic Demand Prediction - Optuna Tuning Log\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Baseline (v33 Model C)     : {BASELINE_SCORE:.2f}\n")
    f.write(f"Optuna best CV Score       : {best_cv_score:.2f}\n")
    f.write(f"Tuned day-split Score      : {val_score:.2f}\n")
    f.write(f"Delta vs baseline          : {val_score - BASELINE_SCORE:+.2f}\n\n")
    f.write(f"Trials total / completed / pruned: "
            f"{len(study.trials)} / {len(completed)} / {len(pruned)}\n")
    f.write(f"Optuna time: {t_optuna_elapsed:.1f}s\n\n")
    f.write("Best hyperparameters:\n")
    for k, v in best_params.items():
        f.write(f"  {k:25s} = {v}\n")
    f.write(f"\nFull JSON:\n{json.dumps(log, indent=2)}\n")

print("\n  [OK] experiment_log_optuna.txt saved")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Baseline Model C score  : {BASELINE_SCORE:.2f}")
print(f"  Optuna CV best score    : {best_cv_score:.2f}")
print(f"  Tuned day-split score   : {val_score:.2f}  "
      f"({val_score - BASELINE_SCORE:+.2f} vs baseline)")
print(f"  Submission              : submission_tuned.csv")
print(f"  Log                     : experiment_log_optuna.txt")
print("\n+----------------------------------------------------------+")
print("|  SUBMISSION ORDER                                        |")
print("+----------------------------------------------------------+")
print("|  1. submission_tuned.csv        <- this run              |")
print("|  2. submission_iterative.csv    <- previous best         |")
print("+----------------------------------------------------------+")
