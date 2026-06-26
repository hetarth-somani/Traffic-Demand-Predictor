import sys
import os
import optuna

# Add the root directory to sys.path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.data import load_data
from src.features import clean, build_train_lags, build_val_proxies, finalize, fit_encoder, apply_encoder
from src.model import create_optuna_base_model
import pandas as pd
from sklearn.metrics import r2_score

def make_cv_folds(unique_slots: list, n_folds: int, val_window: int) -> list:
    n = len(unique_slots)
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

def run_single_fold(d48, tr_slots: list, val_slots: list, params: dict) -> float:
    tr_fold  = d48[d48["_ts"].isin(tr_slots)].copy()
    val_fold = d48[d48["_ts"].isin(val_slots)].copy()
    y_val    = val_fold[config.TARGET].values

    if len(tr_fold) < 100 or len(val_fold) < 10:
        return 0.0

    tr_lags = build_train_lags(tr_fold)
    tr_lags = tr_lags.dropna(subset=["lag1"]).reset_index(drop=True)
    for col in config.LAG_COLS:
        if tr_lags[col].isna().any():
            tr_lags[col].fillna(tr_lags[col].median(), inplace=True)

    if len(tr_lags) < 50:
        return 0.0

    val_lags = build_val_proxies(train_portion=tr_fold, val_portion=val_fold)

    X_tr_raw  = finalize(tr_lags.drop(columns=[config.TARGET]))
    X_val_raw = finalize(val_lags.drop(columns=[config.TARGET]))

    enc   = fit_encoder(X_tr_raw, tr_lags[config.TARGET])
    X_tr  = apply_encoder(X_tr_raw, enc)
    X_val = apply_encoder(X_val_raw, enc)

    cat_cols = [c for c in config.LOW_CARD_CATS if c in X_tr.columns]

    es_cut = int(0.80 * len(X_tr))
    y_tr = tr_lags[config.TARGET]

    model = create_optuna_base_model(params)
    model.fit(
        X_tr.iloc[:es_cut], y_tr.iloc[:es_cut],
        eval_set=[(X_tr.iloc[es_cut:], y_tr.iloc[es_cut:])],
        categorical_feature=cat_cols,
        callbacks=[optuna.integration.LightGBMPruningCallback(trial=None, valid_name="valid_0")] if False else [], 
        # Pruning callback is omitted here for simplicity
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration_)
    return r2_score(y_val, preds)

def objective(trial, d48, cv_folds):
    params = {
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    
    scores = []
    for i, (tr_s, val_s) in enumerate(cv_folds):
        r2 = run_single_fold(d48, tr_s, val_s, params)
        scores.append(max(0.0, r2))
    
    return sum(scores) / len(scores) if scores else 0.0

def main():
    print("=== OPTUNA HYPERPARAMETER TUNING ===")
    
    train_raw, _ = load_data()
    train_clean = clean(train_raw)
    train_sorted = train_clean.sort_values("_ts").reset_index(drop=True)
    d48 = train_sorted[train_sorted["day"] == 48].copy().reset_index(drop=True)
    
    unique_slots_d48 = sorted(d48["_ts"].unique())
    cv_folds = make_cv_folds(unique_slots_d48, config.N_CV_FOLDS, config.VAL_WINDOW_SLOTS)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, d48, cv_folds), n_trials=config.OPTUNA_N_TRIALS)

    print(f"\nBest Trial: {study.best_trial.value}")
    print("Best Params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
