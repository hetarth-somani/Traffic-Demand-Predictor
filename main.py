import sys
import os
import pandas as pd
import numpy as np

from src import config
from src.data import load_data
from src.features import clean, build_train_lags, finalize, fit_encoder, apply_encoder
from src.train import train_bagged_models
from src.evaluate import build_bagged_iterative_predictions
from src.utils import Timer

def main():
    print("============================================================")
    print(" FLIPKART GRIDLOCK: BAGGED ITERATIVE PIPELINE ")
    print("============================================================\n")

    with Timer("Data Loading"):
        try:
            train_raw, test_raw = load_data()
            test_index = test_raw["Index"].copy()
        except FileNotFoundError:
            print(f"Data files not found. Please ensure data is present at {config.TRAIN_PATH} and {config.TEST_PATH}")
            return
            
    with Timer("Preprocessing"):
        train_clean = clean(train_raw)
        test_clean  = clean(test_raw)
        train_sorted = train_clean.sort_values("_ts").reset_index(drop=True)
        max_demand = float(train_raw[config.TARGET].max())
        
    with Timer("Train Lags & Target Encoding"):
        tr_all_lags = build_train_lags(train_sorted)
        tr_all_lags = tr_all_lags.dropna(subset=["lag1"]).reset_index(drop=True)
        for col in config.LAG_COLS:
            if tr_all_lags[col].isna().any():
                tr_all_lags[col].fillna(tr_all_lags[col].median(), inplace=True)
        
        X_all_raw = finalize(tr_all_lags.drop(columns=[config.TARGET]))
        enc_all = fit_encoder(X_all_raw, tr_all_lags[config.TARGET])
        X_all = apply_encoder(X_all_raw, enc_all)
        y_all = tr_all_lags[config.TARGET].values

    with Timer(f"Training Bagged Models ({len(config.BAGGING_SEEDS)} models)"):
        models_final = train_bagged_models(X_all, y_all, enc_all)
        
    with Timer("Iterative Forecasting on Test Set"):
        test_preds = build_bagged_iterative_predictions(
            test_df=test_clean,
            train_portion=train_sorted,
            models=models_final,
            max_demand=max_demand
        )
        
    with Timer("Submission Generation"):
        submission = pd.DataFrame({"Index": test_index.values, "demand": test_preds})
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"\nSaved submission to {config.SUBMISSION_PATH}")
        print(f"Demand distribution: min={submission['demand'].min():.4f}, max={submission['demand'].max():.4f}, mean={submission['demand'].mean():.4f}")

if __name__ == "__main__":
    main()
