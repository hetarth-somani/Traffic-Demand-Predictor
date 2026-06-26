"""
Configuration settings for the Flipkart Gridlock project.
Contains file paths, hyperparameters, and feature definitions.
"""
import pandas as pd

# File Paths
# Assuming the data directory is placed in the project root
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SUBMISSION_PATH = "submission.csv"

# Global Constants
RANDOM_STATE = 42
BASE_DATE = pd.Timestamp("2000-01-01")

# Feature Engineering Settings
LAG_STEPS = [1, 2, 4, 8, 12, 96]
ROLLING_WINDOW = 4  # 4 x 15min = 1 hour
LAG_COLS = (
    [f"lag{k}" for k in LAG_STEPS]
    + [f"rolling_mean_{ROLLING_WINDOW}", f"rolling_std_{ROLLING_WINDOW}", "expanding_mean"]
)

LOW_CARD_CATS = ["RoadType", "Weather", "LargeVehicles", "Landmarks"]
GEOHASH_COL = "geohash"
GH_ENC_COL = "geohash_te"
TARGET = "demand"

LOG_TRANSFORM = True  # Used in baseline, but iter forecasting might use original scale. 
                      # The bagged iterative used raw scale with np.clip.

# CV Settings
N_SPLITS = 5  # For TimeSeriesSplit (Baseline)
VAL_WINDOW_SLOTS = 16  # For Optuna custom expanding window
N_CV_FOLDS = 3         # For Optuna custom expanding window

# LightGBM Baseline Params
LGBM_BASELINE_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 10000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
    "n_jobs": -1,
}
EARLY_STOP_BASELINE = 50

# LightGBM Bagged Iterative Params (from notebook)
LGBM_BAGGED_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 10000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbosity": -1,
    "n_jobs": -1,
}
BAGGING_SEEDS = [42, 101, 777, 2023, 1111, 2222, 3333, 4444, 5555, 9999]
EARLY_STOP_BAGGED = 100

# Optuna Tuning Params
OPTUNA_N_TRIALS = 40
OPTUNA_N_STARTUP_TRIALS = 5
OPTUNA_N_WARMUP_STEPS = 1

LGBM_FIXED = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}
EARLY_STOP_OPTUNA = 50
