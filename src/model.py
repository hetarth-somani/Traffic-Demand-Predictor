"""
Model factory functions.
"""
import lightgbm as lgb
from src import config

def create_baseline_model():
    """Create a single LightGBM regressor with baseline params."""
    return lgb.LGBMRegressor(**config.LGBM_BASELINE_PARAMS)

def create_bagged_base_model(seed: int):
    """Create a LightGBM regressor for the bagged ensemble with a specific seed."""
    params = config.LGBM_BAGGED_PARAMS.copy()
    params["random_state"] = seed
    return lgb.LGBMRegressor(**params)

def create_optuna_base_model(params_override: dict):
    """Create a LightGBM regressor using Optuna-suggested params."""
    params = config.LGBM_FIXED.copy()
    params.update(params_override)
    return lgb.LGBMRegressor(**params)
