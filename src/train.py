"""
Training loops, cross-validation, and bagged model fitting.
"""
import pandas as pd
from typing import Tuple, List
import lightgbm as lgb

from src import config
from src import features
from src.model import create_bagged_base_model

def train_lgbm_single(
    model: lgb.LGBMRegressor, 
    X_tr: pd.DataFrame, 
    y_tr: pd.Series, 
    cat_cols: list, 
    early_stop: int = 50,
    val_fraction: float = 0.20
) -> Tuple[lgb.LGBMRegressor, int]:
    """
    Train a single LightGBM model. 
    Uses the last `val_fraction` of X_tr for early stopping.
    """
    es_cut = int((1.0 - val_fraction) * len(X_tr))
    
    model.fit(
        X_tr.iloc[:es_cut], y_tr.iloc[:es_cut] if isinstance(y_tr, pd.Series) else y_tr[:es_cut],
        eval_set=[(X_tr.iloc[es_cut:], y_tr.iloc[es_cut:] if isinstance(y_tr, pd.Series) else y_tr[es_cut:])],
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return model, model.best_iteration_

def train_bagged_models(X_all: pd.DataFrame, y_all: pd.Series, enc) -> List[Tuple]:
    """
    Train a full bagging ensemble on the provided dataset.
    Returns a list of tuples: (model, encoder, feature_cols, cat_cols, best_iter)
    """
    cat_all = [c for c in config.LOW_CARD_CATS if c in X_all.columns]
    feature_cols = list(X_all.columns)
    
    models = []
    for seed in config.BAGGING_SEEDS:
        model = create_bagged_base_model(seed)
        model, best_iter = train_lgbm_single(
            model, 
            X_all, 
            y_all, 
            cat_all, 
            early_stop=config.EARLY_STOP_BAGGED, 
            val_fraction=0.15
        )
        models.append((model, enc, feature_cols, cat_all, best_iter))
        
    return models
