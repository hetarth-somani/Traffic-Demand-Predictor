"""
Evaluation routines including the critical iterative forecasting logic.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
from src import config
from src import features
from src.utils import clip_preds

def build_bagged_iterative_predictions(
    test_df: pd.DataFrame,
    train_portion: pd.DataFrame,
    models: List[Tuple],
    max_demand: float
) -> np.ndarray:
    """
    Iterative one-step-ahead prediction using an ENSEMBLE of models.
    At each time slot:
        - build lag features from buffer
        - predict with every model
        - average predictions -> final demand for that slot
        - use the averaged predictions to update the buffer for next slot
    """
    n_models = len(models)
    test_sorted = test_df.sort_values("_ts").reset_index(drop=True)

    max_lag = max(config.LAG_STEPS)
    tr_sorted = train_portion.sort_values([config.GEOHASH_COL, "_ts"])
    buffers = {}
    global_med = float(train_portion[config.TARGET].median())

    # Initialize buffers from training tail
    for gh, grp in tr_sorted.groupby(config.GEOHASH_COL, sort=False):
        demands = list(grp[config.TARGET].values)
        buffers[gh] = demands[-max_lag:] if len(demands) >= max_lag else demands

    unique_test_slots = sorted(test_sorted["_ts"].unique())
    preds_sorted = np.zeros(len(test_sorted))

    for slot_ts in unique_test_slots:
        slot_mask = test_sorted["_ts"] == slot_ts
        slot_rows = test_sorted[slot_mask].copy()

        # 1) Build lag features from buffers
        for col in config.LAG_COLS:
            slot_rows[col] = np.nan

        for idx, row in slot_rows.iterrows():
            gh = row[config.GEOHASH_COL]
            buf = buffers.get(gh, None)

            if buf is None or len(buf) == 0:
                for k in config.LAG_STEPS:
                    slot_rows.at[idx, f"lag{k}"] = global_med
                slot_rows.at[idx, f"rolling_mean_{config.ROLLING_WINDOW}"] = global_med
                slot_rows.at[idx, f"rolling_std_{config.ROLLING_WINDOW}"]  = 0.0
                slot_rows.at[idx, "expanding_mean"] = global_med
                continue

            n = len(buf)
            for k in config.LAG_STEPS:
                slot_rows.at[idx, f"lag{k}"] = buf[-k] if n >= k else buf[0]

            tail_r = buf[-config.ROLLING_WINDOW:] if n >= config.ROLLING_WINDOW else buf
            slot_rows.at[idx, f"rolling_mean_{config.ROLLING_WINDOW}"] = float(np.mean(tail_r))
            slot_rows.at[idx, f"rolling_std_{config.ROLLING_WINDOW}"]  = (
                float(np.std(tail_r)) if len(tail_r) > 1 else 0.0
            )
            slot_rows.at[idx, "expanding_mean"] = float(np.mean(buf))

        # 2) Finalize features
        slot_feat = features.finalize(slot_rows.drop(columns=[config.TARGET], errors="ignore"))

        # 3) Predict with each model and average
        all_preds = np.zeros((len(slot_rows), n_models))
        for m_idx, (model, enc, feat_cols, cat_cols, best_iter) in enumerate(models):
            slot_enc = features.apply_encoder(slot_feat, enc)
            
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
        slot_preds = clip_preds(slot_preds, max_demand)

        # Store
        slot_indices = test_sorted.index[slot_mask].tolist()
        preds_sorted[slot_indices] = slot_preds

        # 4) Update buffers with averaged predictions
        for (idx2, row2), pred in zip(slot_rows.iterrows(), slot_preds):
            gh = row2[config.GEOHASH_COL]
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
