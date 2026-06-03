"""
Diagnose why folds 4-5 collapse.
Tests: demand cyclicity, proxy staleness, day-split performance.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import category_encoders as ce

train_raw = pd.read_csv("dataset/train.csv")
test_raw  = pd.read_csv("dataset/test.csv")

# Reconstruct datetime
base = pd.Timestamp("2000-01-01")
def make_ts(df):
    ts = base + pd.to_timedelta(df["day"].astype(int), unit="D")
    p  = df["timestamp"].str.split(":", expand=True).astype(int)
    return ts + pd.to_timedelta(p[0], unit="h") + pd.to_timedelta(p[1], unit="m")

train_raw["_ts"] = make_ts(train_raw)
test_raw["_ts"]  = make_ts(test_raw)

# --- 1. What time slots are in test? ---
print("=== TEST SET TIME STRUCTURE ===")
print(f"Test rows: {len(test_raw)}")
print(f"Test day values: {sorted(test_raw['day'].unique())}")
test_slots = (test_raw["_ts"].dt.hour * 4 + test_raw["_ts"].dt.minute // 15)
print(f"Test slots (0-95): {sorted(test_slots.unique())[:20]}...")
print(f"Test _ts range: {test_raw['_ts'].min()} -> {test_raw['_ts'].max()}")
print()

# --- 2. Demand by time slot (hourly average) ---
train_raw["slot"] = (train_raw["_ts"].dt.hour * 4 +
                     train_raw["_ts"].dt.minute // 15)
slot_demand = train_raw.groupby("slot")["demand"].mean()
print("=== DEMAND BY HOUR (avg across all geohashes) ===")
for hour in range(24):
    slots = range(hour*4, hour*4+4)
    avg = slot_demand.reindex(slots).mean()
    bar = "#" * int(avg * 200)
    print(f"  {hour:02d}:00  {avg:.4f}  {bar}")
print()

# --- 3. Slot coverage in train ---
print("=== SLOT COVERAGE IN TRAIN ===")
slot_counts = train_raw.groupby("slot").size()
print(f"Slots with data: {len(slot_counts)} of 96")
print(f"Rows per slot: min={slot_counts.min()}  max={slot_counts.max()}  "
      f"mean={slot_counts.mean():.0f}")
print()

# --- 4. Day split: train day48, validate day49 ---
print("=== NATURAL DAY SPLIT: train=day48, validate=day49 ===")
d48 = train_raw[train_raw["day"] == 48].copy()
d49 = train_raw[train_raw["day"] == 49].copy()
print(f"Day 48 rows: {len(d48)}  |  Day 49 rows: {len(d49)}")
print(f"Day 49 slots: {sorted(d49['slot'].unique())}")
print(f"Day 49 time range: {d49['_ts'].min()} -> {d49['_ts'].max()}")
print()

# --- 5. How stale are the lag proxies across folds? ---
print("=== PROXY STALENESS ANALYSIS ===")
train_s = train_raw.sort_values("_ts").reset_index(drop=True)
tscv = TimeSeriesSplit(n_splits=5)
for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_s), start=1):
    tr = train_s.iloc[tr_idx]
    vl = train_s.iloc[val_idx]
    tr_end = tr["_ts"].max()
    vl_start = vl["_ts"].min()
    vl_end   = vl["_ts"].max()
    gap_slots = int((vl_end - tr_end).total_seconds() / (15*60))
    print(f"  Fold {fold}: train ends {tr_end.strftime('%H:%M')}  "
          f"val {vl_start.strftime('%H:%M')}-{vl_end.strftime('%H:%M')}  "
          f"max_lag_staleness={gap_slots} slots ({gap_slots*15}min)")
