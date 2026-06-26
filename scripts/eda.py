import sys
import os
import matplotlib.pyplot as plt

# Add the root directory to sys.path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.data import load_data

def main():
    print("=== EXPLORATORY DATA ANALYSIS ===")
    
    # Ensure data directory exists or at least handled gracefully
    try:
        train, test = load_data()
    except FileNotFoundError:
        print(f"Data files not found in {config.TRAIN_PATH}. Please ensure data is present.")
        return

    print("--- 1. Test Set Time Structure ---")
    print(f"Test rows: {len(test)}")
    print(f"Test day values: {sorted(test['day'].unique())}")
    test_slots = (test["_ts"].dt.hour * 4 + test["_ts"].dt.minute // 15)
    print(f"Test slots (0-95): {sorted(test_slots.unique())[:20]}...")
    print(f"Test _ts range: {test['_ts'].min()} -> {test['_ts'].max()}\n")

    print("--- 2. Demand By Hour ---")
    train["slot"] = (train["_ts"].dt.hour * 4 + train["_ts"].dt.minute // 15)
    slot_demand = train.groupby("slot")["demand"].mean()
    for hour in range(24):
        slots = range(hour*4, hour*4+4)
        avg = slot_demand.reindex(slots).mean()
        bar = "#" * int(avg * 200)
        print(f"  {hour:02d}:00  {avg:.4f}  {bar}")
    print()

    print("--- 3. Slot Coverage In Train ---")
    slot_counts = train.groupby("slot").size()
    print(f"Slots with data: {len(slot_counts)} of 96")
    print(f"Rows per slot: min={slot_counts.min()}  max={slot_counts.max()}  mean={slot_counts.mean():.0f}\n")

    print("--- 4. Day Split ---")
    d48 = train[train["day"] == 48].copy()
    d49 = train[train["day"] == 49].copy()
    print(f"Day 48 rows: {len(d48)}  |  Day 49 rows: {len(d49)}")
    if len(d49) > 0:
        print(f"Day 49 slots: {sorted(d49['slot'].unique())}")
        print(f"Day 49 time range: {d49['_ts'].min()} -> {d49['_ts'].max()}\n")

    print("--- EDA Script Finished ---")

if __name__ == "__main__":
    main()
