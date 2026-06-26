# Flipkart Gridlock: Traffic Demand Prediction

## Overview
This repository contains a modular Python solution for the Flipkart Gridlock Hackathon 2.0 (Traffic Demand Prediction). The goal is to predict continuous traffic demand at 15-minute granularity across multiple geographic locations (geohash-encoded cells).

The core challenge is a time-series forecasting problem with high-cardinality location identifiers and an extremely short time window (~26 hours total).

**Best LB Score**: ~90.30

## Solution Architecture
The final solution relies on an **Iterative Bagged LightGBM Model**.
Instead of using constant lagged proxies that become stale over the test horizon, this model employs **iterative one-step-ahead forecasting**. For each 15-minute slot, the model predicts the demand, which is then dynamically fed back into the lag feature buffers for subsequent slots. 

To reduce variance and improve stability across the iterative process, predictions are averaged across an ensemble of 10 identical models trained with different random seeds (Bagging). 

Key feature engineering steps include:
- Reconstructing datetime from time-of-day strings and abstract days.
- Generating autoregressive lags (up to 1-day back) and rolling statistics (mean, std).
- Target encoding for geohash location identifiers (fitted out-of-fold/in-fold to prevent leakage).
- Extracting cyclic encodings of hour-of-day and slot-of-day.

## Repository Structure

```
flipkart-gridlock/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                   # Entrypoint to run the full training and prediction pipeline
├── src/                      # Core modular components
│   ├── __init__.py
│   ├── config.py             # Hyperparameters, paths, and constants
│   ├── data.py               # Data loading and datetime parsing
│   ├── features.py           # Feature engineering and lag building
│   ├── model.py              # Model factory functions
│   ├── train.py              # Training logic
│   ├── evaluate.py           # Iterative one-step-ahead forecasting and evaluation
│   └── utils.py              # Helper functions (timing, clipping, etc.)
├── scripts/                  # Standalone execution scripts
│   ├── eda.py                # Exploratory data analysis (slot coverage, demand curves)
│   └── tune_optuna.py        # Hyperparameter tuning with Optuna using an expanding window
└── notebooks/                # Original competition notebooks (reference only)
    ├── eda-flipkart-gridlock.ipynb
    ├── baseline-regressor.ipynb
    ├── bagged-iterative.ipynb
    └── optuna-tune.ipynb
```

## Setup Instructions
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset files (`train.csv` and `test.csv`) inside a `data/` directory at the root of the project.

## How to Run

### Exploratory Data Analysis (EDA)
Run the standalone EDA script to inspect data coverage, basic statistics, and demand distributions:
```bash
python scripts/eda.py
```

### Hyperparameter Tuning
Run the Optuna hyperparameter optimization script using an expanding cross-validation window over Day 48:
```bash
python scripts/tune_optuna.py
```

### Training & Generating Submissions
To execute the full final pipeline (loading data, feature engineering, training the 10 bagged LightGBM models, iterative forecasting on the test set, and saving the final CSV):
```bash
python main.py
```
This will output `submission.csv` in the root directory.
