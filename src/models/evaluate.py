# src/models/evaluate.py
from __future__ import annotations
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from src.data.preprocess import load_or_split, CANONICAL_FEATURES, TARGET_COL

def evaluate_on_csv(model_path: str, csv_path: str):
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]

    df = load_dataframe(csv_path)
    X = df[CANONICAL_FEATURES]
    y = df[TARGET_COL]

    y_pred = pipe.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2 = float(r2_score(y, y_pred))
    return rmse, r2

if __name__ == "__main__":
    print(evaluate_on_csv("models/price_pipeline.joblib", "data/processed/test.csv"))
