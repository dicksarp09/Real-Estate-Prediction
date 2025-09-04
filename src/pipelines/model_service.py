# src/pipelines/model_service.py
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join("models", "price_pipeline.joblib")
_model = joblib.load(MODEL_PATH)
print(f"[INFO] Model loaded from {MODEL_PATH}")

def predict_from_dict(data: dict):
    df = pd.DataFrame([data])
    prediction = _model.predict(df)
    return prediction.tolist()
