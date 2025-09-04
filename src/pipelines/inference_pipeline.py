import pandas as pd
import joblib
import os

def run_inference():
    # Load model
    model_path = os.path.join("models", "price_pipeline.joblib")
    model = joblib.load(model_path)
    print(f"[INFO] Loaded model from {model_path}")

    # Load new data
    data_path = os.path.join("data", "inference", "new_data.csv")
    data = pd.read_csv(data_path)
    print(f"[INFO] Loaded input data with shape {data.shape}")

    # Drop columns not used for prediction
    drop_cols = ["No", "price"]
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])
    print(f"[INFO] Data after dropping unused columns: {data.shape}")

    # Run predictions
    predictions = model.predict(data)
    print("[INFO] Predictions generated")

    # Save predictions
    output_path = os.path.join("data", "processed", "predictions.csv")
    pd.DataFrame({"prediction": predictions}).to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved to {output_path}")

if __name__ == "__main__":
    run_inference()
