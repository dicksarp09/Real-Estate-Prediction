from __future__ import annotations
from pathlib import Path
import json
from src.models.train import train_and_save

def run_training(
    raw_csv: str = "src/data/raw/real_estate.csv",
    model_path: str = "models/price_pipeline.joblib",
    metrics_path: str = "models/metrics.json",
):
    rmse, r2, saved = train_and_save(
        raw_csv=raw_csv,
        model_path=model_path
    )

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"rmse": rmse, "r2": r2, "model_path": saved}, f, indent=2)

    print(f"[TRAINING DONE] RMSE={rmse:.4f} R2={r2:.4f}")
    print(f"Model: {saved}\nMetrics: {metrics_path}")


if __name__ == "__main__":
    run_training()
