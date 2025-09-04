import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.data.preprocess import load_or_split
from src.data.preprocess import build_preprocessor  # moved here


def build_pipeline():
    """
    Build the full modeling pipeline.
    Uses StandardScaler + LinearRegression.
    """
    pre = build_preprocessor()  # ✅ no arguments needed

    steps = [
        ("preprocessor", pre),
        ("model", LinearRegression())
    ]

    pipe = Pipeline(steps=steps)
    return pipe


def train_and_save(raw_csv="src/data/raw/real_estate.csv", model_path="models/price_pipeline.joblib"):
    """
    Train a model on the dataset and save it.
    Returns RMSE, R2, and saved path.
    """
    # Load train/test data
    X_train, X_test, y_train, y_test = load_or_split(raw_csv=raw_csv)

    # Build pipeline
    pipe = build_pipeline()

    # Train
    pipe.fit(X_train, y_train)

    # Predict
    preds = pipe.predict(X_test)

    # Evaluate
    rmse = mean_squared_error(y_test, preds) ** 0.5

    r2 = r2_score(y_test, preds)

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)

    return rmse, r2, model_path
