import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Map raw dataset column names → canonical short names
COLUMN_RENAME_MAP = {
    "X1_transaction_date": "transaction_date",
    "X2_house_age": "house_age",
    "X3_distance_to_MRT_station": "distance_to_mrt",
    "X4_number_of_convenience_stores": "convenience_stores",
    "X5_latitude": "latitude",
    "X6_longitude": "longitude",
    "Y_house_price_of_unit_area": "price_per_unit_area"
}

# Canonical features and target
CANONICAL_FEATURES = [
    "transaction_date",
    "house_age",
    "distance_to_mrt",
    "convenience_stores",
    "latitude",
    "longitude",
]

TARGET_COL = "price_per_unit_area"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename dataset columns to canonical names if possible."""
    return df.rename(columns=COLUMN_RENAME_MAP)


def load_or_split(raw_csv: str, train_csv="src/data/processed/train.csv", test_csv="src/data/processed/test.csv"):
    """
    Load train/test splits if they exist.
    Otherwise split from raw CSV and save them.
    Always normalizes column names.
    """
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)

        df_train = normalize_columns(df_train)
        df_test = normalize_columns(df_test)
    else:
        df = pd.read_csv(raw_csv)
        df = normalize_columns(df)

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

        os.makedirs(os.path.dirname(train_csv), exist_ok=True)
        df_train.to_csv(train_csv, index=False)
        df_test.to_csv(test_csv, index=False)

    X_train = df_train[CANONICAL_FEATURES]
    y_train = df_train[TARGET_COL]

    X_test = df_test[CANONICAL_FEATURES]
    y_test = df_test[TARGET_COL]

    return X_train, X_test, y_train, y_test


def build_preprocessor():
    """
    Build preprocessing pipeline for numeric features.
    Currently just standardizes features.
    """
    return Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
