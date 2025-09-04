import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic real estate dataset.
    """

    rng = np.random.default_rng(random_state)

    # Features
    transaction_date = 2012 + rng.random(n_samples) * 3   # between 2012 and 2015
    house_age = rng.uniform(0, 40, n_samples)             # age of house
    distance_to_mrt = rng.exponential(1000, n_samples)    # skewed like real distances
    convenience_stores = rng.integers(0, 10, n_samples)   # integers between 0-9
    latitude = rng.uniform(24.9, 25.1, n_samples)         # narrow band around Taipei
    longitude = rng.uniform(121.4, 121.6, n_samples)

    # Target: house price per unit area
    price = (
        50
        - 0.2 * house_age
        - 0.01 * distance_to_mrt
        + 2 * convenience_stores
        + 10 * (latitude - 25)
        + 15 * (longitude - 121.5)
        + rng.normal(0, 3, n_samples)  # noise
    )

    df = pd.DataFrame({
        "transaction_date": transaction_date,
        "house_age": house_age,
        "distance_to_mrt": distance_to_mrt,
        "convenience_stores": convenience_stores,
        "latitude": latitude,
        "longitude": longitude,
        "price_per_unit_area": price,
    })

    return df


def save_datasets(df: pd.DataFrame, output_dir: str = "data/processed", test_size: float = 0.2):
    """
    Split into train/test and save CSVs.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"✅ Data saved to {output_dir}: {len(train_df)} train, {len(test_df)} test")


if __name__ == "__main__":
    df = generate_synthetic_data()
    save_datasets(df)
