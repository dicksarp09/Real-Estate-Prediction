# src/features/build_features.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DistanceToCenter(BaseEstimator, TransformerMixin):
    """
    Adds 'distance_to_center' (and optional 'distance_to_center_sq') using Euclidean
    distance in degrees between (lat, lon) and a reference point. Leaves other columns intact.
    """
    def __init__(
        self,
        lat_col: str = "X5_latitude",
        lon_col: str = "X6_longitude",
        ref_lat: float = 25.0330,
        ref_lon: float = 121.5654,
        add_squared: bool = False,
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.add_squared = add_squared

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, (pd.DataFrame,)):
            raise ValueError("DistanceToCenter expects a pandas DataFrame.")
        X = X.copy()
        if self.lat_col not in X or self.lon_col not in X:
            raise KeyError(f"Expected '{self.lat_col}' and '{self.lon_col}' in columns.")
        dist = np.sqrt((X[self.lat_col] - self.ref_lat) ** 2 +
                       (X[self.lon_col] - self.ref_lon) ** 2)
        X["distance_to_center"] = dist
        if self.add_squared:
            X["distance_to_center_sq"] = dist ** 2
        return X

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, (pd.DataFrame,)):
            raise ValueError("DropColumns expects a pandas DataFrame.")
        X = X.copy()
        existing = [c for c in self.cols if c in X.columns]
        return X.drop(columns=existing)
