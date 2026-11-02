"""Weekly seasonal naive: the price of the same local hour one week before the delivery day.

This is the rMAE denominator, so every other model is judged against it.
"""

from __future__ import annotations

import pandas as pd


class SeasonalNaiveForecaster:
    name = "seasonal_naive"
    column = "price_d7_same_hour"

    def fit(self, train: pd.DataFrame) -> SeasonalNaiveForecaster:
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return features[self.column].astype(float).rename(self.name)
