"""Naive benchmark: the price of the same local hour on the issue day (one day before delivery)."""

from __future__ import annotations

import pandas as pd


class NaiveForecaster:
    name = "naive"
    column = "price_d1_same_hour"

    def fit(self, train: pd.DataFrame) -> NaiveForecaster:
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return features[self.column].astype(float).rename(self.name)
