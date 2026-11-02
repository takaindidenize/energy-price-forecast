"""Global gradient-boosted trees: one LightGBM model for all hours of the day.

Hour of day, day of week and holiday enter as features, alongside the price lags and the delivery
hour's weather forecast. Training on the asinh-transformed target (as LEAR does) keeps a squared
loss from chasing the rare spikes and negative hours; predictions are transformed back.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from price_forecast.features import FEATURE_COLUMNS, TARGET_COLUMN
from price_forecast.models._common import AsinhScaler, with_target


class LightGBMForecaster:
    name = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        seed: int = 0,
        n_jobs: int = 1,
    ) -> None:
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "random_state": seed,
            "n_jobs": n_jobs,
            "deterministic": True,
            "force_row_wise": True,
            "verbose": -1,
        }

    @staticmethod
    def _design(df: pd.DataFrame) -> pd.DataFrame:
        return df[list(FEATURE_COLUMNS)].astype(float)

    def fit(self, train: pd.DataFrame) -> LightGBMForecaster:
        train = with_target(train)
        self.scaler_ = AsinhScaler().fit(train[TARGET_COLUMN])
        self.model_ = lgb.LGBMRegressor(**self.params)
        self.model_.fit(self._design(train), self.scaler_.transform(train[TARGET_COLUMN]))
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        z = self.model_.predict(self._design(features))
        return pd.Series(self.scaler_.inverse_transform(z), index=features.index, name=self.name)
