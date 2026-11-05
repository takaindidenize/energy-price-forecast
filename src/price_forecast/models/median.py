"""The hourly median of three forecasters: LEAR on all history, LEAR on 180 days, and LightGBM.

Each member fails at different times. LEAR on all history is the best single model overall, but
slow to follow a new price level; the 180-day LEAR follows it but is unstable, with results that
depend on the day it is refitted; LightGBM is better at unusual hours and weaker on ordinary ones.
The median discards the outlier of the three every hour. It was the only combination better than
LEAR in both markets and both years tested (`analysis/ensemble.py`).

Clipping to the price limits gives the same result before or after taking the median, so
`median_bounded` is also the median of the three bounded forecasts.
"""

from __future__ import annotations

import pandas as pd

from price_forecast.models.lear import LEARForecaster
from price_forecast.models.lightgbm import LightGBMForecaster

SHORT_WINDOW_DAYS = 180


class MedianForecaster:
    name = "median"

    def __init__(self) -> None:
        self.members = [
            LEARForecaster(),
            LEARForecaster(calibration_days=SHORT_WINDOW_DAYS),
            LightGBMForecaster(),
        ]

    def fit(self, train: pd.DataFrame) -> MedianForecaster:
        for member in self.members:
            member.fit(train)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        predictions = pd.concat([m.predict(features) for m in self.members], axis=1)
        return predictions.median(axis=1).rename(self.name)
