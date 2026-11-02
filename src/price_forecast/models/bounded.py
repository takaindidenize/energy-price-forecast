"""Any forecaster, with its forecasts kept within the market's regulated limits (`bounds.py`)."""

from __future__ import annotations

import pandas as pd

from price_forecast import bounds
from price_forecast.models.base import Forecaster


class BoundedForecaster:
    """Wraps a forecaster; `make_model("lear_bounded")` builds LEAR wrapped in this."""

    def __init__(self, inner: Forecaster) -> None:
        self.inner = inner
        self.name = f"{inner.name}_bounded"
        self._rates: dict[str, pd.Series | None] = {}

    def fit(self, train: pd.DataFrame) -> BoundedForecaster:
        self.inner.fit(train)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        predictions = self.inner.predict(features)
        if features.empty:
            return predictions
        market = str(features["market"].iloc[0])
        if market not in self._rates:
            # Rates for every day on disk; `bounds.clip` only reads the issue day's.
            self._rates[market] = bounds.eur_per_local(market)
        return bounds.clip(
            market, features["delivery_date_local"], predictions, self._rates[market]
        )
