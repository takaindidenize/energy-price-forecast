"""The interface shared by the modelling, backtesting and serving code.

A forecaster is trained on the modelling table (see "Modelling table" in CLAUDE.md) up to an issue
time and predicts the 24 hourly prices of the next delivery day. It may only use rows whose
information was available at the issue time; `features.py` is responsible for that, the model
just consumes the feature frame it is given.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Forecaster(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> Forecaster:
        """Fit on feature rows with a known `price_eur` target. Returns self."""
        ...

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """One prediction (EUR/MWh) per row of `features`, aligned to its index."""
        ...
