"""Helpers shared by the models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from price_forecast.features import TARGET_COLUMN

# Scales a median absolute deviation to the standard deviation for normally distributed data.
_MAD_TO_SD = 1.4826


class AsinhScaler:
    """Variance-stabilising transform for prices: asinh((p - median) / MAD).

    From Uniejewski, Weron and Ziel (2018) and used by LEAR in Lago et al. (2021). Close to linear
    near the median and logarithmic in the tails, and unlike a log it is defined for negative
    prices, so spikes and negative hours do not dominate a squared-error fit.
    """

    def fit(self, prices: pd.Series | np.ndarray) -> AsinhScaler:
        p = np.asarray(prices, dtype=float)
        p = p[np.isfinite(p)]
        if p.size == 0:
            raise ValueError("cannot fit AsinhScaler without any known price")
        self.median_ = float(np.median(p))
        mad = float(np.median(np.abs(p - self.median_))) * _MAD_TO_SD
        self.scale_ = mad if mad > 0 else (float(np.std(p)) or 1.0)
        return self

    def transform(self, prices: pd.Series | np.ndarray) -> np.ndarray:
        return np.arcsinh((np.asarray(prices, dtype=float) - self.median_) / self.scale_)

    def inverse_transform(self, z: pd.Series | np.ndarray) -> np.ndarray:
        return np.sinh(np.asarray(z, dtype=float)) * self.scale_ + self.median_


def with_target(train: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known target; raises if there are none."""
    known = train[train[TARGET_COLUMN].notna()]
    if known.empty:
        raise ValueError("training frame has no rows with a known price_eur")
    return known
