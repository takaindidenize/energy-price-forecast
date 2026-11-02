"""Forecasting models. Every model implements `Forecaster` from `base.py`.

`make_model(name)` builds one by name (`MODEL_NAMES`), so callers need not import each module.
"""

from __future__ import annotations

import importlib
from typing import Any

from price_forecast.models.base import Forecaster

# model name (= module name in this package) -> class name
_REGISTRY = {
    "naive": "NaiveForecaster",
    "seasonal_naive": "SeasonalNaiveForecaster",
    "lear": "LEARForecaster",
    "lightgbm": "LightGBMForecaster",
}
MODEL_NAMES = tuple(_REGISTRY)


def make_model(name: str, **kwargs: Any) -> Forecaster:
    if name not in _REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    module = importlib.import_module(f"{__name__}.{name}")
    return getattr(module, _REGISTRY[name])(**kwargs)
