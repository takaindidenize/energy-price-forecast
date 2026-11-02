"""Forecasting models. Every model implements `Forecaster` from `base.py`.

`make_model(name)` builds one by name (`MODEL_NAMES`), so callers need not import each module.
Appending `_bounded` to a name (`"lear_bounded"`) wraps the model so its forecasts stay within
the market's regulated price limits (`models/bounded.py`).
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
BOUNDED_SUFFIX = "_bounded"


def make_model(name: str, **kwargs: Any) -> Forecaster:
    if name.endswith(BOUNDED_SUFFIX):
        from price_forecast.models.bounded import BoundedForecaster

        return BoundedForecaster(make_model(name.removesuffix(BOUNDED_SUFFIX), **kwargs))
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown model {name!r}; choose from {MODEL_NAMES}, optionally with '{BOUNDED_SUFFIX}'"
        )
    module = importlib.import_module(f"{__name__}.{name}")
    return getattr(module, _REGISTRY[name])(**kwargs)
