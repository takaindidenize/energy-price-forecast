"""Rolling-origin backtest: one forecast per delivery day, as it would have been issued.

For every delivery day T in [test_start, test_end] (local dates, inclusive):

1. The origin is `features.issue_time(T)`: T-1 at the market's `issue_hour_local`.
2. Training rows are those whose price was known at the origin (`features.price_known_at`), i.e.
   delivery days up to and including T-1, optionally limited to a rolling window. The run asserts
   this for every origin.
3. Models are refitted every `refit_every` days (1 = daily, as in operation); in between, the
   last fit, made at an earlier origin, is reused, which can only use less information.
4. Each model predicts T's rows with the target column removed.

The modelling table is built once for the whole period; that is safe because every row's features
only use information from before that row's own issue time (tested in tests/test_features.py).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd

from price_forecast import metrics
from price_forecast.features import (
    TARGET_COLUMN,
    build_features,
    issue_time,
    load_hourly,
    price_known_at,
)
from price_forecast.models.base import Forecaster

log = logging.getLogger(__name__)

PREDICTION_COLUMNS = (
    "market",
    "model",
    "issued_at",
    "delivery_hour",
    "delivery_date_local",
    "hour_local",
    "price_eur",
    "price_eur_pred",
    "price_eur_snaive",  # weekly seasonal naive on the same row, the rMAE benchmark
)
SUMMARY_COLUMNS = ("market", "model", "n", "mae", "rmse", "rmae")


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def run(
    market: str,
    models: Sequence[Forecaster],
    test_start: str | date,
    test_end: str | date,
    *,
    features: pd.DataFrame | None = None,
    refit_every: int = 7,
    window_days: int | None = None,
    min_train_days: int = 28,
) -> pd.DataFrame:
    """Backtest `models` on delivery days `test_start`..`test_end` and return one row per model and
    delivery hour (columns `PREDICTION_COLUMNS`), ordered by issue time, delivery hour and model.

    `features`: the modelling table; built from the raw partitions when omitted.
    `refit_every`: refit every this many delivery days (1 = daily).
    `window_days`: train on the latest this many days only (rolling); None = expanding window.
    `min_train_days`: delivery days with fewer known training days are skipped.

    Use `summarize` on the result for per-model MAE, RMSE and rMAE.
    """
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ValueError(f"model names must be unique, got {names}")
    start, end = _as_date(test_start), _as_date(test_end)
    if end < start:
        raise ValueError(f"test_end {end} is before test_start {start}")
    if features is None:
        features = build_features(load_hourly(market), market)

    feats = features[features["market"] == market].sort_values("delivery_hour")
    day_of = pd.to_datetime(feats["delivery_date_local"])
    known = feats[feats[TARGET_COLUMN].notna()]
    known_day = day_of[known.index]

    last_fit: dict[str, date] = {}
    out = []
    n_days = (end - start).days + 1
    for target_day in (start + timedelta(days=i) for i in range(n_days)):
        rows = feats[day_of == pd.Timestamp(target_day)]
        if rows.empty:
            log.warning("%s: no feature rows for delivery day %s, skipped", market, target_day)
            continue
        issued_at = issue_time(target_day, market)
        last_known = pd.Timestamp(target_day - timedelta(days=1))
        in_train = known_day <= last_known
        if window_days is not None:
            in_train &= known_day > last_known - pd.Timedelta(days=window_days)
        train = known[in_train.to_numpy()]
        n_train_days = train["delivery_date_local"].nunique()
        if n_train_days < min_train_days:
            log.info(
                "%s %s: %d training days < %d, skipped",
                market,
                target_day,
                n_train_days,
                min_train_days,
            )
            continue

        # No look-ahead: every training price was known at the origin, and the forecast is for
        # hours after it.
        if price_known_at(train["delivery_date_local"].max(), market) > issued_at:
            raise RuntimeError(f"look-ahead: training prices for {target_day} not yet known")
        if rows["delivery_hour"].min() <= issued_at:
            raise RuntimeError(f"look-ahead: {target_day} starts before its issue time")

        inputs = rows.drop(columns=[TARGET_COLUMN])
        for model in models:
            due = model.name not in last_fit or (target_day - last_fit[model.name]).days >= (
                refit_every
            )
            if due:
                model.fit(train)
                last_fit[model.name] = target_day
            pred = model.predict(inputs)
            out.append(
                pd.DataFrame(
                    {
                        "market": market,
                        "model": model.name,
                        "issued_at": issued_at,
                        "delivery_hour": rows["delivery_hour"].to_numpy(),
                        "delivery_date_local": rows["delivery_date_local"].to_numpy(),
                        "hour_local": rows["hour_local"].to_numpy(),
                        "price_eur": rows[TARGET_COLUMN].to_numpy(),
                        "price_eur_pred": pred.reindex(rows.index).to_numpy(dtype=float),
                        "price_eur_snaive": rows["price_d7_same_hour"].to_numpy(),
                    }
                )
            )

    if not out:
        log.warning("%s: backtest produced no forecasts for %s..%s", market, start, end)
        return pd.DataFrame(columns=list(PREDICTION_COLUMNS))
    result = pd.concat(out, ignore_index=True)
    result["model"] = pd.Categorical(result["model"], categories=names)
    result = result.sort_values(["issued_at", "delivery_hour", "model"], kind="stable")
    result["model"] = result["model"].astype(str)
    return result[list(PREDICTION_COLUMNS)].reset_index(drop=True)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per market and model: n, MAE, RMSE and rMAE vs the weekly seasonal naive, on the rows where
    the price, the forecast and the benchmark are all known."""
    rows = []
    for (market, model), g in predictions.groupby(["market", "model"], sort=False):
        scores = metrics.evaluate(g["price_eur"], g["price_eur_pred"], g["price_eur_snaive"])
        rows.append({"market": market, "model": model, **scores})
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
