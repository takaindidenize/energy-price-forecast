"""Keep forecasts within a market's regulated price limits.

A linear model knows nothing about a price floor or cap and extrapolates past them: in the 2025
backtest LEAR forecast negative Turkish prices on public holidays, where the market's floor is
0 TRY, and prices above the cap in about 1,000 hours. Clipping to the limits is the whole fix (see
the README's error analysis).

Limits are in the market's own currency (`config.Market.limits`) while forecasts are in EUR. The
conversion uses the EUR/local rate of the **issue day**, which is the day before delivery: that
day's prices, in both currencies, are published before the forecast is made, so nothing from the
delivery day is used.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from price_forecast import config


def cap_on(market: str, day: date) -> float | None:
    """The cap in local currency for delivery `day`, or None when the market has none."""
    limits = config.MARKETS[market].limits
    if limits is None:
        return None
    cap = None
    for start, value in limits.caps:
        if date.fromisoformat(start) <= day:
            cap = value
    return cap


def eur_per_local(market: str) -> pd.Series | None:
    """EUR per unit of local currency by local delivery date, from the raw prices.

    Uses the source's own EUR conversion (`price_eur / price`). None for EUR markets.
    """
    if config.MARKETS[market].currency == "EUR":
        return None
    import duckdb

    pattern = (config.PRICES_RAW_DIR / f"market={market}" / "*" / "*" / "data.parquet").as_posix()
    raw = duckdb.sql(
        f"SELECT delivery_start, price, price_eur FROM read_parquet('{pattern}') "
        "WHERE price > 0 AND price_eur IS NOT NULL"
    ).df()
    if raw.empty:
        return pd.Series(dtype="float64")
    tz = config.MARKETS[market].timezone
    local_date = (
        pd.to_datetime(raw["delivery_start"]).dt.tz_localize("UTC").dt.tz_convert(tz).dt.date
    )
    rate = (raw["price_eur"] / raw["price"]).groupby(local_date).mean()
    return rate.sort_index()


def limits_eur(
    market: str, delivery_dates: pd.Series, rates: pd.Series | None
) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) bounds in EUR/MWh per row; -inf/+inf where there is no limit.

    `rates` is `eur_per_local(market)`. Each row uses the rate of the latest day up to its issue
    day (delivery day - 1) that has one.
    """
    n = len(delivery_dates)
    lower = np.full(n, -np.inf)
    upper = np.full(n, np.inf)
    limits = config.MARKETS[market].limits
    if limits is None:
        return lower, upper

    days = pd.Series(delivery_dates).map(lambda d: pd.Timestamp(d).date())
    if rates is None:  # EUR market: limits are already in EUR
        rate = np.ones(n)
    else:
        known = rates.copy()
        known.index = pd.to_datetime(known.index)
        issue_days = pd.to_datetime(days.map(lambda d: d - timedelta(days=1)))
        rate = known.asof(pd.DatetimeIndex(issue_days)).to_numpy(dtype="float64")
    caps = days.map(lambda d: cap_on(market, d)).to_numpy(dtype="float64")
    if limits.floor is not None:
        # A floor of zero needs no exchange rate.
        unknown = 0.0 if limits.floor == 0 else -np.inf
        lower = np.where(np.isnan(rate), unknown, limits.floor * rate)
    upper = np.where(np.isnan(rate) | np.isnan(caps), np.inf, caps * rate)
    return lower, upper


def clip(
    market: str, delivery_dates: pd.Series, predictions: pd.Series, rates: pd.Series | None
) -> pd.Series:
    lower, upper = limits_eur(market, delivery_dates, rates)
    values = np.clip(predictions.to_numpy(dtype="float64"), lower, upper)
    return pd.Series(values, index=predictions.index, name=predictions.name)
