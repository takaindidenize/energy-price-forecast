"""Raw Parquet partitions -> hourly series -> the modelling table (see CLAUDE.md).

Timing model, used by every feature and by the backtest:

- A forecast for delivery day T (local) is issued on T-1 ("the issue day") at the market's
  `issue_hour_local`, see `issue_time`.
- Prices for delivery day X are treated as known from local midnight at the start of X, see
  `price_known_at`. That is conservative: the day-ahead auction for X clears and is published on
  X-1 around midday. With it, the issue day's own prices are known at issue time and the delivery
  day's prices are not.
- Weather columns hold forecasts for the delivery hour itself, which exist at issue time.

So for a row of delivery day T the price features may only read days T-1 and earlier. They are
built from a per-day profile of 24 local hours: on the 25-hour autumn DST day the repeated hour is
averaged, on the 23-hour spring DST day the skipped hour is filled with the mean of its
neighbours. That keeps "same hour yesterday" defined on the day after a DST switch.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import holidays
import numpy as np
import pandas as pd

from price_forecast import config

log = logging.getLogger(__name__)

WEATHER_COLUMNS: tuple[str, ...] = tuple(config.WEATHER_VARIABLES)
PRICE_FEATURE_COLUMNS = ("price_d1_same_hour", "price_d7_same_hour", "price_d1_mean")
CALENDAR_FEATURE_COLUMNS = ("hour_local", "day_of_week", "is_holiday")
TARGET_COLUMN = "price_eur"
# Exactly the "Modelling table" of CLAUDE.md, in order.
MODEL_TABLE_COLUMNS = (
    "market",
    "delivery_hour",
    "delivery_date_local",
    "hour_local",
    "day_of_week",
    "is_holiday",
    TARGET_COLUMN,
    *PRICE_FEATURE_COLUMNS,
    *WEATHER_COLUMNS,
)
# A daily mean over fewer known hours than this (data edges, gaps) is treated as missing.
MIN_HOURS_FOR_DAILY_MEAN = 18
# Everything a model may use. All of it is known at the issue time of the row's delivery day.
FEATURE_COLUMNS = (*CALENDAR_FEATURE_COLUMNS, *PRICE_FEATURE_COLUMNS, *WEATHER_COLUMNS)


def issue_time(delivery_date: date, market: str) -> pd.Timestamp:
    """UTC issue time of the forecast for local `delivery_date`: the day before, at issue hour."""
    m = config.MARKETS[market]
    issue_day = delivery_date - timedelta(days=1)
    local = datetime.combine(issue_day, time(m.issue_hour_local), tzinfo=ZoneInfo(m.timezone))
    return pd.Timestamp(local).tz_convert("UTC")


def price_known_at(delivery_date: date, market: str) -> pd.Timestamp:
    """UTC time from which the prices of local `delivery_date` count as known: its midnight."""
    m = config.MARKETS[market]
    local = datetime.combine(delivery_date, time(0), tzinfo=ZoneInfo(m.timezone))
    return pd.Timestamp(local).tz_convert("UTC")


def _to_utc(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(pd.to_datetime(values))
    s = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
    return s.dt.as_unit("us")


def _partitions(base: Path, market: str) -> list[str]:
    return sorted(str(p) for p in (base / f"market={market}").glob("year=*/month=*/data.parquet"))


def load_hourly(market: str) -> pd.DataFrame:
    """Read the raw price and weather partitions of `market` into one row per UTC hour.

    Sub-hourly prices (15-minute products) are averaged to the hour. Weather is averaged over the
    market's configured weather points. Hours with weather but no price (e.g. tomorrow) are kept
    with null prices. If an interval appears in several partitions, the latest `loaded_at` wins.

    Columns: market, delivery_hour (UTC, tz-aware), price (own currency), price_eur, n_intervals,
    and the weather variables.
    """
    if market not in config.MARKETS:
        raise ValueError(f"unknown market {market!r}")
    price_files = _partitions(config.PRICES_RAW_DIR, market)
    if not price_files:
        raise FileNotFoundError(
            f"no price partitions for market {market!r} under {config.PRICES_RAW_DIR}"
        )
    con = duckdb.connect()
    prices = con.execute(
        """
        WITH raw AS (
            SELECT delivery_start, price, price_eur
            FROM read_parquet($files, hive_partitioning = false)
            QUALIFY row_number() OVER (PARTITION BY delivery_start ORDER BY loaded_at DESC) = 1
        )
        SELECT date_trunc('hour', delivery_start) AS delivery_hour,
               avg(price) AS price,
               avg(price_eur) AS price_eur,
               count(*)::INTEGER AS n_intervals
        FROM raw
        GROUP BY 1
        """,
        {"files": price_files},
    ).df()

    weather_files = _partitions(config.WEATHER_RAW_DIR, market)
    weather_cols = ", ".join(f"avg({c}) AS {c}" for c in WEATHER_COLUMNS)
    if weather_files:
        points = [p.name for p in config.MARKETS[market].weather_points]
        weather = con.execute(
            f"""
            WITH raw AS (
                SELECT *
                FROM read_parquet($files, hive_partitioning = false)
                WHERE list_contains($points, point)
                QUALIFY row_number() OVER (
                    PARTITION BY point, valid_time ORDER BY loaded_at DESC) = 1
            )
            SELECT date_trunc('hour', valid_time) AS delivery_hour, {weather_cols}
            FROM raw
            GROUP BY 1
            """,
            {"files": weather_files, "points": points},
        ).df()
    else:
        log.warning("no weather partitions for %s under %s", market, config.WEATHER_RAW_DIR)
        weather = pd.DataFrame(columns=["delivery_hour", *WEATHER_COLUMNS])
    con.close()

    prices["delivery_hour"] = _to_utc(prices["delivery_hour"])
    weather["delivery_hour"] = _to_utc(weather["delivery_hour"])
    hourly = prices.merge(weather, on="delivery_hour", how="outer")
    hourly.insert(0, "market", market)
    return hourly.sort_values("delivery_hour").reset_index(drop=True)


def _hourly_grid(hourly: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Every UTC hour from the first input hour through the end of the last local day needed.

    The last day is the later of the last input hour's local day and the day after the last day
    with a known price, so the delivery day of the next forecast always has rows.
    """
    h = hourly.copy()
    h["delivery_hour"] = _to_utc(h["delivery_hour"]).dt.floor("h")
    numeric = [c for c in (TARGET_COLUMN, *WEATHER_COLUMNS) if c in h.columns]
    h = h.groupby("delivery_hour")[numeric].mean()

    last_day = h.index.max().tz_convert(tz).date()
    priced = h.index[h[TARGET_COLUMN].notna()] if TARGET_COLUMN in h else h.index[:0]
    if len(priced):
        last_day = max(last_day, priced.max().tz_convert(tz).date() + timedelta(days=1))
    end = pd.Timestamp(datetime.combine(last_day + timedelta(days=1), time(0))).tz_localize(tz)
    grid = pd.date_range(h.index.min(), end.tz_convert("UTC"), freq="h", inclusive="left")
    out = h.reindex(grid.as_unit("us"))
    for col in (TARGET_COLUMN, *WEATHER_COLUMNS):
        if col not in out:
            out[col] = np.nan
    out.index.name = "delivery_hour"
    return out.reset_index()


def _day_profiles(day: pd.Series, hour: pd.Series, price: pd.Series) -> pd.DataFrame:
    """Local day x 24 local hours of prices; DST-repeated hours averaged, skipped hours filled."""
    frame = pd.DataFrame({"day": day.to_numpy(), "hour": hour.to_numpy(), "price": price})
    grouped = frame.groupby(["day", "hour"])["price"]
    profile = grouped.mean().unstack("hour").reindex(columns=range(24))
    present = grouped.size().unstack("hour").reindex(columns=range(24)).notna()
    # A clock hour absent from the grid between two present hours is the spring DST gap.
    neighbours = (profile.shift(1, axis=1) + profile.shift(-1, axis=1)) / 2
    return profile.where(present, neighbours)


def _lookup(profile: pd.DataFrame, day: pd.Series, hour: pd.Series, lag_days: int) -> np.ndarray:
    rows = profile.index.get_indexer(day - pd.Timedelta(days=lag_days))
    values = profile.to_numpy()[rows, hour.to_numpy()]
    return np.where(rows >= 0, values, np.nan)


def build_features(hourly: pd.DataFrame, market: str) -> pd.DataFrame:
    """The modelling table: one row per UTC delivery hour, columns `MODEL_TABLE_COLUMNS`.

    `hourly` needs `delivery_hour` (UTC) and `price_eur`; weather columns are optional (null if
    absent). See the module docstring for when each feature becomes known.
    """
    if hourly.empty:
        raise ValueError("hourly frame is empty")
    m = config.MARKETS[market]
    df = _hourly_grid(hourly, m.timezone)

    local = df["delivery_hour"].dt.tz_convert(m.timezone)
    day = local.dt.tz_localize(None).dt.normalize()  # naive local midnight, the day key
    hour = local.dt.hour

    profile = _day_profiles(day, hour, df[TARGET_COLUMN])
    by_day = df.groupby(day.to_numpy())[TARGET_COLUMN]
    daily_mean = by_day.mean().where(by_day.count() >= MIN_HOURS_FOR_DAILY_MEAN)

    years = range(day.dt.year.min(), day.dt.year.max() + 1)
    hols = holidays.country_holidays(m.holiday_country, years=years)
    dates = day.dt.date

    out = pd.DataFrame(
        {
            "market": market,
            "delivery_hour": df["delivery_hour"],
            "delivery_date_local": dates,
            "hour_local": hour.astype("int64"),
            "day_of_week": (local.dt.dayofweek + 1).astype("int64"),
            "is_holiday": dates.map(lambda d: d in hols).astype(bool),
            TARGET_COLUMN: df[TARGET_COLUMN].astype(float),
            # Issue day = delivery day - 1: its prices were published the day before it.
            "price_d1_same_hour": _lookup(profile, day, hour, 1),
            "price_d7_same_hour": _lookup(profile, day, hour, 7),
            "price_d1_mean": (day - pd.Timedelta(days=1)).map(daily_mean).astype(float),
        }
    )
    for col in WEATHER_COLUMNS:
        out[col] = df[col].astype(float)
    return out[list(MODEL_TABLE_COLUMNS)]
