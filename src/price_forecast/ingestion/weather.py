"""Hourly weather forecasts per market weather point from Open-Meteo.

Source: Open-Meteo's historical forecast API (`config.OPEN_METEO_HISTORICAL_FORECAST_URL`), which
stores weather model forecasts as they were issued, so backtests only see weather a forecaster could
have known. For hours the forecast API does not cover (no row or every variable null, or the API
rejects the date range) we fall back to the reanalysis archive (`config.OPEN_METEO_ARCHIVE_URL`)
and log it, because archive values are observations, not forecasts.

Variables: the output columns are `config.WEATHER_VARIABLES`. The forecast API serves wind at the
weather models' own levels (10, 80, 120, 180 m), not at 100 m, so `wind_speed_100m` is interpolated
from 80 m and 120 m on log-height (see `wind_speed_100m`). The archive (ERA5) has 100 m natively.
All requests use UTC (`timezone=GMT`), unix timestamps and km/h.

    python -m price_forecast.ingestion.weather --market de_lu --start 2023-01 --end 2025-12
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pyarrow as pa

from price_forecast import config
from price_forecast.config import WeatherPoint
from price_forecast.ingestion.energy_charts import (
    get_json,
    iter_months,
    month_bounds_utc,
    new_client,
    utc_now_naive,
    write_partition,
)

log = logging.getLogger(__name__)

# What each API is asked for. Everything except the wind levels maps 1:1 to WEATHER_VARIABLES.
FORECAST_REQUEST_VARIABLES = (
    "temperature_2m",
    "wind_speed_80m",
    "wind_speed_120m",
    "shortwave_radiation",
    "cloud_cover",
)
ARCHIVE_REQUEST_VARIABLES = tuple(config.WEATHER_VARIABLES)

WEATHER_SCHEMA = pa.schema(
    [
        ("point", pa.string()),
        ("valid_time", pa.timestamp("us")),
        *[(name, pa.float64()) for name in config.WEATHER_VARIABLES],
        ("loaded_at", pa.timestamp("us")),
    ]
)

# Wind shear exponent for extrapolating from a single level (1/7 power law, neutral conditions).
_SHEAR_ALPHA = 1 / 7
_W_100 = math.log(100 / 80) / math.log(120 / 80)


# How far past today the current month is fetched. The historical forecast API also serves the
# latest forecast for the coming days (up to about 15 days ahead) and rejects a request whose
# end date is further out, so a whole current month cannot be requested in one go. A week covers
# the next delivery day with room to spare.
FORECAST_HORIZON_DAYS = 7


class NoCoverage(Exception):
    """The API rejected the request (HTTP 400), typically a date range outside its coverage."""


def _fetch_window(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC [start, end) of `month`, cut at the forecast horizon for the current month."""
    start, end = month_bounds_utc(month)
    today = pd.Timestamp(utc_now_naive()).normalize()
    horizon = today + pd.Timedelta(days=FORECAST_HORIZON_DAYS + 1)
    return pd.Timestamp(start), min(pd.Timestamp(end), horizon)


def wind_speed_100m(v80: pd.Series, v120: pd.Series) -> pd.Series:
    """Wind speed at 100 m from 80 m and 120 m: linear in log-height, a 0.55/0.45 blend.

    If only one level is present it is scaled with the 1/7 power law.
    """
    both = v80 + _W_100 * (v120 - v80)
    only80 = v80 * (100 / 80) ** _SHEAR_ALPHA
    only120 = v120 * (100 / 120) ** _SHEAR_ALPHA
    return both.fillna(only80).fillna(only120)


def _request(
    client: httpx.Client,
    url: str,
    point: WeatherPoint,
    start_date: str,
    end_date: str,
    variables: tuple[str, ...],
) -> pd.DataFrame:
    params = {
        "latitude": point.lat,
        "longitude": point.lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": "GMT",
        "timeformat": "unixtime",
        "wind_speed_unit": "kmh",
    }
    response = get_json(client, url, params)
    if response.status_code == 400:
        try:
            reason = response.json().get("reason", response.text)
        except ValueError:
            reason = response.text
        raise NoCoverage(str(reason)[:300])
    response.raise_for_status()
    hourly: dict[str, Any] = response.json().get("hourly") or {}
    times = hourly.get("time") or []
    df = pd.DataFrame({"valid_time": pd.to_datetime(pd.Series(times, dtype="int64"), unit="s")})
    for name in variables:
        values = hourly.get(name)
        if values is None:
            log.warning("%s: %s missing from the response for %s", url, name, point.name)
            values = [None] * len(times)
        df[name] = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").astype(
            "float64"
        )
    df["valid_time"] = df["valid_time"].astype("datetime64[us]")
    return df


def _forecast_frame(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw[["valid_time", "temperature_2m", "shortwave_radiation", "cloud_cover"]].copy()
    out["wind_speed_100m"] = wind_speed_100m(raw["wind_speed_80m"], raw["wind_speed_120m"])
    return out[["valid_time", *config.WEATHER_VARIABLES]]


def _covered(df: pd.DataFrame) -> pd.Series:
    """Rows with at least one non-null variable."""
    return df[list(config.WEATHER_VARIABLES)].notna().any(axis=1)


def fetch_point_month(client: httpx.Client, point: WeatherPoint, month: str) -> pd.DataFrame:
    """Hourly values for one point and UTC month: historical forecasts, archive for the gaps.

    For the current month only the days up to `FORECAST_HORIZON_DAYS` ahead are fetched.
    """
    start, end = _fetch_window(month)
    return fetch_point(client, point, start, end, month)


def fetch_point(
    client: httpx.Client, point: WeatherPoint, start: pd.Timestamp, end: pd.Timestamp, label: str
) -> pd.DataFrame:
    """Hourly values for one point over UTC [start, end), in one request per API.

    The APIs return a whole year in one response, so a long range is far fewer requests than one
    per month; that matters on shared CI runners, where each Open-Meteo request can take a minute.
    """
    month = label
    if end <= start:
        return pd.DataFrame(columns=["valid_time", *config.WEATHER_VARIABLES])
    first_day, last_day = f"{start:%Y-%m-%d}", f"{end - timedelta(days=1):%Y-%m-%d}"
    hours = pd.date_range(start, end, freq="h", inclusive="left").astype("datetime64[us]")

    try:
        raw = _request(
            client,
            config.OPEN_METEO_HISTORICAL_FORECAST_URL,
            point,
            first_day,
            last_day,
            FORECAST_REQUEST_VARIABLES,
        )
        forecast = _forecast_frame(raw)
        forecast = forecast[forecast["valid_time"].isin(hours) & _covered(forecast)]
    except NoCoverage as exc:
        log.warning(
            "%s %s: historical forecast API rejected the request: %s", point.name, month, exc
        )
        forecast = pd.DataFrame(columns=["valid_time", *config.WEATHER_VARIABLES])

    missing = hours[~hours.isin(forecast["valid_time"])]
    if missing.empty:
        return forecast.reset_index(drop=True)

    from_day, to_day = f"{missing.min():%Y-%m-%d}", f"{missing.max():%Y-%m-%d}"
    log.warning(
        "%s %s: %d hour(s) not covered by the historical forecast API, falling back to the "
        "archive (reanalysis, not forecasts) for %s..%s",
        point.name,
        month,
        len(missing),
        from_day,
        to_day,
    )
    try:
        archive = _request(
            client,
            config.OPEN_METEO_ARCHIVE_URL,
            point,
            from_day,
            to_day,
            ARCHIVE_REQUEST_VARIABLES,
        )
    except NoCoverage as exc:
        log.warning("%s %s: archive API rejected the request: %s", point.name, month, exc)
        return forecast.reset_index(drop=True)
    archive = archive[archive["valid_time"].isin(missing) & _covered(archive)]
    log.info("%s %s: %d hour(s) filled from the archive", point.name, month, len(archive))
    frames = [f for f in (forecast, archive) if not f.empty]
    if not frames:
        return forecast.iloc[0:0]
    return pd.concat(frames, ignore_index=True).sort_values("valid_time", ignore_index=True)


def _check_quality(df: pd.DataFrame, market: str, month: str, n_hours: int) -> pd.DataFrame:
    dupes = df.duplicated(["point", "valid_time"], keep="last")
    if dupes.any():
        log.warning("%s %s: dropped %d duplicate (point, hour) row(s)", market, month, dupes.sum())
        df = df[~dupes]
    for point, rows in df.groupby("point"):
        if len(rows) < n_hours:
            log.warning(
                "%s %s: %s is missing %d of %d hours",
                market,
                month,
                point,
                n_hours - len(rows),
                n_hours,
            )
    nulls = df[list(config.WEATHER_VARIABLES)].isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        log.warning("%s %s: null values per variable: %s", market, month, nulls.to_dict())
    return df


def ingest_month(market: str, month: str) -> Path | None:
    """Fetch one UTC month for every weather point of `market` and write its partition."""
    written, _ = ingest_months(market, [month])
    return written[0] if written else None


def ingest_months(market: str, months: list[str]) -> tuple[list[Path], list[str]]:
    """Fetch `months` for every weather point with one request per point and calendar year, then
    write one partition per month. Returns (written paths, months without data)."""
    by_year: dict[str, list[str]] = {}
    for month in months:
        by_year.setdefault(month[:4], []).append(month)
    written: list[Path] = []
    empty: list[str] = []
    with new_client() as client:
        for year_months in by_year.values():
            start = _fetch_window(year_months[0])[0]
            end = _fetch_window(year_months[-1])[1]
            frames = []
            for point in config.MARKETS[market].weather_points:
                label = f"{year_months[0]}..{year_months[-1]}"
                df = fetch_point(client, point, start, end, label)
                if not df.empty:
                    frames.append(df.assign(point=point.name))
            data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            for month in year_months:
                path = _write_month(market, month, data)
                if path is None:
                    empty.append(month)
                else:
                    written.append(path)
    return written, empty


def _write_month(market: str, month: str, data: pd.DataFrame) -> Path | None:
    start, end = _fetch_window(month)
    n_hours = max(0, int((end - start) / timedelta(hours=1)))
    if data.empty:
        df = data
    else:
        df = data[(data["valid_time"] >= start) & (data["valid_time"] < end)]
    if df.empty:
        log.warning("%s %s: no weather data at any point, nothing written", market, month)
        return None
    for point in config.MARKETS[market].weather_points:
        if not (df["point"] == point.name).any():
            log.warning("%s %s: no weather data for %s", market, month, point.name)

    df = _check_quality(df, market, month, n_hours)
    df = df.sort_values(["point", "valid_time"], ignore_index=True)
    df["loaded_at"] = pd.Timestamp(utc_now_naive()).as_unit("us")
    for name in config.WEATHER_VARIABLES:
        df[name] = df[name].astype(np.float64)
    df = df[[field.name for field in WEATHER_SCHEMA]]
    table = pa.Table.from_pandas(df, schema=WEATHER_SCHEMA, preserve_index=False)
    path = write_partition(
        table, config.month_partition_path(config.WEATHER_RAW_DIR, market, month)
    )
    log.info(
        "%s %s: wrote %d rows for %d point(s) to %s",
        market,
        month,
        len(df),
        df["point"].nunique(),
        path,
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest Open-Meteo historical weather forecasts.")
    parser.add_argument("--market", required=True, choices=sorted(config.MARKETS))
    parser.add_argument("--start", default=config.START_MONTH, help="first month, YYYY-MM")
    parser.add_argument("--end", default=config.END_MONTH, help="last month (inclusive), YYYY-MM")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    written, empty = ingest_months(args.market, list(iter_months(args.start, args.end)))
    log.info("Done: %d month(s) written, %d without data", len(written), len(empty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
