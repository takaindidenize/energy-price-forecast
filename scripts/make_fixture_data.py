"""Write deterministic synthetic raw prices and weather that follow the raw data contracts.

The fixtures exist so that features, models, backtests and CI can run without network access or
real data. They are built to exercise the awkward parts of both markets:

- de_lu: negative prices (windy and sunny low-demand hours), and 15-minute day-ahead products from
  delivery day 2025-10-01 (local) onwards, as on the real market.
- tr: a price cap in TRY (the cap's EUR level drifts down as the lira depreciates), a floor at 0,
  and a few hours with a missing EUR price.
- Both: national holidays (lower demand), daylight saving switches (Europe/Berlin has 23- and
  25-hour days; Turkey has had no DST since 2016, so it never does).

Prices are a noisy function of the synthetic weather and calendar, so models that use weather
should beat the naive benchmarks, as they do on real data. Nothing here is calibrated to real
market data; the numbers are only plausible in size.

Usage (PF_DATA_DIR must point to a scratch directory, never the real data directory):

    PF_DATA_DIR=/tmp/pf-fixtures python scripts/make_fixture_data.py [--start 2025-01 --end 2025-11]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import lfilter

from price_forecast import config

log = logging.getLogger("make_fixture_data")

FIXTURE_LOADED_AT = datetime(2026, 1, 1)  # fixed, so that output is byte-for-byte reproducible
# The European day-ahead market switched to 15-minute products for delivery from this local day.
QUARTER_HOUR_START_LOCAL = date(2025, 10, 1)
TR_PRICE_CAP_TRY = 3400.0  # synthetic constant, in the range of EPİAŞ's maximum price limit
DE_PRICE_FLOOR_EUR = -500.0  # harmonised minimum clearing price of the European day-ahead market
MARKER_FILE = "FIXTURE_DATA"  # written into PF_DATA_DIR so a fixture directory is recognisable

PRICE_SCHEMA = pa.schema(
    [
        ("delivery_start", pa.timestamp("us")),
        ("resolution_minutes", pa.int16()),
        ("price", pa.float64()),
        ("price_eur", pa.float64()),
        ("currency", pa.string()),
        ("source", pa.string()),
        ("loaded_at", pa.timestamp("us")),
    ]
)
WEATHER_SCHEMA = pa.schema(
    [
        ("point", pa.string()),
        ("valid_time", pa.timestamp("us")),
        *[(var, pa.float64()) for var in config.WEATHER_VARIABLES],
        ("loaded_at", pa.timestamp("us")),
    ]
)

# Per-market climate and price-formation parameters. Purely synthetic.
_CLIMATE = {
    "de_lu": {"temp_mean": 10.0, "temp_amp": 9.0, "wind_kmh": 21.0, "cloud_bias": 0.5},
    "tr": {"temp_mean": 15.0, "temp_amp": 10.0, "wind_kmh": 17.0, "cloud_bias": -0.7},
}
# TRY per EUR, log-linear between two fixed anchors so any month range gives the same rates.
_FX_ANCHORS = ((pd.Timestamp("2023-01-01"), 20.0), (pd.Timestamp("2026-01-01"), 50.0))


def _month_range(start_month: str, end_month: str) -> list[str]:
    months = pd.period_range(start_month, end_month, freq="M")
    if len(months) == 0:
        raise ValueError(f"empty month range {start_month}..{end_month}")
    return [str(p) for p in months]


def _utc_hours(start_month: str, end_month: str) -> pd.DatetimeIndex:
    """Naive UTC hours from the first hour of `start_month` to the last hour of `end_month`."""
    start = pd.Period(start_month, freq="M").start_time
    end = pd.Period(end_month, freq="M").end_time.floor("h")
    return pd.date_range(start, end, freq="h")


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    """Stationary AR(1) series with marginal standard deviation `sd`."""
    eps = rng.normal(0.0, sd * np.sqrt(1.0 - phi**2), n)
    eps[0] = rng.normal(0.0, sd)
    return lfilter([1.0], [1.0, -phi], eps)


def _day_of_year(hours: pd.DatetimeIndex) -> np.ndarray:
    return hours.dayofyear.to_numpy() + hours.hour.to_numpy() / 24.0


def generate_weather(
    market: str, hours: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    """Hourly weather per configured point: a shared synoptic signal plus local noise."""
    clim = _CLIMATE[market]
    n = len(hours)
    doy = _day_of_year(hours)
    utc_hour = hours.hour.to_numpy().astype(float)
    season = np.cos(2 * np.pi * (doy - 20) / 365.25)  # +1 mid-January, -1 mid-July

    shared_temp = _ar1(rng, n, 0.995, 3.0)
    shared_wind = _ar1(rng, n, 0.97, 0.45)
    shared_cloud = _ar1(rng, n, 0.95, 1.0)

    frames = []
    for point in config.MARKETS[market].weather_points:
        solar_time = utc_hour + point.lon / 15.0
        temp = (
            clim["temp_mean"]
            - clim["temp_amp"] * season
            + 4.0 * np.cos(2 * np.pi * (solar_time - 15.0) / 24.0)
            + shared_temp
            + _ar1(rng, n, 0.9, 1.0)
        )
        # Windier in winter and further north.
        log_wind = (
            np.log(clim["wind_kmh"])
            + 0.03 * (point.lat - 45.0)
            + 0.15 * season
            + shared_wind
            + _ar1(rng, n, 0.9, 0.2)
        )
        wind = np.exp(log_wind)
        cloud_latent = clim["cloud_bias"] + 0.6 * season + shared_cloud + _ar1(rng, n, 0.9, 0.5)
        cloud = 100.0 / (1.0 + np.exp(-1.5 * cloud_latent))

        declination = np.radians(23.44) * np.sin(2 * np.pi * (doy - 81) / 365.25)
        hour_angle = np.radians((solar_time - 12.0) * 15.0)
        lat = np.radians(point.lat)
        sin_elev = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(
            hour_angle
        )
        clear_sky = 1000.0 * np.clip(sin_elev, 0.0, None) ** 1.15
        radiation = clear_sky * (1.0 - 0.75 * (cloud / 100.0) ** 3)

        frames.append(
            pd.DataFrame(
                {
                    "point": point.name,
                    "valid_time": hours,
                    "temperature_2m": np.round(temp, 1),
                    "wind_speed_100m": np.round(wind, 1),
                    "shortwave_radiation": np.round(radiation, 0),
                    "cloud_cover": np.round(cloud, 0),
                    "loaded_at": FIXTURE_LOADED_AT,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _calendar(market: str, hours: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local hour, local day of year and a low-demand flag (weekend or holiday) per UTC hour."""
    local = hours.tz_localize("UTC").tz_convert(config.MARKETS[market].timezone)
    years = range(local.year.min(), local.year.max() + 1)
    hols = holidays.country_holidays(config.MARKETS[market].holiday_country, years=years)
    local_dates = pd.Series(local.date)
    is_holiday = local_dates.map(lambda d: d in hols).to_numpy(dtype=bool)
    low_demand = (local.dayofweek.to_numpy() >= 5) | is_holiday
    hour_local = local.hour.to_numpy().astype(float)
    return hour_local, local.dayofyear.to_numpy().astype(float), low_demand


def _bump(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-(((x - centre) / width) ** 2))


def _capacity_factor_wind(wind_kmh: np.ndarray) -> np.ndarray:
    ms = wind_kmh / 3.6
    return np.clip((ms - 3.0) / 9.0, 0.0, 1.0) ** 2


def _latent_hourly_price_eur(
    market: str, hours: pd.DatetimeIndex, weather: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """A merit-order-shaped hourly price in EUR/MWh driven by residual demand."""
    w = weather.groupby("valid_time")[list(config.WEATHER_VARIABLES)].mean().reindex(hours)
    temp = w["temperature_2m"].to_numpy()
    wind_cf = _capacity_factor_wind(w["wind_speed_100m"].to_numpy())
    solar_cf = w["shortwave_radiation"].to_numpy() / 1000.0
    hour_local, doy_local, low_demand = _calendar(market, hours)
    n = len(hours)
    fuel = np.exp(_ar1(rng, n, 0.999, 0.12))  # slow fuel and carbon price drift
    noise = _ar1(rng, n, 0.8, 1.0)

    if market == "de_lu":
        load = 45.0 + 15.0 * _bump(hour_local, 12.5, 5.0) + 4.0 * _bump(hour_local, 19.0, 2.0)
        load = load * np.where(low_demand, 0.85, 1.0) + 0.4 * np.clip(15.0 - temp, 0.0, None)
        residual = load - 50.0 * wind_cf - 52.0 * solar_cf
        price = np.where(
            residual >= 0.0, 20.0 + 1.6 * residual + 0.012 * residual**2, 20.0 + 3.0 * residual
        )
        price = fuel * price + 8.0 * noise
        return np.clip(price, DE_PRICE_FLOOR_EUR, None)

    load = 30.0 + 6.0 * _bump(hour_local, 14.0, 5.0) + 5.0 * _bump(hour_local, 20.5, 1.8)
    load = load * np.where(low_demand, 0.9, 1.0)
    load += 0.25 * np.clip(temp - 24.0, 0.0, None) + 0.2 * np.clip(10.0 - temp, 0.0, None)
    hydro = 6.0 + 5.0 * _bump(doy_local, 120.0, 40.0)  # spring snowmelt
    residual = load - hydro - 13.0 * wind_cf - 16.0 * solar_cf
    return fuel * (12.0 + 2.2 * residual) + 6.0 * noise


def _try_per_eur(hours: pd.DatetimeIndex) -> np.ndarray:
    (t0, fx0), (t1, fx1) = _FX_ANCHORS
    frac = (hours - t0) / (t1 - t0)
    return fx0 * (fx1 / fx0) ** np.asarray(frac, dtype=float)


def generate_prices(
    market: str, hours: pd.DatetimeIndex, weather: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Raw price rows for `market` at the source's resolution (see the contract in CLAUDE.md)."""
    m = config.MARKETS[market]
    latent = _latent_hourly_price_eur(market, hours, weather, rng)

    if market == "de_lu":
        switch = (
            pd.Timestamp(QUARTER_HOUR_START_LOCAL).tz_localize(m.timezone).tz_convert("UTC")
        ).tz_localize(None)
        hourly_mask = hours < switch
        parts = [
            pd.DataFrame(
                {
                    "delivery_start": hours[hourly_mask],
                    "resolution_minutes": 60,
                    "price": latent[hourly_mask],
                }
            )
        ]
        if (~hourly_mask).any():
            # Quarter-hours ramp between neighbouring hours; the ramp averages out over the hour.
            prev_p = np.r_[latent[0], latent[:-1]]
            next_p = np.r_[latent[1:], latent[-1]]
            slope = (next_p - prev_p) / 8.0
            q_hours = hours[~hourly_mask]
            for q in range(4):
                p = latent[~hourly_mask] + slope[~hourly_mask] * (q - 1.5)
                p = p + rng.normal(0.0, 3.0, len(p))
                parts.append(
                    pd.DataFrame(
                        {
                            "delivery_start": q_hours + pd.Timedelta(minutes=15 * q),
                            "resolution_minutes": 15,
                            "price": np.clip(p, DE_PRICE_FLOOR_EUR, None),
                        }
                    )
                )
        df = pd.concat(parts, ignore_index=True).sort_values("delivery_start")
        df["price"] = df["price"].round(2)
        df["price_eur"] = df["price"]
    else:
        fx = _try_per_eur(hours)
        price_try = np.clip(np.round(latent * fx, 2), 0.0, TR_PRICE_CAP_TRY)
        price_eur = np.round(price_try / fx, 2)
        price_eur[rng.random(len(hours)) < 0.003] = np.nan  # EPİAŞ EUR price occasionally missing
        df = pd.DataFrame(
            {
                "delivery_start": hours,
                "resolution_minutes": 60,
                "price": price_try,
                "price_eur": price_eur,
            }
        )

    df["resolution_minutes"] = df["resolution_minutes"].astype("int16")
    df["currency"] = m.currency
    df["source"] = m.source
    df["loaded_at"] = FIXTURE_LOADED_AT
    return df[PRICE_SCHEMA.names].reset_index(drop=True)


def _write_partitions(
    df: pd.DataFrame, time_col: str, base: Path, market: str, schema: pa.Schema
) -> list[Path]:
    written = []
    months = df[time_col].dt.to_period("M").astype(str)
    for month, part in df.groupby(months, sort=True):
        if part.empty:
            continue
        path = config.month_partition_path(base, market, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(part, schema=schema, preserve_index=False)
        pq.write_table(table, path)
        written.append(path)
    return written


def write_fixtures(
    prices_dir: Path,
    weather_dir: Path,
    start_month: str,
    end_month: str,
    markets: Iterable[str] | None = None,
    seed: int = 0,
) -> list[Path]:
    """Write fixture partitions for `markets` (default: all) and return the written paths.

    Partitions are by UTC month of `delivery_start` / `valid_time`. Same arguments give identical
    files.
    """
    _month_range(start_month, end_month)  # validates the range
    hours = _utc_hours(start_month, end_month)
    written: list[Path] = []
    for i, market in enumerate(markets or config.MARKETS):
        if market not in config.MARKETS:
            raise ValueError(f"unknown market {market!r}")
        rng = np.random.default_rng([seed, i])
        weather = generate_weather(market, hours, rng)
        prices = generate_prices(market, hours, weather, rng)
        written += _write_partitions(prices, "delivery_start", prices_dir, market, PRICE_SCHEMA)
        written += _write_partitions(weather, "valid_time", weather_dir, market, WEATHER_SCHEMA)
        log.info("%s: %d price rows, %d weather rows", market, len(prices), len(weather))
    return written


def _check_target_dir() -> Path:
    """Refuse to write unless PF_DATA_DIR names a scratch directory that holds no real data."""
    raw = os.environ.get("PF_DATA_DIR")
    if not raw:
        raise SystemExit(
            "PF_DATA_DIR is not set. Point it at a scratch directory, e.g.\n"
            "  PF_DATA_DIR=/tmp/pf-fixtures python scripts/make_fixture_data.py"
        )
    data_dir = Path(raw).resolve()
    if data_dir == (config.PROJECT_ROOT / "data").resolve():
        raise SystemExit(f"PF_DATA_DIR={data_dir} is the default real data directory; refusing.")
    if data_dir != config.DATA_DIR.resolve():
        raise SystemExit("PF_DATA_DIR changed after price_forecast.config was imported.")
    has_parquet = config.RAW_DIR.exists() and any(config.RAW_DIR.rglob("*.parquet"))
    if has_parquet and not (data_dir / MARKER_FILE).exists():
        raise SystemExit(
            f"{config.RAW_DIR} already holds Parquet files that were not written by this script; "
            "refusing to mix fixtures with real data."
        )
    return data_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=config.START_MONTH, help="first month, YYYY-MM")
    parser.add_argument("--end", default=config.END_MONTH, help="last month, YYYY-MM")
    parser.add_argument("--market", action="append", choices=sorted(config.MARKETS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    data_dir = _check_target_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / MARKER_FILE).write_text("Synthetic data written by scripts/make_fixture_data.py\n")
    paths = write_fixtures(
        config.PRICES_RAW_DIR, config.WEATHER_RAW_DIR, args.start, args.end, args.market, args.seed
    )
    log.info("wrote %d partitions under %s", len(paths), config.RAW_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
