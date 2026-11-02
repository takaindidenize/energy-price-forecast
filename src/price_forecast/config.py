"""Shared settings: paths, markets and their weather points. Every module reads these from here."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PF_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PRICES_RAW_DIR = RAW_DIR / "prices"
WEATHER_RAW_DIR = RAW_DIR / "weather"
WAREHOUSE_PATH = DATA_DIR / "warehouse.duckdb"

# Training and backtest range (inclusive months). 2022 is left out on purpose: the gas crisis made
# prices several times higher and far more volatile than before or since.
START_MONTH = os.environ.get("PF_START_MONTH", "2023-01")
END_MONTH = os.environ.get("PF_END_MONTH", "2025-12")

ENERGY_CHARTS_URL = "https://api.energy-charts.info/price"
EPIAS_TICKET_URL = "https://giris.epias.com.tr/cas/v1/tickets"
EPIAS_BASE_URL = "https://seffaflik.epias.com.tr/electricity-service"
OPEN_METEO_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_VARIABLES = (
    "temperature_2m",
    "wind_speed_100m",
    "shortwave_radiation",
    "cloud_cover",
)


@dataclass(frozen=True)
class WeatherPoint:
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class PriceLimits:
    """Regulated price bounds of a market, in its own currency per MWh.

    `caps` is a schedule of (first delivery day it applies to, cap), oldest first; a cap applies
    until the next entry. Regulators publish these decisions in advance, so using them in a
    forecast does not look ahead.
    """

    floor: float | None
    caps: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Market:
    """One bidding zone. `code` is the value of the `market` column everywhere."""

    code: str
    name: str
    timezone: str
    currency: str
    source: str
    holiday_country: str
    # Forecasts for delivery day D+1 are issued on day D at this local hour, before the
    # day-ahead auction closes (12:00 CET in Germany, 12:30 in Turkey).
    issue_hour_local: int
    weather_points: tuple[WeatherPoint, ...]
    # None: no bounds are applied (the European harmonised limits of -500 and +4,000 EUR/MWh
    # are far outside the prices seen here).
    limits: PriceLimits | None = None


MARKETS: dict[str, Market] = {
    "de_lu": Market(
        code="de_lu",
        name="Germany-Luxembourg",
        timezone="Europe/Berlin",
        currency="EUR",
        source="energy_charts",
        holiday_country="DE",
        issue_hour_local=11,
        weather_points=(
            WeatherPoint("hamburg", 53.55, 9.99),  # north: wind
            WeatherPoint("berlin", 52.52, 13.40),
            WeatherPoint("frankfurt", 50.11, 8.68),
            WeatherPoint("munich", 48.14, 11.58),  # south: solar
        ),
    ),
    "tr": Market(
        code="tr",
        name="Turkey",
        timezone="Europe/Istanbul",
        currency="TRY",
        source="epias",
        holiday_country="TR",
        issue_hour_local=11,
        weather_points=(
            WeatherPoint("istanbul", 41.01, 28.98),  # largest load centre
            WeatherPoint("ankara", 39.93, 32.86),
            WeatherPoint("izmir", 38.42, 27.14),  # Aegean wind
            WeatherPoint("konya", 37.87, 32.48),  # solar
        ),
        # EPDK's day-ahead market limits: the floor has been 0 TRY throughout. Cap dates marked
        # "first cleared" are the first day the market cleared at that cap, because the decision
        # itself was not found; check them against EPDK's decisions.
        limits=PriceLimits(
            floor=0.0,
            caps=(
                ("2023-01-01", 4200.0),  # first cleared
                ("2023-02-01", 3650.0),  # first cleared
                ("2023-03-01", 3050.0),  # first cleared
                ("2023-04-01", 2600.0),  # in force from 1 April 2023 (TSKB energy bulletin)
                ("2023-07-05", 2700.0),  # EPDK decision reported on 4 July 2023
                ("2024-07-01", 3000.0),  # first cleared
                ("2025-04-07", 3400.0),  # first cleared
            ),
        ),
    ),
}


def month_partition_path(base: Path, market: str, month: str) -> Path:
    """`<base>/market=<code>/year=YYYY/month=MM/data.parquet` for a 'YYYY-MM' month."""
    year, mm = month.split("-")
    return base / f"market={market}" / f"year={year}" / f"month={mm}" / "data.parquet"
