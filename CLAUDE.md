# Project guide for Claude

Portfolio project: day-ahead electricity price forecasting for two very different markets,
Germany-Luxembourg (lots of wind and solar, negative prices) and Turkey (price cap, gas and hydro).
The question: how well does the same forecasting approach work in both, and why does it differ?
Code, comments, commits and docs are in English. Read `docs/ROADMAP.md` first.

## Stack
Python 3.11+, httpx + tenacity, DuckDB + Parquet, pandas, scikit-learn, LightGBM, `holidays`,
FastAPI, pytest, ruff. Dependencies are declared in `pyproject.toml`; add one only if truly needed
and say why in the PR.

## Layout and ownership
Several sessions work on this repo in parallel, each on its own branch. Stay inside your area:

| Area | Files |
|---|---|
| European prices + weather ingestion | `src/price_forecast/ingestion/energy_charts.py`, `src/price_forecast/ingestion/weather.py`, `tests/test_energy_charts.py`, `tests/test_weather.py` |
| Turkish prices ingestion (EPİAŞ) | `src/price_forecast/ingestion/epias.py`, `tests/test_epias.py` |
| Features, models, backtesting | `src/price_forecast/features.py`, `src/price_forecast/metrics.py`, `src/price_forecast/backtest.py`, `src/price_forecast/models/*.py` except `base.py`, `scripts/make_fixture_data.py`, `tests/test_features.py`, `tests/test_models.py`, `tests/test_backtest.py` |
| Pipeline, serving, monitoring, CI | `src/price_forecast/pipeline.py`, `src/price_forecast/store.py`, `src/price_forecast/api.py`, `src/price_forecast/monitor.py`, `Dockerfile`, `.github/**`, `tests/test_pipeline.py`, `tests/test_api.py`, `tests/test_monitor.py` |

Shared, do not change without updating every consumer in the same PR: `src/price_forecast/config.py`,
`src/price_forecast/models/base.py`, the contracts below. `src/price_forecast/ingestion/__init__.py`
may be created by whichever ingestion session needs it first (empty docstring only).

## Conventions
- All timestamps are stored in **UTC** (naive `TIMESTAMP` columns hold UTC). Local time only
  appears in derived columns named `*_local`. Daylight saving days have 23 or 25 local hours; key
  everything on UTC.
- Paths, months and markets come from `price_forecast.config`, never hard-coded.
- Ingestion is idempotent: re-running a month overwrites that month's partition. Months with no
  data are logged and skipped, not written as empty files.
- Tests never hit the network (mock with `respx`) and never contain real credentials.
- **Credentials**: EPİAŞ needs a username and password. Read them only from the environment
  variables `EPIAS_USERNAME` and `EPIAS_PASSWORD`; never log, print or commit them, and fail with a
  clear message when they are missing.
- **No look-ahead.** A forecast issued on day D at `issue_hour_local` for delivery day D+1 may use
  only information published before that moment. Every feature must say when it becomes known.
- Metrics: MAE and RMSE in EUR/MWh, and rMAE (MAE divided by the MAE of the weekly seasonal naive
  forecast on the same rows). Never MAPE: prices cross zero.
- Run `ruff check . && ruff format --check . && pytest` before committing.

## Raw data contracts

`data/raw/prices/market=<code>/year=YYYY/month=MM/data.parquet`, one file per market and month,
at the source's own resolution:

| column | type | notes |
|---|---|---|
| `delivery_start` | TIMESTAMP | UTC start of the delivery interval |
| `resolution_minutes` | SMALLINT | 60, or 15: the European day-ahead market switched to 15-minute products on 2025-10-01 |
| `price` | DOUBLE | per MWh in the market's own currency (`config.MARKETS[code].currency`) |
| `price_eur` | DOUBLE | per MWh in EUR. Germany: equals `price`. Turkey: EPİAŞ's own EUR price; null if missing |
| `currency` | VARCHAR | `EUR` or `TRY` |
| `source` | VARCHAR | `energy_charts` or `epias` |
| `loaded_at` | TIMESTAMP | UTC |

`data/raw/weather/market=<code>/year=YYYY/month=MM/data.parquet`, hourly, one row per weather point
and hour:

| column | type | notes |
|---|---|---|
| `point` | VARCHAR | `config.MARKETS[code].weather_points[*].name` |
| `valid_time` | TIMESTAMP | UTC hour the values are for |
| `temperature_2m` | DOUBLE | °C |
| `wind_speed_100m` | DOUBLE | km/h, hub height |
| `shortwave_radiation` | DOUBLE | W/m² |
| `cloud_cover` | DOUBLE | % |
| `loaded_at` | TIMESTAMP | UTC |

Weather comes from Open-Meteo's **historical forecast** API (forecasts as they were issued), not
the reanalysis archive, so that backtests do not use observed weather the forecaster could not have
known. Fall back to the archive only for periods the forecast API does not cover, and log it.

## Module interfaces

```python
# ingestion, each returns the written path or None when the month has no data
price_forecast.ingestion.energy_charts.ingest_month(month: str, market: str = "de_lu") -> Path | None
price_forecast.ingestion.epias.ingest_month(month: str) -> Path | None
price_forecast.ingestion.weather.ingest_month(market: str, month: str) -> Path | None

# modelling
price_forecast.features.load_hourly(market: str) -> pd.DataFrame          # raw parquet -> hourly
price_forecast.features.build_features(hourly: pd.DataFrame, market: str) -> pd.DataFrame
price_forecast.backtest.run(market: str, models: list[Forecaster], test_start: str, test_end: str) -> pd.DataFrame
price_forecast.models.<name> implement price_forecast.models.base.Forecaster
```

## Modelling table (output of `build_features`)

One row per market and delivery hour (UTC). 15-minute prices are averaged to the hour first.

| column | type | known at issue time? |
|---|---|---|
| `market` | str | |
| `delivery_hour` | datetime64 UTC | |
| `delivery_date_local` | date | |
| `hour_local` | int 0–23 | |
| `day_of_week` | int, ISO 1=Mon | yes, calendar |
| `is_holiday` | bool | yes, calendar (`holidays` package, `config.MARKETS[code].holiday_country`) |
| `price_eur` | float | **target**; null for hours not yet delivered |
| `price_d1_same_hour` | float | yes: the issue day's prices were published the day before |
| `price_d7_same_hour` | float | yes |
| `price_d1_mean` | float | yes |
| `temperature_2m`, `wind_speed_100m`, `shortwave_radiation`, `cloud_cover` | float | yes: mean over the market's weather points, from forecasts |

## Forecast store (DuckDB, `config.WAREHOUSE_PATH`)

`forecasts`: `market VARCHAR, model VARCHAR, issued_at TIMESTAMP (UTC), delivery_hour TIMESTAMP (UTC),
price_eur_pred DOUBLE`, unique on (`market`, `model`, `issued_at`, `delivery_hour`).
`backtest_results`: `market, model, delivery_hour, price_eur, price_eur_pred, run_id VARCHAR`.
