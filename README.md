# Energy Price Forecast

Day-ahead electricity price forecasting for Germany-Luxembourg and Turkey: the same pipeline and
models applied to two very different markets.

Status and next steps: [docs/ROADMAP.md](docs/ROADMAP.md).

## First results: 2025, both markets

Rolling-origin backtest over every delivery day of 2025 (8,760 hours per market). Each forecast is
issued at 11:00 local time on the day before delivery, using only prices published by then and
weather *forecasts* (Open-Meteo historical forecasts, not observed weather). Models are refitted
weekly on an expanding window starting in January 2023. Turkish prices are compared in EUR.

| Model | DE-LU MAE | DE-LU rMAE | Turkey MAE | Turkey rMAE |
|---|---:|---:|---:|---:|
| Seasonal naive (same hour last week) | 32.8 | 1.00 | 9.2 | 1.00 |
| Naive (same hour on the issue day) | 26.0 | 0.79 | 9.0 | 0.98 |
| LightGBM | 17.6 | 0.54 | **6.3** | **0.69** |
| LEAR (LASSO on asinh-transformed prices) | **15.9** | **0.48** | 6.6 | 0.72 |

MAE in EUR/MWh; rMAE is the MAE relative to the seasonal naive on the same hours.

- **Germany is harder in absolute terms but more predictable from weather.** Errors are about
  2.5 times larger, yet the models halve the naive error: with this much wind and solar, a weather
  forecast explains a large part of tomorrow's price.
- **Turkey is calmer but gains less.** The price cap (3,400 TRY/MWh in 2025) and a gas- and
  hydro-driven supply make yesterday's price almost as good as last week's (rMAE 0.98), and the
  models improve on it by about 30%.
- **The best model differs, until the market's rules are applied.** The linear LEAR wins in
  Germany, as in the forecasting literature (Lago et al., 2021). In Turkey LightGBM wins at
  first, and the error analysis below shows why.

## Why: where the forecasts go wrong

![Forecast error by situation](docs/img/error_by_regime.png)

**Turkey: the linear model ignores the market's bounds.** Turkish prices can be neither negative
nor above a regulated cap, which the market hit in 1,218 hours of 2025 (14%). On public holidays
demand collapses and prices sit at zero for hours; LEAR, a linear model on asinh-transformed
prices, extrapolates far below zero there (89 negative forecasts, down to −145 EUR/MWh during
Kurban Bayramı) and above the cap in 1,030 hours. LightGBM cannot predict outside the range it was
trained on, so it is naturally bounded, which is its whole advantage:

| Turkey, 2025 | MAE (EUR/MWh) |
|---|---:|
| LEAR | 6.61 |
| LightGBM | 6.34 |
| LEAR, never below zero (a fixed market rule) | 6.23 |
| **LEAR, bounded: within zero and the cap** (`lear_bounded`, the pipeline's default) | **5.94** |

Keeping LEAR within the market's bounds makes it the best model in every situation: its error on
zero-price hours falls from 48.0 to 0.8 EUR/MWh. The bounds live in
[`bounds.py`](src/price_forecast/bounds.py) and any model can use them (`make_model("<name>_bounded")`).
The cap is converted to EUR with the issue day's exchange rate, so nothing unknown at forecast
time is used. The cap schedule is in `config.py`: 4,200 TRY in January 2023, cut to 2,600 by
April 2023, then 2,700, 3,000 and 3,400 from April 2025. Where the regulator's (EPDK) decision
could not be found, the date is the first day the market cleared at the new cap.

![Example weeks](docs/img/example_weeks.png)

**Germany: negative prices are the hard part.** Prices fell below zero in 576 hours of 2025, on
sunny, windy days with low demand. LEAR exaggerates these dips (down to −145 EUR/MWh when the
price was −40), and these hours are the only situation where LightGBM is better (25.4 vs 28.4
EUR/MWh). On ordinary hours, holidays included, LEAR stays ahead.

![Monthly error](docs/img/monthly_mae.png)

The seasonal naive is worst in winter and in September–October in Germany, when wind makes prices
swing from week to week; the models stay much flatter. The switch to 15-minute day-ahead products on
1 October 2025 left no visible mark on the hourly forecasts. Full tables:
[docs/analysis/results.md](docs/analysis/results.md).

**Takeaway.** Knowing the market's rules beat changing the model: a two-line bound on the linear
model's output did more than switching to gradient boosting.

## Live forecasts

[![Latest forecasts](https://takaindidenize.github.io/energy-price-forecast/latest.svg)](https://takaindidenize.github.io/energy-price-forecast/)

**[Open the live dashboard](https://takaindidenize.github.io/energy-price-forecast/)**: the chart
above is regenerated every morning (GitHub caches README images for a few hours).

Every morning a GitHub Actions job ([`daily-pipeline.yml`](.github/workflows/daily-pipeline.yml))
ingests the newest prices and weather forecasts, forecasts the next delivery day for both markets
with `lear_bounded`, scores the forecasts whose prices have since been published, and publishes a
static dashboard ([`report.py`](src/price_forecast/report.py)) to GitHub Pages. No server and no
paid cloud: data is carried between runs in the Actions cache, and an empty cache triggers a full
re-ingest.

The first live forecasts, issued on 3 November 2026 for 4 November: Turkey's average forecast was
59.2 EUR/MWh against a realised 59.3, with an hourly MAE of 4.0 (seasonal naive 5.1). Over the
preceding two weeks the bounded model's MAE was 50.3 EUR/MWh in Germany (seasonal naive 118.2)
and 6.1 in Turkey (naive 10.8). German prices jumped to about 250 EUR/MWh from 28 October and the
model has been forecasting about 50 below since: the dashboard shows it, and it is the next thing
to investigate.

Two lessons from going live:

- **Regulated limits change.** EPDK raised the Turkish cap from 3,400 to 4,500 TRY on 4 April
  2026, so a forecaster built on 2025 data clipped every higher price. The monitor now warns when
  realised prices break the configured limits.
- **Tomorrow's weather has to be asked for differently.** Open-Meteo's historical forecast API
  serves the latest forecast for the coming days too, but rejects end dates more than about 15
  days ahead, so the current month is fetched only up to a week ahead.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m price_forecast.ingestion.energy_charts --start 2023-01 --end 2025-12
python -m price_forecast.ingestion.weather --market de_lu --start 2023-01 --end 2025-12
python -m price_forecast.pipeline --market de_lu --date 2025-12-30 --models seasonal_naive,lear_bounded
python -m price_forecast.monitor --market de_lu      # joins forecasts with realised prices
python -m price_forecast.report --out site           # site/index.html, the dashboard
uvicorn price_forecast.api:app                       # http://localhost:8000/docs
```

`python -m price_forecast.pipeline --days 14` backfills the last two weeks of forecasts, so a new
dashboard has history to show.

Turkish prices need an account on the EPİAŞ Transparency Platform; set `EPIAS_USERNAME` and
`EPIAS_PASSWORD` in the environment, then `python -m price_forecast.ingestion.epias`.

Reproduce the analysis (after ingesting both markets for 2023–2025):

```bash
pip install -e ".[analysis]"
python analysis/run_backtests.py      # a few minutes; writes data/backtest_<market>_2025.parquet
python analysis/error_analysis.py     # writes docs/analysis/results.md and docs/img/*.png
```

## Data sources

- Day-ahead prices for DE-LU: [Energy-Charts](https://energy-charts.info) (Fraunhofer ISE), data
  from Bundesnetzagentur | SMARD.de, CC BY 4.0.
- Turkish market clearing prices: [EPİAŞ Transparency Platform](https://seffaflik.epias.com.tr).
- Weather: [Open-Meteo](https://open-meteo.com) historical forecast API.
