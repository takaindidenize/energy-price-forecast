# Energy Price Forecast

Day-ahead electricity price forecasting for Germany-Luxembourg and Turkey: the same pipeline and
models applied to two very different markets.

Work in progress; see [docs/ROADMAP.md](docs/ROADMAP.md).

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
- **The best model differs.** The linear LEAR wins in Germany, as in the forecasting literature
  (Lago et al., 2021); LightGBM wins in Turkey. Why is the next analysis step.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m price_forecast.ingestion.energy_charts --start 2023-01 --end 2025-12
python -m price_forecast.ingestion.weather --market de_lu --start 2023-01 --end 2025-12
python -m price_forecast.pipeline --market de_lu --date 2025-12-30 --models seasonal_naive,lear
python -m price_forecast.monitor --market de_lu      # joins forecasts with realised prices
uvicorn price_forecast.api:app                       # http://localhost:8000/docs
```

Turkish prices need an account on the EPİAŞ Transparency Platform; set `EPIAS_USERNAME` and
`EPIAS_PASSWORD` in the environment, then `python -m price_forecast.ingestion.epias`.

## Data sources

- Day-ahead prices for DE-LU: [Energy-Charts](https://energy-charts.info) (Fraunhofer ISE), data
  from Bundesnetzagentur | SMARD.de, CC BY 4.0.
- Turkish market clearing prices: [EPİAŞ Transparency Platform](https://seffaflik.epias.com.tr).
- Weather: [Open-Meteo](https://open-meteo.com) historical forecast API.
