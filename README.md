# Energy Price Forecast

Day-ahead electricity price forecasting for Germany-Luxembourg and Turkey: the same pipeline and
models applied to two very different markets.

Work in progress; see [docs/ROADMAP.md](docs/ROADMAP.md).

## First results: Germany-Luxembourg, 2025

Rolling-origin backtest over every delivery day of 2025 (8,760 hours). Each forecast is issued at
11:00 Berlin time on the day before delivery, using only prices published by then and weather
*forecasts* (Open-Meteo historical forecasts, not observed weather). Models are refitted weekly on
an expanding window starting in January 2023.

| Model | MAE (EUR/MWh) | RMSE (EUR/MWh) | rMAE |
|---|---:|---:|---:|
| Seasonal naive (same hour last week) | 32.8 | 49.4 | 1.00 |
| Naive (same hour on the issue day) | 26.0 | 40.5 | 0.79 |
| LightGBM | 17.6 | 27.3 | 0.54 |
| **LEAR** (LASSO on asinh-transformed prices) | **15.9** | **25.3** | **0.48** |

rMAE is the MAE relative to the seasonal naive on the same hours. The linear LEAR model beating
LightGBM matches the electricity price forecasting literature (Lago et al., 2021).

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
