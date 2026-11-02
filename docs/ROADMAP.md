# Roadmap

## Phase 1: build (parallel sessions) ✅
- [x] **European prices + weather**: Energy-Charts day-ahead prices for DE-LU (hourly until
      2025-09-30, 15-minute after), Open-Meteo historical weather forecasts for every market's
      weather points. Retries, idempotent month partitions, mocked tests.
- [x] **Turkish prices**: EPİAŞ Transparency Platform market clearing price (PTF) in TRY and EUR,
      CAS ticket (TGT) authentication from environment variables, ticket renewal, mocked tests.
- [x] **Features, models, backtesting**: hourly modelling table without look-ahead; seasonal naive
      (same hour last week), naive (same hour yesterday), LEAR-style Lasso and LightGBM; rolling-origin
      daily backtest with MAE, RMSE and rMAE; fixture data generator for CI.
- [x] **Pipeline, serving, monitoring, CI**: daily pipeline (ingest → features → fit → forecast →
      store), DuckDB forecast store, FastAPI endpoints for forecasts and accuracy, monitoring that
      flags when a model has been worse than the seasonal naive for several days, Dockerfile,
      GitHub Actions.

## Phase 2: integration (local)
- [x] Germany: 2023–2025 prices and weather ingested; backtest over 2025 (see README); daily
      pipeline, monitor and API run end to end on real data.
- [x] Turkey: 2023–2025 prices and weather ingested from the live EPİAŞ API (rate limit of 60
      requests per minute found and handled); backtest over 2025 (see README).
- [x] Analysis: where each market is hard to forecast (negative prices, price cap hours, holidays,
      the 15-minute switch in October 2025); `analysis/error_analysis.py`.
- [x] README with findings and charts.
- [x] Move the bounds into the product: price limits per market in `config.py`, `bounds.py`,
      `*_bounded` models; `lear_bounded` is the pipeline's default.
- [ ] Confirm the remaining cap dates (July 2024, April 2025) against EPDK's decisions.

## Phase 3: deployment
- [ ] Scheduled daily run in the cloud, forecast API publicly reachable, accuracy dashboard.
