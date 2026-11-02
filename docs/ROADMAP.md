# Roadmap

## Phase 1: build (parallel sessions)
- [ ] **European prices + weather**: Energy-Charts day-ahead prices for DE-LU (hourly until
      2025-09-30, 15-minute after), Open-Meteo historical weather forecasts for every market's
      weather points. Retries, idempotent month partitions, mocked tests.
- [ ] **Turkish prices**: EPİAŞ Transparency Platform market clearing price (PTF) in TRY and EUR,
      CAS ticket (TGT) authentication from environment variables, ticket renewal, mocked tests.
- [ ] **Features, models, backtesting**: hourly modelling table without look-ahead; seasonal naive
      (same hour last week), naive (same hour yesterday), LEAR-style Lasso and LightGBM; rolling-origin
      daily backtest with MAE, RMSE and rMAE; fixture data generator for CI.
- [ ] **Pipeline, serving, monitoring, CI**: daily pipeline (ingest → features → fit → forecast →
      store), DuckDB forecast store, FastAPI endpoints for forecasts and accuracy, monitoring that
      flags when a model has been worse than the seasonal naive for several days, Dockerfile,
      GitHub Actions.

## Phase 2: integration (local)
- [ ] Real-data run for both markets, backtest over the last 12 months.
- [ ] Analysis: where each market is hard to forecast (negative prices, price cap hours, holidays).
- [ ] README with findings, charts and design decisions.

## Phase 3: deployment
- [ ] Scheduled daily run in the cloud, forecast API publicly reachable, accuracy dashboard.
