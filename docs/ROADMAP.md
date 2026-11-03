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
- [x] Confirm cap dates: 3,400 TRY from 5 April 2025 (EPDK decision 13423) and 4,500 TRY from
      4 April 2026 (decision 14459). Still unconfirmed: 3,000 TRY from July 2024 (first cleared).

## Phase 3: deployment
Plan: free hosting with GitHub Actions and GitHub Pages instead of a paid cloud account.
- [x] Data up to today for both markets; the weather ingestion cuts the current month at a
      7-day horizon (the API rejects end dates more than ~15 days ahead) and so gets tomorrow.
- [x] `src/price_forecast/report.py`: static `site/index.html` per market with tomorrow's
      forecast (lear_bounded), the last 14 days of forecast vs actual, and 30-day MAE vs the
      seasonal naive. Inline SVG, no JS dependencies, light and dark.
- [x] `.github/workflows/daily-pipeline.yml`: restore `data/` with actions/cache (full ingest when
      missing), run pipeline + monitor + report, deploy `site/` to GitHub Pages, save the cache.
      Needs from the owner: repo secrets `EPIAS_USERNAME` / `EPIAS_PASSWORD`, Pages enabled
      (public repo, or a paid plan for a private one), then uncomment the schedule.
- [x] Live since 3 November 2026: secrets added, first full run green, daily schedule on
      (07:30 UTC). Pages deploys once the repository is public.
- [ ] Investigate the German under-forecast since 28 October 2026 (prices ~250 EUR/MWh, model ~50
      below): a calibration window instead of an expanding one, or gas prices as a feature.
