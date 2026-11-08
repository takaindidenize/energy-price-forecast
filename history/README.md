# Forecast history

Written by the daily job of https://github.com/takaindidenize/energy-price-forecast
(`python -m price_forecast.history`). One CSV per warehouse table, sorted by key, timestamps in
naive UTC. `forecasts` holds every live forecast as it was made; `pipeline_runs` holds when each
daily run finished. The job restores these when its cache is empty.
