"""Run the 2025 backtest for both markets and save the predictions for `error_analysis.py`.

Needs the raw data for 2023–2025 (see README). Takes a few minutes; prints the summary tables.
Run from the repository root: `python analysis/run_backtests.py`.
"""

from __future__ import annotations

import warnings

from price_forecast import backtest, config, features
from price_forecast.models import MODEL_NAMES, make_model

# The four models plus LEAR within the market's price limits (the pipeline's default).
NAMES = (*MODEL_NAMES, "lear_bounded")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    for market in config.MARKETS:
        feats = features.build_features(features.load_hourly(market), market)
        preds = backtest.run(
            market,
            [make_model(name) for name in NAMES],
            "2025-01-01",
            "2025-12-31",
            features=feats,
            refit_every=7,
        )
        preds.to_parquet(config.DATA_DIR / f"backtest_{market}_2025.parquet")
        print(backtest.summarize(preds).round(3).to_string(), "\n")


if __name__ == "__main__":
    main()
