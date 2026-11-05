"""Do combined forecasts or a recent-error correction beat bounded LEAR?

The decision rule was fixed before the results were seen: a method replaces the default only if it
is better in all four market-years (two markets, 2025 and 2026), reliably (paired t above 2 on
daily errors). 2026 had been looked at before this test, so it is not fully independent; the live
job, which keeps running both models, is the real test.

Writes `docs/analysis/ensemble.md`. Needs the raw data to 4 November 2026; takes about ten minutes
on one core.

    python analysis/ensemble.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from price_forecast import backtest, bounds, config, features
from price_forecast.models.bounded import BoundedForecaster
from price_forecast.models.lear import LEARForecaster
from price_forecast.models.lightgbm import LightGBMForecaster
from price_forecast.models.median import SHORT_WINDOW_DAYS

OUT_MD = config.PROJECT_ROOT / "docs" / "analysis" / "ensemble.md"
PERIODS = (("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-11-04"))
# The issue day's prices are known at issue time, so errors through T-1 can correct day T.
CORRECTION_DAYS, CORRECTION_WEIGHT = 7, 0.5
BASE = "lear_all_bounded"
TRIO = [BASE, "lear_short_bounded", "lightgbm_bounded"]
LABELS = {
    BASE: "LEAR (bounded)",
    "lear_short_bounded": f"LEAR, last {SHORT_WINDOW_DAYS} days",
    "lightgbm_bounded": "LightGBM",
    "pair_mean": "Mean: LEAR, LightGBM",
    "trio_mean": "Mean of the three",
    "trio_median": "Median of the three",
    "lear_corrected": "LEAR + correction",
    "mean_corrected": "Mean of the three + correction",
    "median_corrected": "Median of the three + correction",
}


class NamedLear(LEARForecaster):
    def __init__(self, name: str, days: int | None) -> None:
        super().__init__(calibration_days=days)
        self.name = name


def wide_predictions(market: str, feats: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    models = [
        BoundedForecaster(NamedLear("lear_all", None)),
        BoundedForecaster(NamedLear("lear_short", SHORT_WINDOW_DAYS)),
        BoundedForecaster(LightGBMForecaster()),
    ]
    # The correction needs errors from the days before the period starts.
    warm = (pd.Timestamp(start) - pd.Timedelta(days=CORRECTION_DAYS + 1)).date().isoformat()
    preds = backtest.run(market, models, warm, end, features=feats, refit_every=7)
    wide = preds.pivot_table(index="delivery_hour", columns="model", values="price_eur_pred")
    info = preds.drop_duplicates("delivery_hour").set_index("delivery_hour")
    wide["price"] = info["price_eur"]
    wide["day"] = pd.to_datetime(info["delivery_date_local"])
    return wide.sort_index()


def corrected(wide: pd.DataFrame, column: str, market: str, rates) -> pd.Series:
    """Subtract half of the mean signed error of the seven days before each delivery day."""
    daily = (wide[column] - wide["price"]).groupby(wide["day"]).mean()
    bias = daily.rolling(CORRECTION_DAYS).mean().shift(1)
    value = wide[column] - CORRECTION_WEIGHT * wide["day"].map(bias).fillna(0.0)
    return bounds.clip(market, wide["day"].dt.date, value, rates)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    mae: dict[str, dict[str, float]] = {m: {} for m in LABELS}
    ts: dict[str, dict[str, float]] = {m: {} for m in LABELS}
    for market in ("de_lu", "tr"):
        feats = features.build_features(features.load_hourly(market), market)
        rates = bounds.eur_per_local(market)
        for label, start, end in PERIODS:
            w = wide_predictions(market, feats, start, end)
            w["pair_mean"] = w[[BASE, "lightgbm_bounded"]].mean(axis=1)
            w["trio_mean"] = w[TRIO].mean(axis=1)
            w["trio_median"] = w[TRIO].median(axis=1)
            w["lear_corrected"] = corrected(w, BASE, market, rates)
            w["mean_corrected"] = corrected(w, "trio_mean", market, rates)
            w["median_corrected"] = corrected(w, "trio_median", market, rates)
            w = w[w["day"] >= start]
            key = f"{config.MARKETS[market].name} {label}"
            base_err = (w[BASE] - w["price"]).abs()
            for method in LABELS:
                err = (w[method] - w["price"]).abs()
                diff = (base_err - err).groupby(w["day"]).mean()
                mae[method][key] = err.mean()
                ts[method][key] = (
                    diff.mean() / (diff.std() / np.sqrt(len(diff))) if diff.std() > 0 else 0.0
                )

    keys = list(mae[BASE])
    rows = []
    for method, label in LABELS.items():
        passed = method != BASE and all(ts[method][k] > 2 for k in keys)
        cells = [
            f"{mae[method][k]:.2f}"
            if method == BASE
            else f"{mae[method][k]:.2f} (t = {ts[method][k]:.1f})"
            for k in keys
        ]
        verdict = "" if method == BASE else ("**passes**" if passed else "fails")
        rows.append("| " + " | ".join([label, *cells, verdict]) + " |")
    header = "| Method | " + " | ".join(keys) + " | Rule |"
    align = "|---|" + "---:|" * len(keys) + "---|"
    OUT_MD.write_text(
        f"""# Combining forecasts

Generated by `analysis/ensemble.py`. MAE in EUR/MWh, all forecasts within the market's price
limits. The rule, fixed before the results: better than LEAR in all four columns, reliably (paired
t above 2 on daily errors).

{header}
{align}
{chr(10).join(rows)}

The correction subtracts half of the mean signed error of the previous seven days.
""",
        encoding="utf-8",
    )
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
