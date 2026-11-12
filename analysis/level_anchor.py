"""Does anchoring the forecast to the issue day's price level beat the median of three?

Why: since late October 2026 German prices sit far above what the weather explains, the live
median forecast is about 30 EUR/MWh too low, and yesterday's price (MAE 35.4) beat it (40.5). The
models see the issue day's prices, but they learn price *levels* over two to three years, so a new
level is only partly carried over. Gas and carbon prices, the usual level inputs, have no free
source whose licence allows publishing forecasts made from them, so this test uses no new data.

Anchoring: each model learns the delivery day's prices as deviations from the issue day's daily
mean L (`price_d1_mean`), with its price inputs shifted the same way, and L itself as an input; the
forecast is the predicted deviation plus L. The level then carries over one to one unless the
model learns otherwise, while the weather and calendar still shape the day.

Fixed before the results were seen:
- Candidates: bounded LEAR anchored, and the median of the three members all anchored (LEAR on all
  history, LEAR on 180 days, LightGBM).
- Rule, as in `analysis/ensemble.py`: a candidate replaces the median only if it is better in all
  four market-years (two markets, 2025 and 2026), reliably (paired t above 2 on daily errors).
- The yesterday's-price benchmark and the German live window (28 October onward) are shown for
  context only.
Most of 2026, and the German window above all, had been looked at before this test, so they are
not independent; 2025 is the cleaner check.

Writes `docs/analysis/level_anchor.md`. Takes about twenty minutes on one core.

    python analysis/level_anchor.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from price_forecast import backtest, config, features
from price_forecast.features import TARGET_COLUMN
from price_forecast.models.bounded import BoundedForecaster
from price_forecast.models.lear import LEARForecaster
from price_forecast.models.lightgbm import LightGBMForecaster
from price_forecast.models.median import SHORT_WINDOW_DAYS

OUT_MD = config.PROJECT_ROOT / "docs" / "analysis" / "level_anchor.md"
PERIODS = (("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-11-13"))
LIVE_WINDOW = ("2026-10-28", "2026-11-13")
SHIFTED = (TARGET_COLUMN, "price_d1_same_hour", "price_d7_same_hour")


def level(df: pd.DataFrame) -> pd.Series:
    """The issue day's mean price; if missing, the mean of the row's issue-day profile."""
    profile_mean = df.groupby("delivery_date_local")["price_d1_same_hour"].transform("mean")
    return df["price_d1_mean"].fillna(profile_mean).fillna(0.0)


class Anchored:
    """Fits `inner` on prices minus the issue day's level and adds the level back."""

    def __init__(self, inner, name: str) -> None:
        self.inner, self.name = inner, name

    @staticmethod
    def _shift(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        lv = level(df)
        for col in SHIFTED:
            if col in out:
                out[col] = out[col] - lv
        out["price_d1_mean"] = lv  # the level stays an input
        return out

    def fit(self, train: pd.DataFrame) -> Anchored:
        self.inner.fit(self._shift(train))
        return self

    def predict(self, feats: pd.DataFrame) -> pd.Series:
        return (self.inner.predict(self._shift(feats)) + level(feats)).rename(self.name)


class LevelLear(LEARForecaster):
    """LEAR with the issue day's level as one more input (LEAR's own design leaves it out)."""

    def __init__(self, name: str, days: int | None) -> None:
        super().__init__(calibration_days=days)
        self.name = name

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        return super()._design(df).assign(level=df["price_d1_mean"].astype(float))


class NamedLear(LEARForecaster):
    def __init__(self, name: str, days: int | None) -> None:
        super().__init__(calibration_days=days)
        self.name = name


class NamedLgbm(LightGBMForecaster):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


def models() -> list:
    return [
        BoundedForecaster(NamedLear("lear_all", None)),
        BoundedForecaster(NamedLear("lear_short", SHORT_WINDOW_DAYS)),
        BoundedForecaster(NamedLgbm("lightgbm")),
        BoundedForecaster(Anchored(LevelLear("lear_all_in", None), "a_lear_all")),
        BoundedForecaster(Anchored(LevelLear("lear_short_in", SHORT_WINDOW_DAYS), "a_lear_short")),
        BoundedForecaster(Anchored(NamedLgbm("lightgbm_in"), "a_lightgbm")),
    ]


LABELS = {
    "median": "Median of three (live default)",
    "a_lear_all_bounded": "LEAR, anchored",
    "a_median": "Median of three, anchored",
    "naive_d1": "Yesterday's price (context)",
}
CANDIDATES = ("a_lear_all_bounded", "a_median")


def scores(w: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    base = (w["median"] - w["price"]).abs()
    mae, t = {}, {}
    for method in LABELS:
        err = (w[method] - w["price"]).abs()
        diff = (base - err).groupby(w["day"]).mean()
        mae[method] = err.mean()
        t[method] = diff.mean() / (diff.std() / np.sqrt(len(diff))) if diff.std() > 0 else 0.0
    return mae, t


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    mae: dict[str, dict[str, float]] = {m: {} for m in LABELS}
    ts: dict[str, dict[str, float]] = {m: {} for m in LABELS}
    window: dict[str, float] = {}
    bias: dict[str, float] = {}
    for market in ("de_lu", "tr"):
        feats = features.build_features(features.load_hourly(market), market)
        naive = feats.set_index("delivery_hour")["price_d1_same_hour"]
        for label, start, end in PERIODS:
            preds = backtest.run(market, models(), start, end, features=feats, refit_every=7)
            w = preds.pivot_table(index="delivery_hour", columns="model", values="price_eur_pred")
            info = preds.drop_duplicates("delivery_hour").set_index("delivery_hour")
            w["price"] = info["price_eur"]
            w["day"] = pd.to_datetime(info["delivery_date_local"])
            w = w.dropna(subset=["price"])
            w["median"] = w[["lear_all_bounded", "lear_short_bounded", "lightgbm_bounded"]].median(
                axis=1
            )
            w["a_median"] = w[
                ["a_lear_all_bounded", "a_lear_short_bounded", "a_lightgbm_bounded"]
            ].median(axis=1)
            w["naive_d1"] = naive.reindex(w.index).fillna(w["median"])
            key = f"{config.MARKETS[market].name} {label}"
            m, t = scores(w)
            for method in LABELS:
                mae[method][key], ts[method][key] = m[method], t[method]
            print(key, {k: round(v, 2) for k, v in m.items()}, flush=True)
            if market == "de_lu" and label == "2026":
                live = w[w["day"] >= LIVE_WINDOW[0]]
                for method in LABELS:
                    window[method] = (live[method] - live["price"]).abs().mean()
                    bias[method] = (live[method] - live["price"]).mean()

    keys = list(mae["median"])
    rows = []
    for method, label in LABELS.items():
        if method == "median":
            cells, verdict = [f"{mae[method][k]:.2f}" for k in keys], ""
        else:
            cells = [f"{mae[method][k]:.2f} (t = {ts[method][k]:.1f})" for k in keys]
            passed = all(ts[method][k] > 2 for k in keys)
            verdict = ("**passes**" if passed else "fails") if method in CANDIDATES else "context"
        rows.append("| " + " | ".join([label, *cells, verdict]) + " |")
    header = "| Method | " + " | ".join(keys) + " | Rule |"
    align = "|---|" + "---:|" * len(keys) + "---|"
    live_rows = "\n".join(f"| {LABELS[m]} | {window[m]:.1f} | {bias[m]:+.1f} |" for m in LABELS)
    OUT_MD.write_text(
        f"""# Anchoring the forecast to the issue day's level

Generated by `analysis/level_anchor.py`. MAE in EUR/MWh, all model forecasts within the market's
price limits. Anchored models learn each delivery day as deviations from the issue day's mean price
and add that mean back. The rule, fixed before the results: better than the median of three in all
four columns, reliably (paired t above 2 on daily errors). 2026 runs to {PERIODS[1][2]}.

{header}
{align}
{chr(10).join(rows)}

Germany since {LIVE_WINDOW[0]}, the period that prompted the test (backtest, not the live
forecasts; looked at before, so context only):

| Method | MAE | Mean error |
|---|---:|---:|
{live_rows}
""",
        encoding="utf-8",
    )
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
