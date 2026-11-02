"""Forecast monitoring: realised accuracy per day, rolling MAE/rMAE and a degradation flag.

Once realised prices exist, stored forecasts are joined with them per market, model and local
delivery day. Each day's errors are compared with the weekly seasonal naive on the same rows: the
realised price of the same local hour seven days earlier (hours without that baseline are left
out of both). A model is **degraded** when its daily MAE has been worse than the seasonal naive's
on `degraded_after_days` consecutive delivery days, counted back from its latest evaluated day.

Results go to the warehouse (`realised_prices`, `daily_accuracy`, `model_status`) for the API.

    python -m price_forecast.monitor [--market de_lu] [--window-days 30] [--degraded-after 3]
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from price_forecast import config, store
from price_forecast.pipeline import local_dates, utc_now

log = logging.getLogger("price_forecast.monitor")

WINDOW_DAYS = int(os.environ.get("PF_MONITOR_WINDOW_DAYS", "30"))
DEGRADED_AFTER_DAYS = int(os.environ.get("PF_DEGRADED_AFTER_DAYS", "3"))


@dataclass(frozen=True)
class MonitorSettings:
    window_days: int = WINDOW_DAYS
    degraded_after_days: int = DEGRADED_AFTER_DAYS

    def __post_init__(self) -> None:
        if self.window_days < 1 or self.degraded_after_days < 1:
            raise ValueError("window_days and degraded_after_days must be >= 1")


def _load_realised(market: str) -> pd.DataFrame:
    return importlib.import_module("price_forecast.features").load_hourly(market)


def realised_prices(hourly: pd.DataFrame, market: str) -> pd.DataFrame:
    """`market, delivery_hour (naive UTC), price_eur` for every hour with a realised price."""
    out = pd.DataFrame(
        {
            "market": market,
            "delivery_hour": store.to_utc_naive(hourly["delivery_hour"]),
            "price_eur": pd.to_numeric(hourly["price_eur"], errors="coerce").astype("float64"),
        }
    )
    out = out[out["price_eur"].notna()]
    return out.groupby(["market", "delivery_hour"], as_index=False)["price_eur"].mean()


def seasonal_naive(realised: pd.DataFrame, market: str) -> pd.Series:
    """Weekly seasonal naive per row of `realised`: the price at the same local hour 7 days earlier.

    Keyed on local wall-clock time so the daily profile lines up across DST changes. Hours with no
    counterpart a week earlier (e.g. the week after a spring-forward day) get NaN.
    """
    tz = config.MARKETS[market].timezone
    local = realised["delivery_hour"].dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
    lookup = realised.assign(_local=local).groupby("_local")["price_eur"].mean()
    return pd.Series(
        (local - pd.Timedelta(days=7)).map(lookup).to_numpy(dtype="float64"), index=realised.index
    )


def latest_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """One forecast per market, model and delivery hour: the latest issued."""
    if forecasts.empty:
        return forecasts
    forecasts = forecasts.sort_values("issued_at")
    return forecasts.drop_duplicates(["market", "model", "delivery_hour"], keep="last")


def daily_accuracy(forecasts: pd.DataFrame, realised: pd.DataFrame, market: str) -> pd.DataFrame:
    """Error sums per model and local delivery day.

    Only rows where the forecast, the realised price and the seasonal naive baseline all exist.
    """
    columns = [
        "market",
        "model",
        "delivery_date_local",
        "n_hours",
        "abs_err_sum",
        "sq_err_sum",
        "snaive_abs_err_sum",
    ]
    if forecasts.empty or realised.empty:
        return pd.DataFrame(columns=columns)
    realised = realised.assign(snaive=seasonal_naive(realised, market))
    joined = latest_forecasts(forecasts).merge(
        realised[["delivery_hour", "price_eur", "snaive"]], on="delivery_hour", how="inner"
    )
    joined = joined.dropna(subset=["price_eur", "price_eur_pred", "snaive"])
    if joined.empty:
        return pd.DataFrame(columns=columns)
    err = joined["price_eur_pred"] - joined["price_eur"]
    joined = joined.assign(
        delivery_date_local=local_dates(market, joined["delivery_hour"]).to_numpy(),
        abs_err=err.abs(),
        sq_err=err**2,
        snaive_abs_err=(joined["snaive"] - joined["price_eur"]).abs(),
    )
    out = joined.groupby(["market", "model", "delivery_date_local"], as_index=False).agg(
        n_hours=("abs_err", "size"),
        abs_err_sum=("abs_err", "sum"),
        sq_err_sum=("sq_err", "sum"),
        snaive_abs_err_sum=("snaive_abs_err", "sum"),
    )
    return out[columns]


def worse_streak(days: pd.DataFrame) -> int:
    """Consecutive calendar days, back from the latest, on which the model's MAE was strictly
    worse than the seasonal naive's on the same rows. A day without an evaluation ends the streak.
    """
    days = days.sort_values("delivery_date_local", ascending=False)
    streak, expected = 0, None
    for row in days.itertuples(index=False):
        if expected is not None and row.delivery_date_local != expected:
            break
        if not row.abs_err_sum > row.snaive_abs_err_sum:
            break
        streak += 1
        expected = row.delivery_date_local - timedelta(days=1)
    return streak


def aggregate(days: pd.DataFrame) -> dict[str, float | int | None]:
    """MAE, RMSE and rMAE over a set of daily accuracy rows, weighted by hours."""
    n = int(days["n_hours"].sum()) if not days.empty else 0
    if n == 0:
        return {"n_days": 0, "n_hours": 0, "mae": None, "rmse": None, "rmae": None}
    abs_sum = float(days["abs_err_sum"].sum())
    snaive_sum = float(days["snaive_abs_err_sum"].sum())
    return {
        "n_days": int(days["delivery_date_local"].nunique()),
        "n_hours": n,
        "mae": abs_sum / n,
        "rmse": float(np.sqrt(days["sq_err_sum"].sum() / n)),
        "rmae": abs_sum / snaive_sum if snaive_sum > 0 else None,
    }


def model_status(
    accuracy: pd.DataFrame,
    market: str,
    models: Sequence[str],
    settings: MonitorSettings,
    as_of: date | None = None,
) -> pd.DataFrame:
    """One status row per model from its daily accuracy up to `as_of` (default: all)."""
    rows = []
    now = utc_now()
    for model in sorted(models):
        days = accuracy[(accuracy["market"] == market) & (accuracy["model"] == model)]
        if as_of is not None:
            days = days[days["delivery_date_local"] <= as_of]
        through = max(days["delivery_date_local"]) if not days.empty else None
        if through is not None:
            start = through - timedelta(days=settings.window_days - 1)
            window = days[days["delivery_date_local"] >= start]
        else:
            window = days
        stats = aggregate(window)
        streak = worse_streak(days) if not days.empty else 0
        degraded = streak >= settings.degraded_after_days
        rows.append(
            {
                "market": market,
                "model": model,
                "status": "no_data" if through is None else ("degraded" if degraded else "ok"),
                "degraded": degraded,
                "worse_streak_days": streak,
                "degraded_after_days": settings.degraded_after_days,
                "window_days": settings.window_days,
                "window_n_days": stats["n_days"],
                "window_mae": stats["mae"],
                "window_rmae": stats["rmae"],
                "evaluated_through": through,
                "updated_at": now,
            }
        )
    return pd.DataFrame(rows)


def run_market(
    market: str,
    settings: MonitorSettings | None = None,
    *,
    load_realised: Callable[[str], pd.DataFrame] = _load_realised,
    warehouse: Path | str | None = None,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Refresh realised prices, daily accuracy and model status for one market."""
    settings = settings or MonitorSettings()
    realised = realised_prices(load_realised(market), market)
    with store.session(warehouse) as con:
        store.upsert(con, "realised_prices", realised.assign(loaded_at=utc_now()))
        forecasts = store.read_forecasts(con, market)
        accuracy = daily_accuracy(forecasts, realised, market)
        store.upsert(con, "daily_accuracy", accuracy.assign(updated_at=utc_now()))
        all_accuracy = con.execute("SELECT * FROM daily_accuracy WHERE market = ?", [market]).df()
        if not all_accuracy.empty:
            all_accuracy["delivery_date_local"] = pd.to_datetime(
                all_accuracy["delivery_date_local"]
            ).dt.date
        models = sorted(set(forecasts["model"]))
        status = model_status(all_accuracy, market, models, settings, as_of)
        store.upsert(con, "model_status", status)
    _warn_if_limits_outdated(market)
    for row in status.itertuples(index=False):
        level = logging.WARNING if row.degraded else logging.INFO
        log.log(
            level,
            "%s/%s: %s (worse than seasonal naive %d day(s) in a row; %d-day rMAE %s)",
            market,
            row.model,
            row.status,
            row.worse_streak_days,
            settings.window_days,
            "n/a" if pd.isna(row.window_rmae) else f"{row.window_rmae:.3f}",
        )
    return status


def _warn_if_limits_outdated(market: str) -> None:
    """Warn when realised prices break the configured price limits (a stale cap schedule)."""
    from price_forecast import bounds

    try:
        days = bounds.outdated_limit_days(market)
    except Exception:  # no raw data yet, or unreadable: monitoring itself must not fail
        log.debug("%s: could not check price limits", market, exc_info=True)
        return
    if days:
        log.warning(
            "%s: realised prices broke the configured price limits on %d day(s), latest %s; "
            "update `limits` in config.py",
            market,
            len(days),
            days[-1],
        )


def run(markets: Sequence[str], settings: MonitorSettings | None = None, **kwargs) -> bool:
    """Monitor every market; one market's failure does not stop the next. True if all succeeded."""
    ok = True
    for market in markets:
        try:
            run_market(market, settings, **kwargs)
        except Exception:
            log.exception("%s: monitoring failed", market)
            ok = False
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--market", action="append", choices=sorted(config.MARKETS))
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    parser.add_argument("--degraded-after", type=int, default=DEGRADED_AFTER_DAYS)
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument("--warehouse", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = MonitorSettings(args.window_days, args.degraded_after)
    ok = run(
        args.market or list(config.MARKETS), settings, warehouse=args.warehouse, as_of=args.as_of
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
