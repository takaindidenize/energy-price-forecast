"""Daily forecasting run: ingest -> features -> fit -> forecast D+1 -> store.

For issue day D, each market is run independently:

1. Ingest the price and weather months that can contain new data (those touched by D-1 .. D+1).
2. Load the hourly table and build features (`price_forecast.features`).
3. Fit every configured model on rows whose price was known at the issue time: delivery days up
   to and including D (day D's prices were published by the day-ahead auction on D-1).
4. Predict every delivery hour of local day D+1 (23, 24 or 25 rows around DST changes).
5. Upsert the predictions into `forecasts` with `issued_at` = D at `issue_hour_local`, in UTC.

A failure in one market is logged and recorded in `pipeline_runs`; the other markets still run.
A failing model is skipped and the market is marked `partial`.

Ingestion, features and models are imported lazily (see `default_deps`) so this module works, and
is testable, before or without those areas.

    python -m price_forecast.pipeline --market de_lu --date 2026-11-02 \
        --models seasonal_naive,median_bounded
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from price_forecast import config, store

log = logging.getLogger("price_forecast.pipeline")

DEFAULT_MODELS: tuple[str, ...] = tuple(
    m.strip()
    # The median of three models is the one the dashboard shows (README). Bounded LEAR keeps
    # running beside it as a live comparison; the seasonal naive is monitoring's baseline.
    for m in os.environ.get("PF_MODELS", "seasonal_naive,lear_bounded,median_bounded").split(",")
    if m.strip()
)


# --- time helpers (shared with monitor and api) -----------------------------------------------


def utc_now() -> datetime:
    """Current time as naive UTC."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None).to_pydatetime()


def issue_time_utc(market: str, issue_date: date) -> pd.Timestamp:
    """Naive UTC timestamp of the forecast issued on `issue_date` at the market's issue hour."""
    m = config.MARKETS[market]
    local = pd.Timestamp(issue_date).replace(hour=m.issue_hour_local).tz_localize(m.timezone)
    return local.tz_convert("UTC").tz_localize(None)


def delivery_day_bounds_utc(market: str, delivery_date: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) of local `delivery_date` as naive UTC timestamps (23, 24 or 25 hours long)."""
    tz = config.MARKETS[market].timezone
    start = pd.Timestamp(delivery_date).tz_localize(tz)
    end = pd.Timestamp(delivery_date + timedelta(days=1)).tz_localize(tz)
    return (
        start.tz_convert("UTC").tz_localize(None),
        end.tz_convert("UTC").tz_localize(None),
    )


def delivery_hours_utc(market: str, delivery_date: date) -> pd.DatetimeIndex:
    start, end = delivery_day_bounds_utc(market, delivery_date)
    return pd.date_range(start, end, freq="h", inclusive="left")


def local_dates(market: str, hours_utc: pd.Series) -> pd.Series:
    """Local delivery date of each naive-UTC hour."""
    tz = ZoneInfo(config.MARKETS[market].timezone)
    return store.to_utc_naive(hours_utc).dt.tz_localize("UTC").dt.tz_convert(tz).dt.date


def months_to_ingest(issue_date: date) -> list[str]:
    """'YYYY-MM' months that may hold new data for an issue on `issue_date`.

    Covers D-1 (late data around a month boundary), D (prices published yesterday) and D+1
    (weather forecasts for the delivery day). Any UTC instant of local D+1 falls in D or D+1.
    """
    days = (issue_date - timedelta(days=1), issue_date, issue_date + timedelta(days=1))
    return sorted({d.strftime("%Y-%m") for d in days})


# --- dependencies -----------------------------------------------------------------------------


@dataclass
class PipelineDeps:
    """The functions the pipeline calls into other areas. Tests pass fakes."""

    ingest_prices: Callable[[str, str], Path | None]  # (market, month)
    ingest_weather: Callable[[str, str], Path | None]  # (market, month)
    load_hourly: Callable[[str], pd.DataFrame]
    build_features: Callable[[pd.DataFrame, str], pd.DataFrame]
    make_model: Callable[[str], Any]  # name -> Forecaster


def _ingest_prices(market: str, month: str) -> Path | None:
    source = config.MARKETS[market].source
    if source == "energy_charts":
        module = importlib.import_module("price_forecast.ingestion.energy_charts")
        return module.ingest_month(month, market=market)
    if source == "epias":
        module = importlib.import_module("price_forecast.ingestion.epias")
        return module.ingest_month(month)
    raise ValueError(f"no price ingestion for source {source!r}")


def _ingest_weather(market: str, month: str) -> Path | None:
    module = importlib.import_module("price_forecast.ingestion.weather")
    return module.ingest_month(market, month)


def _load_hourly(market: str) -> pd.DataFrame:
    return importlib.import_module("price_forecast.features").load_hourly(market)


def _build_features(hourly: pd.DataFrame, market: str) -> pd.DataFrame:
    return importlib.import_module("price_forecast.features").build_features(hourly, market)


def make_model(name: str) -> Any:
    """Instantiate the forecaster `price_forecast.models.<name>`.

    The module must define a class with a no-argument constructor whose `name` attribute equals
    the module name, e.g. `class SeasonalNaive: name = "seasonal_naive"`. A `_bounded` suffix
    wraps the model so its forecasts stay within the market's price limits.
    """
    if not name.isidentifier() or name == "base":
        raise ValueError(f"invalid model name {name!r}")
    if name.endswith("_bounded"):
        from price_forecast.models.bounded import BoundedForecaster

        return BoundedForecaster(make_model(name.removesuffix("_bounded")))
    module = importlib.import_module(f"price_forecast.models.{name}")
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__ and getattr(obj, "name", None) == name:
            return obj()
    raise LookupError(f"price_forecast.models.{name} defines no class with name = {name!r}")


def default_deps() -> PipelineDeps:
    return PipelineDeps(
        ingest_prices=_ingest_prices,
        ingest_weather=_ingest_weather,
        load_hourly=_load_hourly,
        build_features=_build_features,
        make_model=make_model,
    )


# --- one market ---------------------------------------------------------------------------------


@dataclass
class MarketRun:
    market: str
    issue_date: date
    status: str = "failed"  # ok | partial | failed
    models_ok: list[str] = field(default_factory=list)
    models_failed: dict[str, str] = field(default_factory=dict)
    n_forecasts: int = 0
    error: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None

    @property
    def seconds(self) -> float:
        end = self.finished_at or utc_now()
        return (end - self.started_at).total_seconds()


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _with_delivery_day(hourly: pd.DataFrame, market: str, delivery_date: date) -> pd.DataFrame:
    """Append rows for D+1 hours missing from `hourly` (null price), so features cover them.

    `build_features` is expected to emit not-yet-delivered hours with a null target; this only
    adds the hours themselves when the hourly table stops at the last realised price.
    """
    hours = delivery_hours_utc(market, delivery_date)
    existing = (
        set(store.to_utc_naive(hourly["delivery_hour"])) if "delivery_hour" in hourly else set()
    )
    missing = [h for h in hours if h not in existing]
    if not missing:
        return hourly
    extra = pd.DataFrame({"delivery_hour": pd.DatetimeIndex(missing)})
    if "market" in hourly.columns:
        extra["market"] = market
    if "delivery_hour" in hourly.columns and hourly["delivery_hour"].dt.tz is not None:
        extra["delivery_hour"] = extra["delivery_hour"].dt.tz_localize("UTC")
    return pd.concat([hourly, extra], ignore_index=True)


def split_at_issue(
    features: pd.DataFrame, market: str, issue_date: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows known at the issue time, and the D+1 rows to predict.

    Prices are known through the end of local day D. Anything later, even if present in a
    historical re-run, is excluded from training.
    """
    delivery = issue_date + timedelta(days=1)
    hours = store.to_utc_naive(features["delivery_hour"])
    _, known_until = delivery_day_bounds_utc(market, issue_date)
    start, end = delivery_day_bounds_utc(market, delivery)
    train = features[(hours < known_until) & features["price_eur"].notna()]
    target = features[(hours >= start) & (hours < end)]
    return train, target


def run_market(
    market: str,
    issue_date: date,
    models: Sequence[str] = DEFAULT_MODELS,
    *,
    deps: PipelineDeps | None = None,
    warehouse: Path | str | None = None,
    ingest: bool = True,
) -> MarketRun:
    """Run the full daily pipeline for one market. Never raises; failures are in the result."""
    deps = deps or default_deps()
    run = MarketRun(market=market, issue_date=issue_date)
    try:
        if market not in config.MARKETS:
            raise KeyError(f"unknown market {market!r}")
        if not models:
            raise ValueError("no models configured")
        if ingest:
            for month in months_to_ingest(issue_date):
                for kind, fn in (("prices", deps.ingest_prices), ("weather", deps.ingest_weather)):
                    path = fn(market, month)
                    if path is None:
                        log.info("%s: no %s data for %s", market, kind, month)
                    else:
                        log.info("%s: ingested %s %s -> %s", market, kind, month, path)

        delivery = issue_date + timedelta(days=1)
        hourly = _with_delivery_day(deps.load_hourly(market), market, delivery)
        features = deps.build_features(hourly, market)
        train, target = split_at_issue(features, market, issue_date)
        if train.empty:
            raise RuntimeError(f"no training rows with a known price up to {issue_date}")
        if target.empty:
            raise RuntimeError(f"build_features produced no rows for delivery day {delivery}")
        expected = len(delivery_hours_utc(market, delivery))
        if len(target) != expected:
            log.warning(
                "%s: %d feature rows for %s, expected %d", market, len(target), delivery, expected
            )

        issued_at = issue_time_utc(market, issue_date)
        frames = []
        for name in models:
            try:
                model = deps.make_model(name)
                model.fit(train)
                pred = pd.Series(model.predict(target), index=target.index, dtype="float64")
                ok = np.isfinite(pred)
                if not ok.any():
                    raise RuntimeError("no finite predictions")
                if not ok.all():
                    log.warning(
                        "%s/%s: %d non-finite predictions dropped", market, name, (~ok).sum()
                    )
                frames.append(
                    pd.DataFrame(
                        {
                            "market": market,
                            "model": name,
                            "issued_at": issued_at,
                            "delivery_hour": store.to_utc_naive(target["delivery_hour"])[ok],
                            "price_eur_pred": pred[ok],
                        }
                    )
                )
                run.models_ok.append(name)
            except Exception as exc:  # one model must not take the others down
                log.exception("%s: model %s failed", market, name)
                run.models_failed[name] = _error_text(exc)

        if frames:
            with store.session(warehouse) as con:
                run.n_forecasts = store.write_forecasts(con, pd.concat(frames, ignore_index=True))
        if not run.models_ok:
            raise RuntimeError("every model failed")
        run.status = "partial" if run.models_failed else "ok"
    except Exception as exc:
        log.exception("%s: pipeline failed", market)
        run.status = "failed"
        run.error = _error_text(exc)
    run.finished_at = utc_now()
    _record_run(run, warehouse)
    return run


def _record_run(run: MarketRun, warehouse: Path | str | None) -> None:
    row = pd.DataFrame(
        [
            {
                "market": run.market,
                "issue_date": run.issue_date,
                "status": run.status,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "models_ok": ",".join(run.models_ok),
                "models_failed": ",".join(run.models_failed),
                "n_forecasts": run.n_forecasts,
                "error": run.error or "; ".join(f"{k}: {v}" for k, v in run.models_failed.items()),
            }
        ]
    )
    try:
        with store.session(warehouse) as con:
            store.upsert(con, "pipeline_runs", row)
    except Exception:
        log.exception("%s: could not record the pipeline run", run.market)


# --- all markets --------------------------------------------------------------------------------


def run(
    markets: Sequence[str],
    issue_date: date,
    models: Sequence[str] = DEFAULT_MODELS,
    **kwargs: Any,
) -> list[MarketRun]:
    """Run every market in turn; one market's failure does not stop the next."""
    results = [run_market(m, issue_date, models, **kwargs) for m in markets]
    log.info("%s", summary(results))
    return results


def summary(results: Sequence[MarketRun]) -> str:
    lines = [f"Pipeline summary ({len(results)} market(s)):"]
    for r in results:
        line = f"  {r.market:<6} {r.issue_date} -> {r.issue_date + timedelta(days=1)}  "
        line += f"{r.status.upper():<8} {r.n_forecasts:>4} forecasts  {r.seconds:6.1f}s"
        if r.models_ok:
            line += f"  ok: {','.join(r.models_ok)}"
        if r.models_failed:
            line += f"  failed: {', '.join(f'{k} ({v})' for k, v in r.models_failed.items())}"
        if r.error and r.status == "failed":
            line += f"  error: {r.error}"
        lines.append(line)
    return "\n".join(lines)


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--market",
        action="append",
        help="market code (repeat or comma-separate; default: all markets)",
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="issue day D, YYYY-MM-DD (default: today, UTC); forecasts are for D+1",
    )
    parser.add_argument(
        "--models",
        type=_parse_list,
        default=list(DEFAULT_MODELS),
        help=f"comma-separated model names (default: {','.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="run for the N issue days ending at --date, oldest first (backfill; default 1)",
    )
    parser.add_argument("--no-ingest", action="store_true", help="use the raw data already on disk")
    parser.add_argument("--warehouse", type=Path, default=None, help="DuckDB path override")
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be at least 1")

    markets = [
        m for value in (args.market or [",".join(config.MARKETS)]) for m in _parse_list(value)
    ]
    unknown = [m for m in markets if m not in config.MARKETS]
    if unknown:
        parser.error(f"unknown market(s) {unknown}; choose from {sorted(config.MARKETS)}")
    issue_date = args.date or utc_now().date()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    results = []
    ingested: set[str] = set()
    for offset in range(args.days - 1, -1, -1):
        day = issue_date - timedelta(days=offset)
        # Ingest each month once, when the first day that needs it comes up. A backfill must not
        # stop at the oldest day's months, or the newest days run on old data.
        months = set(months_to_ingest(day))
        ingest = not args.no_ingest and not months <= ingested
        ingested |= months
        results += run(markets, day, args.models, warehouse=args.warehouse, ingest=ingest)
    print(summary(results))
    return 0 if all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
