"""Forecast API (FastAPI). Read-only over the DuckDB warehouse written by pipeline and monitor.

    uvicorn price_forecast.api:app            # docs at /docs, schema at /openapi.json

Times are returned in UTC (`delivery_hour_utc`) and in the market's local time
(`delivery_hour_local`, ISO 8601 with offset). A local delivery day has 23, 24 or 25 hours.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from pydantic import BaseModel

from price_forecast import config, store
from price_forecast.monitor import aggregate
from price_forecast.pipeline import delivery_day_bounds_utc

MODEL_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"


# --- response models ----------------------------------------------------------------------------


class Health(BaseModel):
    status: str
    warehouse_exists: bool


class WeatherPointOut(BaseModel):
    name: str
    lat: float
    lon: float


class MarketOut(BaseModel):
    code: str
    name: str
    timezone: str
    currency: str
    issue_hour_local: int
    weather_points: list[WeatherPointOut]


class HourlyForecast(BaseModel):
    delivery_hour_utc: datetime
    delivery_hour_local: datetime
    hour_local: int
    price_eur_pred: float
    price_eur_actual: float | None = None


class ModelForecast(BaseModel):
    model: str
    issued_at: datetime
    n_hours: int
    values: list[HourlyForecast]


class ForecastOut(BaseModel):
    market: str
    delivery_date: date
    timezone: str
    expected_hours: int
    forecasts: list[ModelForecast]


class DailyAccuracy(BaseModel):
    delivery_date: date
    n_hours: int
    mae: float
    rmae: float | None


class ModelAccuracy(BaseModel):
    model: str
    n_days: int
    n_hours: int
    mae: float | None
    rmse: float | None
    rmae: float | None
    daily: list[DailyAccuracy]


class AccuracyOut(BaseModel):
    market: str
    days: int
    date_from: date | None
    date_to: date | None
    models: list[ModelAccuracy]


class ModelStatusOut(BaseModel):
    model: str
    status: str
    degraded: bool
    worse_streak_days: int
    degraded_after_days: int
    window_days: int
    window_n_days: int
    window_mae: float | None
    window_rmae: float | None
    evaluated_through: date | None
    updated_at: datetime


class PipelineRunOut(BaseModel):
    issue_date: date
    status: str
    finished_at: datetime
    models_ok: list[str]
    models_failed: list[str]
    n_forecasts: int
    error: str | None


class MarketStatus(BaseModel):
    market: str
    status: str
    last_run: PipelineRunOut | None
    models: list[ModelStatusOut]


class StatusOut(BaseModel):
    status: str
    markets: list[MarketStatus]


# --- helpers --------------------------------------------------------------------------------------


def _none(value):
    """pandas missing values (NaN, NaT, None) as None for JSON."""
    return None if value is None or pd.isna(value) else value


def _utc(value) -> datetime:
    return pd.Timestamp(value).tz_localize("UTC").to_pydatetime()


def _date(value) -> date | None:
    return None if value is None or pd.isna(value) else pd.Timestamp(value).date()


def _split(value: str | None) -> list[str]:
    return [v for v in (value or "").split(",") if v]


def _market(market: str) -> config.Market:
    try:
        return config.MARKETS[market]
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"unknown market {market!r}; see GET /markets"
        ) from None


MarketCode = Annotated[str, PathParam(description="market code, see GET /markets")]


def warehouse_path(request: Request) -> Path:
    configured = request.app.state.warehouse
    return Path(configured) if configured is not None else config.WAREHOUSE_PATH


def get_db(request: Request) -> Iterator[duckdb.DuckDBPyConnection]:
    """A read-only connection per request; the pipeline and monitor are the only writers."""
    try:
        con = store.connect_readonly(warehouse_path(request))
    except duckdb.IOException as exc:  # e.g. the pipeline holds the write lock
        raise HTTPException(status_code=503, detail="warehouse is busy, retry shortly") from exc
    try:
        yield con
    finally:
        con.close()


Db = Annotated[duckdb.DuckDBPyConnection, Depends(get_db)]


# --- app ------------------------------------------------------------------------------------------


def create_app(warehouse: Path | str | None = None) -> FastAPI:
    """Build the app. `warehouse` defaults to `config.WAREHOUSE_PATH`, resolved per request."""

    app = FastAPI(
        title="Energy price forecast",
        version="0.1.0",
        description="Day-ahead electricity price forecasts (EUR/MWh) for Germany-Luxembourg and "
        "Turkey, with realised accuracy against the weekly seasonal naive.",
    )
    app.state.warehouse = warehouse

    @app.get("/health", response_model=Health, tags=["meta"])
    def health(request: Request) -> Health:
        return Health(status="ok", warehouse_exists=warehouse_path(request).exists())

    @app.get("/markets", response_model=list[MarketOut], tags=["meta"])
    def markets() -> list[MarketOut]:
        return [
            MarketOut(
                code=m.code,
                name=m.name,
                timezone=m.timezone,
                currency=m.currency,
                issue_hour_local=m.issue_hour_local,
                weather_points=[
                    WeatherPointOut(name=p.name, lat=p.lat, lon=p.lon) for p in m.weather_points
                ],
            )
            for m in config.MARKETS.values()
        ]

    @app.get("/forecast/{market}", response_model=ForecastOut, tags=["forecasts"])
    def forecast(
        market: MarketCode,
        db: Db,
        date_: Annotated[
            date | None,
            Query(
                alias="date",
                description="local delivery date YYYY-MM-DD (default: latest forecast day)",
            ),
        ] = None,
        model: Annotated[
            str | None, Query(pattern=MODEL_PATTERN, description="one model (default: all)")
        ] = None,
    ) -> ForecastOut:
        m = _market(market)
        tz = ZoneInfo(m.timezone)
        model_filter = "AND model = ?" if model else ""
        params: list = [market] + ([model] if model else [])
        if date_ is None:
            latest = db.execute(
                f"SELECT max(delivery_hour) FROM forecasts WHERE market = ? {model_filter}", params
            ).fetchone()[0]
            if latest is None:
                raise HTTPException(status_code=404, detail=f"no forecasts for {market}")
            date_ = pd.Timestamp(latest).tz_localize("UTC").tz_convert(tz).date()
        start, end = delivery_day_bounds_utc(market, date_)
        rows = db.execute(
            f"""
            WITH day AS (
                SELECT * FROM forecasts
                WHERE market = ? {model_filter} AND delivery_hour >= ? AND delivery_hour < ?
            ),
            latest AS (SELECT model, max(issued_at) AS issued_at FROM day GROUP BY model)
            SELECT d.model, d.issued_at, d.delivery_hour, d.price_eur_pred, r.price_eur
            FROM day d
            JOIN latest USING (model, issued_at)
            LEFT JOIN realised_prices r ON r.market = d.market AND r.delivery_hour = d.delivery_hour
            ORDER BY d.model, d.delivery_hour
            """,
            params + [start.to_pydatetime(), end.to_pydatetime()],
        ).df()
        if rows.empty:
            what = f"model {model!r}" if model else "any model"
            raise HTTPException(
                status_code=404, detail=f"no forecast for {market} on {date_} from {what}"
            )
        out = []
        for name, group in rows.groupby("model", sort=True):
            values = []
            for r in group.itertuples(index=False):
                utc = _utc(r.delivery_hour)
                local = utc.astimezone(tz)
                values.append(
                    HourlyForecast(
                        delivery_hour_utc=utc,
                        delivery_hour_local=local,
                        hour_local=local.hour,
                        price_eur_pred=r.price_eur_pred,
                        price_eur_actual=_none(r.price_eur),
                    )
                )
            out.append(
                ModelForecast(
                    model=name,
                    issued_at=_utc(group["issued_at"].iloc[0]),
                    n_hours=len(values),
                    values=values,
                )
            )
        return ForecastOut(
            market=market,
            delivery_date=date_,
            timezone=m.timezone,
            expected_hours=int((end - start) / pd.Timedelta(hours=1)),
            forecasts=out,
        )

    @app.get("/accuracy/{market}", response_model=AccuracyOut, tags=["monitoring"])
    def accuracy(
        market: MarketCode,
        db: Db,
        days: Annotated[int, Query(ge=1, le=366, description="window length in days")] = 30,
    ) -> AccuracyOut:
        _market(market)
        frame = db.execute(
            "SELECT * FROM daily_accuracy WHERE market = ? ORDER BY model, delivery_date_local",
            [market],
        ).df()
        if frame.empty:
            raise HTTPException(
                status_code=404,
                detail=f"no evaluated forecasts for {market} yet; run price_forecast.monitor",
            )
        frame["delivery_date_local"] = pd.to_datetime(frame["delivery_date_local"]).dt.date
        date_to = max(frame["delivery_date_local"])
        date_from = date_to - timedelta(days=days - 1)
        frame = frame[frame["delivery_date_local"] >= date_from]
        models = []
        for name, group in frame.groupby("model", sort=True):
            stats = aggregate(group)
            daily = [
                DailyAccuracy(
                    delivery_date=r.delivery_date_local,
                    n_hours=r.n_hours,
                    mae=r.abs_err_sum / r.n_hours,
                    rmae=r.abs_err_sum / r.snaive_abs_err_sum if r.snaive_abs_err_sum > 0 else None,
                )
                for r in group.itertuples(index=False)
            ]
            models.append(ModelAccuracy(model=name, daily=daily, **stats))
        return AccuracyOut(
            market=market, days=days, date_from=date_from, date_to=date_to, models=models
        )

    @app.get("/status", response_model=StatusOut, tags=["monitoring"])
    def status(db: Db) -> StatusOut:
        statuses = db.execute("SELECT * FROM model_status ORDER BY market, model").df()
        runs = db.execute(
            """
            SELECT * FROM pipeline_runs
            QUALIFY row_number() OVER (PARTITION BY market ORDER BY issue_date DESC) = 1
            """
        ).df()
        out = []
        for code in config.MARKETS:
            run_rows = runs[runs["market"] == code]
            last_run = None
            if not run_rows.empty:
                r = run_rows.iloc[0]
                last_run = PipelineRunOut(
                    issue_date=_date(r["issue_date"]),
                    status=r["status"],
                    finished_at=_utc(r["finished_at"]),
                    models_ok=_split(r["models_ok"]),
                    models_failed=_split(r["models_failed"]),
                    n_forecasts=int(r["n_forecasts"]),
                    error=_none(r["error"]) or None,
                )
            models = [
                ModelStatusOut(
                    model=r.model,
                    status=r.status,
                    degraded=bool(r.degraded),
                    worse_streak_days=int(r.worse_streak_days),
                    degraded_after_days=int(r.degraded_after_days),
                    window_days=int(r.window_days),
                    window_n_days=int(r.window_n_days),
                    window_mae=_none(r.window_mae),
                    window_rmae=_none(r.window_rmae),
                    evaluated_through=_date(r.evaluated_through),
                    updated_at=_utc(r.updated_at),
                )
                for r in statuses[statuses["market"] == code].itertuples(index=False)
            ]
            if last_run is not None and last_run.status == "failed":
                market_status = "failed"
            elif any(s.degraded for s in models) or (last_run and last_run.status == "partial"):
                market_status = "degraded"
            elif last_run is None and not models:
                market_status = "no_data"
            else:
                market_status = "ok"
            out.append(
                MarketStatus(market=code, status=market_status, last_run=last_run, models=models)
            )
        overall = "ok"
        if any(m.status in ("failed", "degraded") for m in out):
            overall = "degraded"
        elif all(m.status == "no_data" for m in out):
            overall = "no_data"
        return StatusOut(status=overall, markets=out)

    return app


app = create_app()
