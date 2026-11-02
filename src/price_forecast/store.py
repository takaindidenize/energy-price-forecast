"""DuckDB warehouse: forecasts, backtest results, realised prices, accuracy and model status.

`forecasts` and `backtest_results` follow the contract in CLAUDE.md. The other tables are owned by
the pipeline/monitoring area and are only read by `api.py`:

- `realised_prices`: hourly realised day-ahead prices in EUR, copied by the monitor so the API can
  show actuals without reading raw parquet.
- `daily_accuracy`: per market, model and local delivery day, error sums against realised prices
  and against the weekly seasonal naive on the same rows. Sums (not means) so that any window can
  be aggregated exactly.
- `model_status`: latest monitoring verdict per market and model (ok / degraded / no_data).
- `pipeline_runs`: outcome of the latest daily run per market and issue day.

All writes are upserts on the table's key, so re-running anything is idempotent. All timestamp
columns hold naive UTC.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

from price_forecast import config

SCHEMA: dict[str, str] = {
    "forecasts": """
        market VARCHAR NOT NULL,
        model VARCHAR NOT NULL,
        issued_at TIMESTAMP NOT NULL,
        delivery_hour TIMESTAMP NOT NULL,
        price_eur_pred DOUBLE,
        PRIMARY KEY (market, model, issued_at, delivery_hour)
    """,
    "backtest_results": """
        market VARCHAR NOT NULL,
        model VARCHAR NOT NULL,
        delivery_hour TIMESTAMP NOT NULL,
        price_eur DOUBLE,
        price_eur_pred DOUBLE,
        run_id VARCHAR NOT NULL,
        PRIMARY KEY (run_id, market, model, delivery_hour)
    """,
    "realised_prices": """
        market VARCHAR NOT NULL,
        delivery_hour TIMESTAMP NOT NULL,
        price_eur DOUBLE,
        loaded_at TIMESTAMP NOT NULL,
        PRIMARY KEY (market, delivery_hour)
    """,
    "daily_accuracy": """
        market VARCHAR NOT NULL,
        model VARCHAR NOT NULL,
        delivery_date_local DATE NOT NULL,
        n_hours INTEGER NOT NULL,
        abs_err_sum DOUBLE NOT NULL,
        sq_err_sum DOUBLE NOT NULL,
        snaive_abs_err_sum DOUBLE NOT NULL,
        updated_at TIMESTAMP NOT NULL,
        PRIMARY KEY (market, model, delivery_date_local)
    """,
    "model_status": """
        market VARCHAR NOT NULL,
        model VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        degraded BOOLEAN NOT NULL,
        worse_streak_days INTEGER NOT NULL,
        degraded_after_days INTEGER NOT NULL,
        window_days INTEGER NOT NULL,
        window_n_days INTEGER NOT NULL,
        window_mae DOUBLE,
        window_rmae DOUBLE,
        evaluated_through DATE,
        updated_at TIMESTAMP NOT NULL,
        PRIMARY KEY (market, model)
    """,
    "pipeline_runs": """
        market VARCHAR NOT NULL,
        issue_date DATE NOT NULL,
        status VARCHAR NOT NULL,
        started_at TIMESTAMP NOT NULL,
        finished_at TIMESTAMP NOT NULL,
        models_ok VARCHAR NOT NULL,
        models_failed VARCHAR NOT NULL,
        n_forecasts INTEGER NOT NULL,
        error VARCHAR,
        PRIMARY KEY (market, issue_date)
    """,
}

KEYS: dict[str, tuple[str, ...]] = {
    "forecasts": ("market", "model", "issued_at", "delivery_hour"),
    "backtest_results": ("run_id", "market", "model", "delivery_hour"),
    "realised_prices": ("market", "delivery_hour"),
    "daily_accuracy": ("market", "model", "delivery_date_local"),
    "model_status": ("market", "model"),
    "pipeline_runs": ("market", "issue_date"),
}

TIMESTAMP_COLUMNS = {
    "issued_at",
    "delivery_hour",
    "loaded_at",
    "updated_at",
    "started_at",
    "finished_at",
}


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    for table, columns in SCHEMA.items():
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({columns})")


def connect(path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open the warehouse for writing, creating the file and tables if needed."""
    path = Path(path) if path is not None else config.WAREHOUSE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    ensure_schema(con)
    return con


def connect_readonly(path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open the warehouse read-only. A missing warehouse reads as empty tables.

    Read-only connections never create files, so the API can run before the first pipeline run.
    """
    path = Path(path) if path is not None else config.WAREHOUSE_PATH
    if not path.exists():
        con = duckdb.connect(":memory:")
        ensure_schema(con)
        return con
    return duckdb.connect(str(path), read_only=True)


@contextmanager
def session(path: Path | str | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    con = connect(path)
    try:
        yield con
    finally:
        con.close()


def to_utc_naive(values: pd.Series) -> pd.Series:
    """Timestamps as naive UTC. Aware values are converted; naive values are taken as UTC."""
    ts = pd.to_datetime(values)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts.astype("datetime64[us]")


def upsert(con: duckdb.DuckDBPyConnection, table: str, rows: pd.DataFrame) -> int:
    """Insert or replace `rows` on the table's key. Returns the number of rows written.

    Duplicate keys within `rows` keep the last occurrence, so a batch never conflicts with itself.
    """
    if table not in SCHEMA:
        raise ValueError(f"unknown table {table!r}")
    if rows.empty:
        return 0
    columns = [c for c in table_columns(con, table) if c in rows.columns]
    missing = set(KEYS[table]) - set(columns)
    if missing:
        raise ValueError(f"{table}: missing key columns {sorted(missing)}")
    frame = rows[columns].copy()
    for column in TIMESTAMP_COLUMNS & set(columns):
        frame[column] = to_utc_naive(frame[column])
    frame = frame.drop_duplicates(subset=list(KEYS[table]), keep="last")
    con.register("_upsert_rows", frame)
    try:
        column_list = ", ".join(columns)
        con.execute(
            f"INSERT OR REPLACE INTO {table} ({column_list}) SELECT {column_list} FROM _upsert_rows"
        )
    finally:
        con.unregister("_upsert_rows")
    return len(frame)


def table_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()]


def write_forecasts(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame) -> int:
    """Rows with `market, model, issued_at, delivery_hour, price_eur_pred`."""
    return upsert(con, "forecasts", rows)


def write_backtest_results(
    con: duckdb.DuckDBPyConnection, rows: pd.DataFrame, run_id: str | None = None
) -> int:
    """Rows with `market, model, delivery_hour, price_eur, price_eur_pred[, run_id]`.

    `run_id` fills or overrides the column, so `backtest.run(...)` output can be stored as is.
    """
    if run_id is not None:
        rows = rows.assign(run_id=run_id)
    if "run_id" not in rows.columns:
        raise ValueError("backtest_results needs a run_id")
    return upsert(con, "backtest_results", rows)


def read_forecasts(con: duckdb.DuckDBPyConnection, market: str | None = None) -> pd.DataFrame:
    if market is None:
        return con.execute("SELECT * FROM forecasts").df()
    return con.execute("SELECT * FROM forecasts WHERE market = ?", [market]).df()
