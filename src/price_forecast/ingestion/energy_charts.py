"""Day-ahead prices for European bidding zones from the Energy-Charts API (Fraunhofer ISE).

`GET config.ENERGY_CHARTS_URL?bzn=DE-LU&start=...&end=...` returns parallel arrays
`unix_seconds` (UTC start of each delivery interval) and `price` (EUR/MWh), plus `unit` and
`license_info`. No API key is needed. DE-LU is hourly until delivery day 2025-09-30 and 15-minute
from delivery day 2025-10-01 (local), i.e. from 2025-09-30T22:00 UTC. Partitions are UTC months,
so the 2025-09 partition already ends with eight 15-minute rows.

This module also holds the small HTTP and partition-writing helpers that `weather.py` reuses.

    python -m price_forecast.ingestion.energy_charts --start 2023-01 --end 2025-12
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from price_forecast import config

log = logging.getLogger(__name__)

SOURCE = "energy_charts"

# Market code -> Energy-Charts bidding zone. Add more European zones here.
BIDDING_ZONES: dict[str, str] = {"de_lu": "DE-LU"}

PRICES_SCHEMA = pa.schema(
    [
        ("delivery_start", pa.timestamp("us")),
        ("resolution_minutes", pa.int16()),
        ("price", pa.float64()),
        ("price_eur", pa.float64()),
        ("currency", pa.string()),
        ("source", pa.string()),
        ("loaded_at", pa.timestamp("us")),
    ]
)

# --- shared HTTP and partition helpers ------------------------------------------------------------

HTTP_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
USER_AGENT = "energy-price-forecast/0.1 (portfolio project)"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 6
MAX_RETRY_AFTER_SECONDS = 120.0

_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_backoff = wait_exponential_jitter(initial=1, max=60)
# Indirection so tests can replace the sleep without waiting.
_sleep = time.sleep


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in RETRY_STATUS


def _wait(state: RetryCallState) -> float:
    """Honour a numeric `Retry-After` header (429/503), otherwise exponential backoff."""
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("Retry-After", "")
        try:
            return min(max(float(retry_after), 0.0), MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass
    return _backoff(state)


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning(
        "Attempt %d failed (%s), retrying in %.1fs",
        state.attempt_number,
        exc,
        state.next_action.sleep if state.next_action else 0.0,
    )


def get_json(client: httpx.Client, url: str, params: dict[str, Any]) -> httpx.Response:
    """GET with retries on 429/5xx and transport errors. Other 4xx are returned, not raised."""
    retrying = Retrying(
        retry=retry_if_exception(_is_retryable),
        wait=_wait,
        stop=stop_after_attempt(MAX_ATTEMPTS),
        sleep=lambda seconds: _sleep(seconds),
        before_sleep=_log_retry,
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            response = client.get(url, params=params)
            if response.status_code in RETRY_STATUS:
                response.raise_for_status()
    return response


def new_client() -> httpx.Client:
    return httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})


def validate_month(month: str) -> str:
    if not _MONTH_RE.match(month):
        raise ValueError(f"month must look like 'YYYY-MM', got {month!r}")
    return month


def month_bounds_utc(month: str) -> tuple[datetime, datetime]:
    """Naive UTC [start, end) of a 'YYYY-MM' month."""
    year, mm = map(int, validate_month(month).split("-"))
    start = datetime(year, mm, 1)
    end = datetime(year + mm // 12, mm % 12 + 1, 1)
    return start, end


def iter_months(start: str, end: str) -> Iterator[str]:
    """Inclusive 'YYYY-MM' range."""
    current, _ = month_bounds_utc(start)
    last, _ = month_bounds_utc(end)
    if current > last:
        raise ValueError(f"start month {start} is after end month {end}")
    while current <= last:
        yield f"{current:%Y-%m}"
        current = month_bounds_utc(f"{current:%Y-%m}")[1]


def utc_now_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def write_partition(table: pa.Table, path: Path) -> Path:
    """Write `table` to `path` atomically: temporary file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


# --- Energy-Charts prices -------------------------------------------------------------------------


def bidding_zone(market: str) -> str:
    try:
        return BIDDING_ZONES[market]
    except KeyError:
        raise ValueError(
            f"market {market!r} has no Energy-Charts bidding zone; known: {sorted(BIDDING_ZONES)}"
        ) from None


def _iso_utc(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M+00:00")


def fetch_prices(client: httpx.Client, bzn: str, start: datetime, end: datetime) -> dict[str, Any]:
    """Raw Energy-Charts response for [start, end) UTC, or `{}` if the API reports no data.

    `end` is sent as one minute before the exclusive end so that an inclusive end on the API side
    cannot pull in the first interval of the next month; rows are filtered again afterwards.
    """
    params = {
        "bzn": bzn,
        "start": _iso_utc(start),
        "end": _iso_utc(end - timedelta(minutes=1)),
    }
    response = get_json(client, config.ENERGY_CHARTS_URL, params)
    if response.status_code == 404:
        # Energy-Charts answers 404 with a JSON `detail` when there is no data for the range.
        log.warning("Energy-Charts has no data for %s %s: %s", bzn, params, response.text[:200])
        return {}
    response.raise_for_status()
    return response.json()


def detect_resolution(unix_seconds: np.ndarray) -> np.ndarray:
    """Resolution in minutes (15 or 60) per row, from the gap to the nearest neighbour.

    Expects sorted, unique timestamps. A row that is not on a full hour is always 15-minute. A row
    on the full hour is 15-minute if a neighbour is at most 45 minutes away, otherwise hourly; an
    isolated row (no neighbour) is hourly. Gaps in the data therefore only matter for full-hour
    rows whose quarter-hour neighbours are all missing.
    """
    ts = np.asarray(unix_seconds, dtype="int64")
    if ts.size == 0:
        return np.empty(0, dtype="int16")
    gaps = np.diff(ts).astype("float64")
    forward = np.append(gaps, np.inf)
    backward = np.insert(gaps, 0, np.inf)
    nearest = np.minimum(forward, backward)
    quarter = (ts % 3600 != 0) | (nearest <= 45 * 60)
    return np.where(quarter, 15, 60).astype("int16")


def _log_gaps(df: pd.DataFrame, bzn: str, month: str) -> None:
    step = df["delivery_start"].diff().shift(-1)
    expected = pd.to_timedelta(df["resolution_minutes"].astype("int64"), unit="min")
    gaps = df[(step > expected).fillna(False)]
    if not gaps.empty:
        log.warning(
            "%s %s: %d gap(s) in the price series, first after %s",
            bzn,
            month,
            len(gaps),
            gaps["delivery_start"].iloc[0],
        )


def to_frame(payload: dict[str, Any], bzn: str, month: str) -> pd.DataFrame:
    """Energy-Charts payload -> raw prices contract rows for the UTC month (without `loaded_at`)."""
    start, end = month_bounds_utc(month)
    seconds = payload.get("unix_seconds") or []
    prices = payload.get("price") or []
    if len(seconds) != len(prices):
        raise ValueError(
            f"{bzn} {month}: unix_seconds and price have different lengths "
            f"({len(seconds)} vs {len(prices)})"
        )
    unit = payload.get("unit")
    if unit and unit.replace(" ", "").upper() != "EUR/MWH":
        raise ValueError(f"{bzn} {month}: unexpected price unit {unit!r}")

    df = pd.DataFrame(
        {
            "unix_seconds": pd.array(seconds, dtype="Int64"),
            "price": pd.array(prices, dtype="Float64"),
        }
    )
    df = df[df["unix_seconds"].notna()].astype({"unix_seconds": "int64"})
    df = df.sort_values("unix_seconds", kind="stable")

    n_dupes = int(df["unix_seconds"].duplicated().sum())
    if n_dupes:
        conflicting = df[df["unix_seconds"].duplicated(keep=False)].groupby("unix_seconds")["price"]
        n_conflicts = int((conflicting.nunique(dropna=False) > 1).sum())
        log.warning(
            "%s %s: dropped %d duplicate timestamp(s) (%d with differing prices, kept the last)",
            bzn,
            month,
            n_dupes,
            n_conflicts,
        )
        df = df.drop_duplicates("unix_seconds", keep="last")

    # Resolution comes from the full timestamp grid, before null prices leave holes in it.
    df["resolution_minutes"] = detect_resolution(df["unix_seconds"].to_numpy())
    df["delivery_start"] = pd.to_datetime(df["unix_seconds"], unit="s").astype("datetime64[us]")

    outside = (df["delivery_start"] < start) | (df["delivery_start"] >= end)
    if outside.any():
        log.info("%s %s: ignored %d row(s) outside the UTC month", bzn, month, int(outside.sum()))
        df = df[~outside]

    n_null = int(df["price"].isna().sum())
    if n_null:
        log.warning("%s %s: dropped %d row(s) with a null price", bzn, month, n_null)
        df = df[df["price"].notna()]

    df = df.reset_index(drop=True)
    _log_gaps(df, bzn, month)
    price = df["price"].astype("float64")
    return pd.DataFrame(
        {
            "delivery_start": df["delivery_start"],
            "resolution_minutes": df["resolution_minutes"].astype("int16"),
            "price": price,
            "price_eur": price,
            "currency": "EUR",
            "source": SOURCE,
        }
    )


def ingest_month(month: str, market: str = "de_lu") -> Path | None:
    """Fetch one UTC month of day-ahead prices and write its partition. None if no data."""
    bzn = bidding_zone(market)
    start, end = month_bounds_utc(month)
    with new_client() as client:
        payload = fetch_prices(client, bzn, start, end)
    if payload.get("deprecated"):
        log.warning("Energy-Charts marks the /price response as deprecated")

    df = to_frame(payload, bzn, month)
    if df.empty:
        log.warning("%s %s: no prices, nothing written", bzn, month)
        return None

    df["loaded_at"] = pd.Timestamp(utc_now_naive()).as_unit("us")
    table = pa.Table.from_pandas(df, schema=PRICES_SCHEMA, preserve_index=False)
    path = write_partition(table, config.month_partition_path(config.PRICES_RAW_DIR, market, month))
    counts = df["resolution_minutes"].value_counts().sort_index().to_dict()
    log.info(
        "%s %s: wrote %d rows (by resolution in minutes: %s) to %s",
        bzn,
        month,
        len(df),
        counts,
        path,
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest Energy-Charts day-ahead prices.")
    parser.add_argument("--start", default=config.START_MONTH, help="first month, YYYY-MM")
    parser.add_argument("--end", default=config.END_MONTH, help="last month (inclusive), YYYY-MM")
    parser.add_argument("--market", default="de_lu", choices=sorted(BIDDING_ZONES))
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    written = skipped = 0
    for month in iter_months(args.start, args.end):
        if ingest_month(month, args.market) is None:
            skipped += 1
        else:
            written += 1
    log.info("Done: %d month(s) written, %d without data", written, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
