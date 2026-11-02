"""Turkish day-ahead prices (PTF / MCP) from the EPİAŞ Transparency Platform.

Writes one raw prices partition per month (see "Raw data contracts" in CLAUDE.md) to
`config.month_partition_path(config.PRICES_RAW_DIR, "tr", month)`.

Sources
-------
The official documentation could not be fetched from the environment this module was written in
(the EPİAŞ hosts were blocked by its network policy), so every detail below was taken from the
open-source EPİAŞ client `eptr2` 1.4.0 (PyPI, MIT licence), which calls the same API, and has to
be checked against the official pages before the first real run:

- Electricity service technical docs:
  https://seffaflik.epias.com.tr/electricity-service/technical/en/index.html
  (Day-Ahead Market -> Market Clearing Price (MCP)).
- MCP page on the platform:
  https://seffaflik.epias.com.tr/electricity/electricity-markets/day-ahead-market-dam/market-clearing-price-mcp
- CAS REST login guide: https://www.epias.com.tr/wp-content/themes/epias/docs/cas-rest.pdf

Authentication (CAS ticket-granting ticket)
    `POST config.EPIAS_TICKET_URL` (https://giris.epias.com.tr/cas/v1/tickets), form-encoded
    `username` and `password`, header `Accept: text/plain`. Success is HTTP 201 with the ticket
    (`TGT-...`) as the plain-text body. A ticket lives 2 hours. Data requests carry it in a
    header named `TGT`. (eptr2: `EPTR2.get_tgt` and `transparency_call` in `eptr2/main.py`.)
    Apereo CAS servers also return the ticket URL in a `Location` header, so that is accepted as
    a fallback.

Market clearing price
    `POST {config.EPIAS_BASE_URL}/v1/markets/dam/data/mcp`, JSON body
    `{"startDate": "2023-01-01T00:00:00+03:00", "endDate": "2023-01-07T00:00:00+03:00"}`:
    local (Turkish) dates at midnight with the +03:00 offset. The response is
    `{"items": [{"date", "hour", "price", "priceUsd", "priceEur"}, ...]}`: one item per hour,
    `date` an ISO timestamp with offset, `hour` as "HH:MM", `price` in TRY/MWh and
    `priceUsd`/`priceEur` in USD/EUR per MWh. (eptr2: `eptr2/assets/eptr2_api_schema.json`
    entry "mcp", `eptr2/mapping/parameters.py` for the body labels,
    `eptr2/processing/preprocess/params.py::format_date_epias_hour` for the date format,
    `eptr2/assets/skills/eptr2-price-analysis/api-reference.md` for the response fields.)

Not confirmed (no account and no access to the official pages yet)
    - whether `endDate` is inclusive; it is treated as inclusive here, rows are filtered to the
      requested dates, and a month with missing hours is logged as a warning;
    - any per-request date-range limit; none was found, so each month is split into requests
      of at most `MAX_DAYS_PER_REQUEST` days to stay well under any plausible limit;
    - whether the MCP response is paginated (eptr2 sends no `page` object for it);
    - the status code for an expired or invalid TGT on data requests (401 is assumed);
    - the status code for wrong credentials at the ticket endpoint (400/401/403 are handled).

Turkey has been on UTC+3 all year since 2016. Requests use local dates and the stored
`delivery_start` is UTC, as everywhere in this project.

Credentials come only from the `EPIAS_USERNAME` and `EPIAS_PASSWORD` environment variables. They
and the ticket are never logged, printed or put in exception messages.

CLI: `python -m price_forecast.ingestion.epias --start 2023-01 --end 2025-12`
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from tenacity.wait import wait_base

from price_forecast import config

log = logging.getLogger(__name__)

MARKET = "tr"
SOURCE = "epias"
RESOLUTION_MINUTES = 60
USERNAME_ENV = "EPIAS_USERNAME"
PASSWORD_ENV = "EPIAS_PASSWORD"

MCP_URL = f"{config.EPIAS_BASE_URL}/v1/markets/dam/data/mcp"

TGT_LIFETIME = timedelta(hours=2)
# Renew this long before the documented expiry so a ticket never runs out mid-request.
TGT_RENEW_MARGIN = timedelta(minutes=15)
# No documented limit was found (see module docstring); a week per request is cheap and safe.
MAX_DAYS_PER_REQUEST = 7

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 5

SCHEMA = pa.schema(
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

_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


class EpiasError(RuntimeError):
    """Any failure talking to EPİAŞ. Messages never contain credentials or tickets."""


class EpiasCredentialsError(EpiasError):
    """The credential environment variables are missing."""


class EpiasAuthError(EpiasError):
    """Login failed or a fresh ticket was rejected."""


class EpiasResponseError(EpiasError):
    """The response did not have the documented shape."""


class _RetryableStatus(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _market_tz() -> ZoneInfo:
    return ZoneInfo(config.MARKETS[MARKET].timezone)


def read_credentials() -> tuple[str, str]:
    """(username, password) from the environment, or a clear error naming what is missing."""
    username = os.environ.get(USERNAME_ENV, "")
    password = os.environ.get(PASSWORD_ENV, "")
    missing = [
        name for name, value in ((USERNAME_ENV, username), (PASSWORD_ENV, password)) if not value
    ]
    if missing:
        raise EpiasCredentialsError(
            "EPİAŞ credentials are missing: set the environment variable(s) "
            + " and ".join(missing)
            + " (the EPİAŞ Transparency Platform account's username and password)."
        )
    return username, password


class EpiasClient:
    """Small EPİAŞ client: CAS ticket handling, retries and the MCP endpoint.

    The ticket is reused until `TGT_RENEW_MARGIN` before its 2-hour lifetime ends, and renewed
    once if a data request answers 401.
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        *,
        http: httpx.Client | None = None,
        wait: wait_base | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if username is None or password is None:
            username, password = read_credentials()
        self._username = username
        self._password = password
        self._owns_http = http is None
        self._http = http or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._wait = wait if wait is not None else wait_exponential(multiplier=1, min=1, max=30)
        self._max_attempts = max_attempts
        self._clock = clock
        self._tgt: str | None = None
        self._tgt_obtained_at = 0.0

    def __repr__(self) -> str:  # never show credentials or the ticket
        return f"{type(self).__name__}(has_ticket={self._tgt is not None})"

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> EpiasClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- HTTP with retries -------------------------------------------------------------------

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """One request, retried with exponential backoff on 429, 5xx and network errors."""
        retrying = Retrying(
            retry=retry_if_exception_type((_RetryableStatus, httpx.TransportError)),
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            reraise=True,
            before_sleep=lambda state: log.warning(
                "EPİAŞ request to %s failed (%s), attempt %d of %d; retrying",
                url,
                state.outcome.exception() if state.outcome else "?",
                state.attempt_number,
                self._max_attempts,
            ),
        )
        try:
            for attempt in retrying:
                with attempt:
                    response = self._http.request(method, url, **kwargs)
                    if response.status_code in RETRY_STATUSES:
                        raise _RetryableStatus(response.status_code)
        except _RetryableStatus as exc:
            raise EpiasError(
                f"EPİAŞ request to {url} still failed after {self._max_attempts} attempts "
                f"(last: HTTP {exc.status_code})"
            ) from None
        except httpx.TransportError as exc:
            raise EpiasError(
                f"EPİAŞ request to {url} still failed after {self._max_attempts} attempts "
                f"(last: {type(exc).__name__})"
            ) from None
        return response

    # -- tickets -----------------------------------------------------------------------------

    def _ticket_is_fresh(self) -> bool:
        if self._tgt is None:
            return False
        age = self._clock() - self._tgt_obtained_at
        return age < (TGT_LIFETIME - TGT_RENEW_MARGIN).total_seconds()

    def _ticket(self) -> str:
        if not self._ticket_is_fresh():
            self._tgt = self._new_ticket()
            self._tgt_obtained_at = self._clock()
        assert self._tgt is not None
        return self._tgt

    def _new_ticket(self) -> str:
        log.info("Requesting a new EPİAŞ ticket-granting ticket")
        response = self._send(
            "POST",
            config.EPIAS_TICKET_URL,
            data={"username": self._username, "password": self._password},
            headers={"Accept": "text/plain"},
        )
        # Response bodies from the login endpoint are never echoed: they could quote the input.
        if response.status_code in (400, 401, 403):
            raise EpiasAuthError(
                f"EPİAŞ rejected the login (HTTP {response.status_code}). "
                f"Check {USERNAME_ENV} and {PASSWORD_ENV}."
            )
        if response.status_code != 201:
            raise EpiasAuthError(
                f"Unexpected HTTP {response.status_code} from the EPİAŞ ticket endpoint "
                "(expected 201)."
            )
        body = response.text.strip()
        if body.startswith("TGT-"):
            return body
        location = response.headers.get("Location", "").rstrip("/").rsplit("/", 1)[-1]
        if location.startswith("TGT-"):
            return location
        raise EpiasAuthError("The EPİAŞ ticket endpoint answered 201 but returned no TGT.")

    # -- data --------------------------------------------------------------------------------

    def post_json(self, url: str, body: dict[str, Any]) -> Any:
        """POST a JSON body with the ticket header; renew the ticket once on 401."""
        for renewed in (False, True):
            response = self._send(
                "POST",
                url,
                json=body,
                headers={"TGT": self._ticket(), "Accept": "application/json"},
            )
            if response.status_code != 401:
                break
            if renewed:
                raise EpiasAuthError(f"EPİAŞ answered 401 to {url} even with a fresh ticket.")
            log.info("EPİAŞ answered 401; renewing the ticket and retrying once")
            self._tgt = None
        if response.status_code >= 400:
            snippet = response.text[:200].replace("\n", " ")
            raise EpiasError(f"EPİAŞ answered HTTP {response.status_code} to {url}: {snippet}")
        try:
            return response.json()
        except ValueError:
            raise EpiasResponseError(
                f"EPİAŞ answered {url} with a body that is not JSON."
            ) from None

    def fetch_mcp(self, start: date, end: date) -> list[dict[str, Any]]:
        """Raw MCP items for local delivery dates `start` to `end` (both included)."""
        payload = self.post_json(
            MCP_URL, {"startDate": _local_midnight(start), "endDate": _local_midnight(end)}
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise EpiasResponseError("The MCP response has no 'items' list.")
        return payload["items"]


# -- parsing -----------------------------------------------------------------------------------


def _local_midnight(day: date) -> str:
    """'2023-01-01T00:00:00+03:00' for a local date."""
    return datetime(day.year, day.month, day.day, tzinfo=_market_tz()).isoformat()


def _number(value: Any, *, field: str, index: int, nullable: bool) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EpiasResponseError(f"MCP item {index}: '{field}' is not a number ({value!r}).")
    return float(value)


def parse_mcp_items(items: list[dict[str, Any]]) -> pd.DataFrame:
    """MCP items -> `delivery_start` (naive UTC), `local_date`, `price` (TRY), `price_eur`."""
    tz = _market_tz()
    rows = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "date" not in item or "price" not in item:
            raise EpiasResponseError(f"MCP item {i} lacks 'date' or 'price'.")
        try:
            ts = datetime.fromisoformat(str(item["date"]))
        except ValueError:
            raise EpiasResponseError(f"MCP item {i}: unreadable date {item['date']!r}.") from None
        ts = ts.replace(tzinfo=tz) if ts.tzinfo is None else ts.astimezone(tz)
        hour_text = item.get("hour")
        if hour_text is not None:
            match = re.fullmatch(r"(\d{1,2}):00(:00)?", str(hour_text))
            if match is None or int(match.group(1)) > 23:
                raise EpiasResponseError(f"MCP item {i}: unreadable hour {hour_text!r}.")
            hour = int(match.group(1))
            if ts.hour == 0 and ts.minute == 0:
                ts = ts.replace(hour=hour)
            elif ts.hour != hour:
                raise EpiasResponseError(
                    f"MCP item {i}: date {item['date']!r} and hour {hour_text!r} disagree."
                )
        rows.append(
            {
                "delivery_start": ts.astimezone(UTC).replace(tzinfo=None),
                "local_date": ts.date(),
                "price": _number(item["price"], field="price", index=i, nullable=False),
                "price_eur": _number(
                    item.get("priceEur"), field="priceEur", index=i, nullable=True
                ),
            }
        )
    frame = pd.DataFrame(rows, columns=["delivery_start", "local_date", "price", "price_eur"])
    return frame.astype({"price": "float64", "price_eur": "float64"})


# -- months ------------------------------------------------------------------------------------


def _month_bounds(month: str) -> tuple[date, date]:
    if not _MONTH_RE.match(month):
        raise ValueError(f"Month must look like 'YYYY-MM', got {month!r}.")
    first = date(int(month[:4]), int(month[5:]), 1)
    next_first = date(first.year + first.month // 12, first.month % 12 + 1, 1)
    return first, next_first - timedelta(days=1)


def _chunks(first: date, last: date, max_days: int) -> Iterator[tuple[date, date]]:
    start = first
    while start <= last:
        end = min(start + timedelta(days=max_days - 1), last)
        yield start, end
        start = end + timedelta(days=1)


def iter_months(start: str, end: str) -> Iterator[str]:
    """'YYYY-MM' months from `start` to `end`, both included."""
    first, _ = _month_bounds(start)
    last, _ = _month_bounds(end)
    if first > last:
        raise ValueError(f"Start month {start} is after end month {end}.")
    current = first
    while current <= last:
        yield current.strftime("%Y-%m")
        current = date(current.year + current.month // 12, current.month % 12 + 1, 1)


def _write_atomic(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def ingest_month(month: str, *, client: EpiasClient | None = None) -> Path | None:
    """Fetch the hourly PTF for one month and write its raw partition.

    Returns the written path, or None (and writes nothing) when EPİAŞ has no data for the month.
    Re-running a month overwrites its partition.
    """
    first, last = _month_bounds(month)
    own_client = client is None
    client = client or EpiasClient()
    frames = []
    try:
        for start, end in _chunks(first, last, MAX_DAYS_PER_REQUEST):
            frame = parse_mcp_items(client.fetch_mcp(start, end))
            # Keep only the requested dates, whatever the endDate semantics turn out to be.
            frames.append(frame[(frame["local_date"] >= start) & (frame["local_date"] <= end)])
    finally:
        if own_client:
            client.close()

    frames = [f for f in frames if not f.empty]
    if not frames:
        log.info("No EPİAŞ PTF data for %s; nothing written", month)
        return None
    data = pd.concat(frames, ignore_index=True)
    duplicates = data.duplicated("delivery_start")
    if duplicates.any():
        log.warning("%s: dropping %d duplicate hours", month, int(duplicates.sum()))
        data = data[~duplicates]
    data = data.sort_values("delivery_start", ignore_index=True)

    expected_hours = ((last - first).days + 1) * 24
    if len(data) < expected_hours:
        log.warning("%s: %d of %d hours present", month, len(data), expected_hours)

    loaded_at = datetime.now(UTC).replace(tzinfo=None)
    table = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "delivery_start": data["delivery_start"],
                "resolution_minutes": RESOLUTION_MINUTES,
                "price": data["price"],
                "price_eur": data["price_eur"],
                "currency": config.MARKETS[MARKET].currency,
                "source": SOURCE,
                "loaded_at": loaded_at,
            }
        ),
        schema=SCHEMA,
        preserve_index=False,
    )
    path = config.month_partition_path(config.PRICES_RAW_DIR, MARKET, month)
    _write_atomic(table, path)
    log.info("%s: wrote %d hours to %s", month, len(data), path)
    return path


# -- CLI ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m price_forecast.ingestion.epias",
        description="Download EPİAŞ day-ahead prices (PTF) into monthly raw partitions.",
    )
    parser.add_argument("--start", default=config.START_MONTH, help="first month, YYYY-MM")
    parser.add_argument("--end", default=config.END_MONTH, help="last month, YYYY-MM")
    args = parser.parse_args(argv)
    try:
        months = list(iter_months(args.start, args.end))
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        with EpiasClient() as client:
            for month in months:
                ingest_month(month, client=client)
    except EpiasCredentialsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except EpiasError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
