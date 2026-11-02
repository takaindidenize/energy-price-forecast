import json
import logging
from datetime import datetime
from pathlib import Path

import duckdb
import httpx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import respx

from price_forecast import config
from price_forecast.ingestion import energy_charts

FIXTURES = Path(__file__).parent / "fixtures"
URL = config.ENERGY_CHARTS_URL


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PRICES_RAW_DIR", tmp_path / "prices")
    return tmp_path / "prices"


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Record retry waits instead of sleeping."""
    waits: list[float] = []
    monkeypatch.setattr(energy_charts, "_sleep", waits.append)
    return waits


def read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@respx.mock
def test_hourly_month_writes_the_raw_prices_contract(data_dir):
    route = respx.get(URL).respond(json=fixture("energy_charts_hourly.json"))

    path = energy_charts.ingest_month("2023-03")

    assert path == data_dir / "market=de_lu" / "year=2023" / "month=03" / "data.parquet"
    params = route.calls.last.request.url.params
    assert params["bzn"] == "DE-LU"
    assert params["start"] == "2023-03-01T00:00+00:00"
    assert params["end"] == "2023-03-31T23:59+00:00"

    df = read(path)
    assert list(df.columns) == [
        "delivery_start",
        "resolution_minutes",
        "price",
        "price_eur",
        "currency",
        "source",
        "loaded_at",
    ]
    assert len(df) == 24
    assert df["delivery_start"].iloc[0] == pd.Timestamp("2023-03-01 00:00")
    assert df["delivery_start"].is_monotonic_increasing
    assert (df["resolution_minutes"] == 60).all()
    assert (df["price"] == df["price_eur"]).all()
    assert df["price"].min() == -10.0  # negative prices survive
    assert set(df["currency"]) == {"EUR"}
    assert set(df["source"]) == {"energy_charts"}

    described = duckdb.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{path}', hive_partitioning = false)"
    ).fetchall()
    types = {row[0]: row[1] for row in described}
    assert types == {
        "delivery_start": "TIMESTAMP",
        "resolution_minutes": "SMALLINT",
        "price": "DOUBLE",
        "price_eur": "DOUBLE",
        "currency": "VARCHAR",
        "source": "VARCHAR",
        "loaded_at": "TIMESTAMP",
    }


@respx.mock
def test_quarter_hourly_month():
    respx.get(URL).respond(json=fixture("energy_charts_quarter_hourly.json"))

    df = read(energy_charts.ingest_month("2025-11"))

    assert len(df) == 12
    assert (df["resolution_minutes"] == 15).all()
    assert df["delivery_start"].diff().dropna().eq(pd.Timedelta(minutes=15)).all()


@respx.mock
def test_switchover_month_has_both_resolutions():
    # 15-minute products start with delivery day 2025-10-01 local = 2025-09-30 22:00 UTC, so the
    # September UTC partition ends with eight quarter-hours.
    respx.get(URL).respond(json=fixture("energy_charts_switchover.json"))

    df = read(energy_charts.ingest_month("2025-09"))

    hourly = df[df["resolution_minutes"] == 60]
    quarter = df[df["resolution_minutes"] == 15]
    assert list(hourly["delivery_start"].dt.hour) == [18, 19, 20, 21]
    assert len(quarter) == 8
    assert quarter["delivery_start"].min() == pd.Timestamp("2025-09-30 22:00")
    assert df["delivery_start"].max() == pd.Timestamp("2025-09-30 23:45")


@respx.mock
def test_switchover_rows_of_next_month_go_to_october():
    respx.get(URL).respond(json=fixture("energy_charts_switchover.json"))

    df = read(energy_charts.ingest_month("2025-10"))

    assert list(df["delivery_start"].dt.strftime("%H:%M")) == ["00:00", "00:15", "00:30", "00:45"]
    assert (df["resolution_minutes"] == 15).all()


@respx.mock
def test_empty_month_returns_none_and_writes_nothing(data_dir, caplog):
    respx.get(URL).respond(json=fixture("energy_charts_empty.json"))

    with caplog.at_level(logging.WARNING):
        assert energy_charts.ingest_month("2030-01") is None

    assert not data_dir.exists() or not any(data_dir.rglob("*"))
    assert "no prices" in caplog.text


@respx.mock
def test_404_means_no_data(data_dir):
    respx.get(URL).respond(404, json={"detail": "No data for given parameters"})

    assert energy_charts.ingest_month("2030-01") is None
    assert not data_dir.exists()


@respx.mock
def test_null_prices_duplicates_and_month_boundary(caplog):
    respx.get(URL).respond(json=fixture("energy_charts_dirty.json"))

    with caplog.at_level(logging.INFO):
        df = read(energy_charts.ingest_month("2023-03"))

    # Feb 28 23:00 UTC is March 1 local midnight and Apr 1 00:00 UTC belongs to April: both are
    # outside the UTC month. 02:00 has a null price. 03:00 and 04:00 are duplicated.
    assert list(df["delivery_start"].dt.strftime("%m-%d %H:%M")) == [
        "03-01 00:00",
        "03-01 01:00",
        "03-01 03:00",
        "03-01 04:00",
        "03-01 05:00",
        "03-31 23:00",
    ]
    assert df.set_index(df["delivery_start"].dt.hour)["price"].loc[4] == 91.0  # last one wins
    assert (df["resolution_minutes"] == 60).all()
    assert "dropped 1 row(s) with a null price" in caplog.text
    assert "dropped 2 duplicate timestamp(s) (1 with differing prices" in caplog.text
    assert "ignored 2 row(s) outside the UTC month" in caplog.text
    assert "gap(s)" in caplog.text


@respx.mock
def test_retry_then_success(sleeps):
    route = respx.get(URL)
    route.side_effect = [
        httpx.Response(503),
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.ConnectError("connection refused"),
        httpx.Response(200, json=fixture("energy_charts_hourly.json")),
    ]

    path = energy_charts.ingest_month("2023-03")

    assert path is not None and len(read(path)) == 24
    assert route.call_count == 4
    assert len(sleeps) == 3
    assert sleeps[1] == 7.0  # Retry-After is honoured


@respx.mock
def test_retries_give_up_after_max_attempts(data_dir):
    route = respx.get(URL).respond(502)

    with pytest.raises(httpx.HTTPStatusError):
        energy_charts.ingest_month("2023-03")

    assert route.call_count == energy_charts.MAX_ATTEMPTS
    assert not data_dir.exists()


@respx.mock
def test_client_errors_are_not_retried():
    route = respx.get(URL).respond(400, json={"detail": "bad bzn"})

    with pytest.raises(httpx.HTTPStatusError):
        energy_charts.ingest_month("2023-03")

    assert route.call_count == 1


@respx.mock
def test_rerun_overwrites_the_partition(data_dir):
    payload = fixture("energy_charts_hourly.json")
    route = respx.get(URL).respond(json=payload)
    path = energy_charts.ingest_month("2023-03")

    payload["price"] = [p + 1 for p in payload["price"]]
    route.respond(json=payload)
    assert energy_charts.ingest_month("2023-03") == path

    df = read(path)
    assert len(df) == 24
    assert df["price"].iloc[0] == pytest.approx(99.1)
    assert [p.name for p in path.parent.iterdir()] == ["data.parquet"]


@respx.mock
def test_crash_while_writing_keeps_the_old_partition(monkeypatch):
    respx.get(URL).respond(json=fixture("energy_charts_hourly.json"))
    path = energy_charts.ingest_month("2023-03")
    before = path.read_bytes()

    def broken_write(table, where):
        Path(where).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(pq, "write_table", broken_write)
    with pytest.raises(OSError):
        energy_charts.ingest_month("2023-03")

    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == ["data.parquet"]


def test_detect_resolution():
    hour = 3600
    hourly = np.arange(0, 4 * hour, hour)
    quarter = np.arange(4 * hour, 6 * hour, 900)
    res = energy_charts.detect_resolution(np.concatenate([hourly, quarter]))
    assert list(res) == [60] * 4 + [15] * 8
    # Off-hour rows are 15-minute even when isolated; an isolated full-hour row is hourly.
    assert list(energy_charts.detect_resolution(np.array([900, 10 * hour]))) == [15, 60]
    assert energy_charts.detect_resolution(np.array([], dtype="int64")).size == 0


def test_unknown_market_is_rejected():
    with pytest.raises(ValueError, match="no Energy-Charts bidding zone"):
        energy_charts.ingest_month("2023-03", market="tr")


def test_month_helpers():
    assert energy_charts.month_bounds_utc("2024-12") == (
        datetime(2024, 12, 1),
        datetime(2025, 1, 1),
    )
    assert list(energy_charts.iter_months("2024-11", "2025-02")) == [
        "2024-11",
        "2024-12",
        "2025-01",
        "2025-02",
    ]
    with pytest.raises(ValueError):
        energy_charts.month_bounds_utc("2024-13")
    with pytest.raises(ValueError):
        list(energy_charts.iter_months("2025-02", "2025-01"))


def test_cli_iterates_months(monkeypatch):
    calls = []
    monkeypatch.setattr(energy_charts, "ingest_month", lambda m, mk: calls.append((m, mk)))

    assert energy_charts.main(["--start", "2025-11", "--end", "2026-01"]) == 0
    assert calls == [("2025-11", "de_lu"), ("2025-12", "de_lu"), ("2026-01", "de_lu")]


def test_cli_defaults_to_config_range(monkeypatch):
    calls = []
    monkeypatch.setattr(energy_charts, "ingest_month", lambda m, mk: calls.append(m))

    energy_charts.main([])
    assert calls[0] == config.START_MONTH and calls[-1] == config.END_MONTH
