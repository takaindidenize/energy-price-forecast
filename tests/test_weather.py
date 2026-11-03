import copy
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import httpx
import pandas as pd
import pytest
import respx

from price_forecast import config
from price_forecast.ingestion import energy_charts, weather

FIXTURES = Path(__file__).parent / "fixtures"
FORECAST_URL = config.OPEN_METEO_HISTORICAL_FORECAST_URL
ARCHIVE_URL = config.OPEN_METEO_ARCHIVE_URL
DE_POINTS = [p.name for p in config.MARKETS["de_lu"].weather_points]


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WEATHER_RAW_DIR", tmp_path / "weather")
    return tmp_path / "weather"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(energy_charts, "_sleep", lambda seconds: None)


class FakeOpenMeteo:
    """Answers any date range like Open-Meteo does, from the fixture templates.

    `covered_from` makes the API return null values before that time (no forecast archived yet).
    `drop_hours` / `duplicate_hours` remove or repeat timestamps to imitate a faulty response.
    """

    def __init__(self, template: str, covered_from: datetime | None = None):
        self.template = json.loads((FIXTURES / template).read_text())
        self.covered_from = covered_from
        self.drop_hours: set[datetime] = set()
        self.duplicate_hours: set[datetime] = set()
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        params = request.url.params
        start = datetime.fromisoformat(params["start_date"])
        end = datetime.fromisoformat(params["end_date"]) + timedelta(days=1)
        lat = float(params["latitude"])
        variables = params["hourly"].split(",")
        hours = []
        t = start
        while t < end:
            if t not in self.drop_hours:
                hours.append(t)
                if t in self.duplicate_hours:
                    hours.append(t)
            t += timedelta(hours=1)

        hourly: dict[str, list] = {
            "time": [int((h - datetime(1970, 1, 1)).total_seconds()) for h in hours]
        }
        for i, name in enumerate(variables):
            hourly[name] = [
                None if self.covered_from and h < self.covered_from else lat + i + h.hour / 100
                for h in hours
            ]
        body = copy.deepcopy(self.template)
        body["latitude"] = lat
        body["hourly"] = hourly
        return httpx.Response(200, json=body)

    def value(self, lat: float, variable: str, hour: int) -> float:
        return lat + weather.FORECAST_REQUEST_VARIABLES.index(variable) + hour / 100


def describe(path: Path) -> dict[str, str]:
    return {
        r[0]: r[1]
        for r in duckdb.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{path}', hive_partitioning = false)"
        ).fetchall()
    }


@respx.mock
def test_full_month_from_the_historical_forecast_api(data_dir):
    forecast = FakeOpenMeteo("open_meteo_historical_forecast.json")
    respx.get(FORECAST_URL).mock(side_effect=forecast)
    archive = respx.get(ARCHIVE_URL).respond(500)

    path = weather.ingest_month("de_lu", "2024-02")

    assert path == data_dir / "market=de_lu" / "year=2024" / "month=02" / "data.parquet"
    assert archive.call_count == 0
    assert len(forecast.requests) == len(DE_POINTS)
    params = forecast.requests[0].url.params
    assert params["start_date"] == "2024-02-01"
    assert params["end_date"] == "2024-02-29"
    assert params["timezone"] == "GMT"
    assert params["timeformat"] == "unixtime"
    assert params["hourly"] == ",".join(weather.FORECAST_REQUEST_VARIABLES)

    df = pd.read_parquet(path)
    assert list(df.columns) == [
        "point",
        "valid_time",
        *config.WEATHER_VARIABLES,
        "loaded_at",
    ]
    assert sorted(df["point"].unique()) == sorted(DE_POINTS)
    assert len(df) == 29 * 24 * len(DE_POINTS)
    assert df["valid_time"].min() == pd.Timestamp("2024-02-01 00:00")
    assert df["valid_time"].max() == pd.Timestamp("2024-02-29 23:00")
    assert describe(path) == {
        "point": "VARCHAR",
        "valid_time": "TIMESTAMP",
        "temperature_2m": "DOUBLE",
        "wind_speed_100m": "DOUBLE",
        "shortwave_radiation": "DOUBLE",
        "cloud_cover": "DOUBLE",
        "loaded_at": "TIMESTAMP",
    }

    row = df[(df["point"] == "berlin") & (df["valid_time"] == pd.Timestamp("2024-02-10 05:00"))]
    lat = 52.52
    v80 = forecast.value(lat, "wind_speed_80m", 5)
    v120 = forecast.value(lat, "wind_speed_120m", 5)
    assert row["wind_speed_100m"].item() == pytest.approx(v80 + 0.5503 * (v120 - v80), abs=1e-3)
    assert row["temperature_2m"].item() == pytest.approx(forecast.value(lat, "temperature_2m", 5))


@respx.mock
def test_march_utc_month_has_744_hours_despite_dst():
    respx.get(FORECAST_URL).mock(side_effect=FakeOpenMeteo("open_meteo_historical_forecast.json"))

    df = pd.read_parquet(weather.ingest_month("tr", "2025-03"))

    assert df.groupby("point").size().eq(31 * 24).all()


@respx.mock
def test_falls_back_to_archive_where_forecasts_are_missing(caplog):
    forecast = FakeOpenMeteo(
        "open_meteo_historical_forecast.json", covered_from=datetime(2024, 1, 3)
    )
    archive = FakeOpenMeteo("open_meteo_archive.json")
    respx.get(FORECAST_URL).mock(side_effect=forecast)
    respx.get(ARCHIVE_URL).mock(side_effect=archive)

    with caplog.at_level(logging.INFO):
        df = pd.read_parquet(weather.ingest_month("de_lu", "2024-01"))

    assert len(df) == 31 * 24 * len(DE_POINTS)
    assert len(archive.requests) == len(DE_POINTS)
    params = archive.requests[0].url.params
    assert (params["start_date"], params["end_date"]) == ("2024-01-01", "2024-01-02")
    assert params["hourly"] == ",".join(config.WEATHER_VARIABLES)
    assert "falling back to the archive" in caplog.text
    assert "48 hour(s) filled from the archive" in caplog.text

    berlin = df[df["point"] == "berlin"].set_index("valid_time")
    lat = 52.52
    # Archive hours carry the archive's own wind_speed_100m (2nd requested variable there).
    assert berlin.loc["2024-01-01 05:00", "wind_speed_100m"] == pytest.approx(lat + 1 + 0.05)
    assert berlin.loc["2024-01-03 05:00", "temperature_2m"] == pytest.approx(lat + 0.05)
    assert berlin.notna().all().all()


@respx.mock
def test_forecast_api_rejecting_the_range_uses_the_archive(caplog):
    respx.get(FORECAST_URL).respond(
        400, json={"error": True, "reason": "Parameter 'start_date' is out of allowed range"}
    )
    respx.get(ARCHIVE_URL).mock(side_effect=FakeOpenMeteo("open_meteo_archive.json"))

    with caplog.at_level(logging.WARNING):
        df = pd.read_parquet(weather.ingest_month("de_lu", "2021-01"))

    assert len(df) == 31 * 24 * len(DE_POINTS)
    assert "out of allowed range" in caplog.text


@respx.mock
def test_no_data_anywhere_returns_none(data_dir):
    reason = {"error": True, "reason": "Parameter 'start_date' is out of allowed range"}
    respx.get(FORECAST_URL).respond(400, json=reason)
    respx.get(ARCHIVE_URL).respond(400, json=reason)

    assert weather.ingest_month("de_lu", "2031-01") is None
    assert not data_dir.exists()


@respx.mock
def test_missing_and_duplicate_hours_are_logged(caplog):
    forecast = FakeOpenMeteo("open_meteo_historical_forecast.json")
    gap = datetime(2024, 4, 10, 12)
    forecast.drop_hours = {gap}
    forecast.duplicate_hours = {datetime(2024, 4, 11, 0)}
    archive = FakeOpenMeteo("open_meteo_archive.json")
    archive.drop_hours = {gap}
    respx.get(FORECAST_URL).mock(side_effect=forecast)
    respx.get(ARCHIVE_URL).mock(side_effect=archive)

    with caplog.at_level(logging.WARNING):
        df = pd.read_parquet(weather.ingest_month("de_lu", "2024-04"))

    assert not df.duplicated(["point", "valid_time"]).any()
    assert df.groupby("point").size().eq(30 * 24 - 1).all()
    assert "dropped 4 duplicate (point, hour) row(s)" in caplog.text
    assert "hamburg is missing 1 of 720 hours" in caplog.text
    params = archive.requests[0].url.params
    assert (params["start_date"], params["end_date"]) == ("2024-04-10", "2024-04-10")


@respx.mock
def test_retry_on_rate_limit():
    route = respx.get(FORECAST_URL)
    fake = FakeOpenMeteo("open_meteo_historical_forecast.json")
    responses = iter([httpx.Response(429), httpx.ConnectTimeout("slow")])

    def flaky(request):
        try:
            response = next(responses)
        except StopIteration:
            return fake(request)
        if isinstance(response, Exception):
            raise response
        return response

    route.mock(side_effect=flaky)

    assert weather.ingest_month("de_lu", "2024-02") is not None
    assert route.call_count == len(DE_POINTS) + 2


@respx.mock
def test_rerun_overwrites_the_partition():
    respx.get(FORECAST_URL).mock(side_effect=FakeOpenMeteo("open_meteo_historical_forecast.json"))
    first = weather.ingest_month("de_lu", "2024-02")
    second = weather.ingest_month("de_lu", "2024-02")

    assert first == second
    assert len(pd.read_parquet(second)) == 29 * 24 * len(DE_POINTS)
    assert [p.name for p in second.parent.iterdir()] == ["data.parquet"]


def test_wind_speed_100m_interpolation():
    v80 = pd.Series([10.0, 10.0, None, None])
    v120 = pd.Series([20.0, None, 20.0, None])
    out = weather.wind_speed_100m(v80, v120)
    assert out[0] == pytest.approx(15.503, abs=1e-3)
    assert out[1] == pytest.approx(10 * 1.25 ** (1 / 7))
    assert out[2] == pytest.approx(20 * (100 / 120) ** (1 / 7))
    assert pd.isna(out[3])


def test_cli_iterates_months(monkeypatch):
    calls = []
    monkeypatch.setattr(weather, "ingest_month", lambda mk, m: calls.append((mk, m)))

    assert weather.main(["--market", "tr", "--start", "2024-12", "--end", "2025-01"]) == 0
    assert calls == [("tr", "2024-12"), ("tr", "2025-01")]


def test_current_month_is_cut_at_the_forecast_horizon(monkeypatch):
    """The API rejects end dates over ~15 days ahead: a whole current month cannot be asked."""
    from datetime import datetime

    from price_forecast.ingestion import weather

    monkeypatch.setattr(weather, "utc_now_naive", lambda: datetime(2026, 11, 3, 11, 30))
    start, end = weather._fetch_window("2026-11")
    assert (start, end) == (pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-11"))
    assert weather._fetch_window("2026-10")[1] == pd.Timestamp("2026-11-01")  # past: whole month
    start, end = weather._fetch_window("2027-01")
    assert end <= start  # entirely beyond the horizon: nothing to fetch
