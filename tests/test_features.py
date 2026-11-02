"""Fixture data, hourly loading and the modelling table, including the no-look-ahead guarantee."""

from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from price_forecast import config
from price_forecast.features import (
    FEATURE_COLUMNS,
    MODEL_TABLE_COLUMNS,
    WEATHER_COLUMNS,
    build_features,
    issue_time,
    load_hourly,
    price_known_at,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_fixture_data.py"
FIXTURE_MONTHS = ("2025-02", "2025-11")  # spring and autumn DST, 15-minute switch, holidays


def _fixture_module():
    spec = importlib.util.spec_from_file_location("make_fixture_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fx = _fixture_module()


@pytest.fixture(scope="module")
def raw_dirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("data") / "raw"
    prices, weather = base / "prices", base / "weather"
    fx.write_fixtures(prices, weather, *FIXTURE_MONTHS)
    return prices, weather


@pytest.fixture(scope="module")
def hourly(raw_dirs):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PRICES_RAW_DIR", raw_dirs[0])
        mp.setattr(config, "WEATHER_RAW_DIR", raw_dirs[1])
        return {m: load_hourly(m) for m in config.MARKETS}


@pytest.fixture(scope="module")
def table(hourly):
    return {m: build_features(h, m) for m, h in hourly.items()}


def _raw(base: Path, market: str) -> pd.DataFrame:
    files = sorted((base / f"market={market}").glob("year=*/month=*/data.parquet"))
    return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)


def _day(table: pd.DataFrame, day: str) -> pd.DataFrame:
    return table[table["delivery_date_local"] == date.fromisoformat(day)]


# --- fixture data ------------------------------------------------------------------------------


def test_fixture_partitions_follow_the_contract(raw_dirs):
    prices, weather = raw_dirs
    for market in config.MARKETS:
        for base, schema in ((prices, fx.PRICE_SCHEMA), (weather, fx.WEATHER_SCHEMA)):
            files = sorted((base / f"market={market}").glob("year=*/month=*/data.parquet"))
            assert len(files) == 10
            for f in files:
                assert pq.read_schema(f).remove_metadata() == schema
                assert f == config.month_partition_path(
                    base, market, f"{f.parent.parent.name[5:]}-{f.parent.name[6:]}"
                )


def test_fixture_prices_have_the_awkward_features(raw_dirs):
    de = _raw(raw_dirs[0], "de_lu")
    switch = datetime(2025, 9, 30, 22)  # 2025-10-01 00:00 Europe/Berlin
    assert set(de.loc[de["delivery_start"] < switch, "resolution_minutes"]) == {60}
    assert set(de.loc[de["delivery_start"] >= switch, "resolution_minutes"]) == {15}
    assert (de["delivery_start"].dt.minute != 0).any()
    assert (de["price"] < 0).mean() > 0.01  # negative prices
    assert de["price_eur"].equals(de["price"])
    assert set(de["currency"]) == {"EUR"} and set(de["source"]) == {"energy_charts"}

    tr = _raw(raw_dirs[0], "tr")
    assert set(tr["resolution_minutes"]) == {60}
    assert tr["price"].max() == fx.TR_PRICE_CAP_TRY
    assert (tr["price"] == fx.TR_PRICE_CAP_TRY).mean() > 0.01  # the cap binds
    assert tr["price"].min() >= 0
    assert tr["price_eur"].isna().any() and tr["price_eur"].notna().mean() > 0.99
    assert set(tr["currency"]) == {"TRY"} and set(tr["source"]) == {"epias"}

    weather = _raw(raw_dirs[1], "tr")
    assert set(weather["point"]) == {p.name for p in config.MARKETS["tr"].weather_points}
    assert not weather.duplicated(["point", "valid_time"]).any()


def test_fixture_data_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    paths_a = fx.write_fixtures(a / "p", a / "w", "2025-03", "2025-03")
    paths_b = fx.write_fixtures(b / "p", b / "w", "2025-03", "2025-03")
    assert [p.relative_to(a) for p in paths_a] == [p.relative_to(b) for p in paths_b]
    for pa_, pb_ in zip(paths_a, paths_b, strict=True):
        assert pa_.read_bytes() == pb_.read_bytes()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point config (as if PF_DATA_DIR had been set before import) at a scratch directory."""
    d = tmp_path / "pf"
    monkeypatch.setenv("PF_DATA_DIR", str(d))
    monkeypatch.setattr(config, "DATA_DIR", d)
    monkeypatch.setattr(config, "RAW_DIR", d / "raw")
    monkeypatch.setattr(config, "PRICES_RAW_DIR", d / "raw" / "prices")
    monkeypatch.setattr(config, "WEATHER_RAW_DIR", d / "raw" / "weather")
    return d


def test_script_refuses_without_pf_data_dir(data_dir, monkeypatch):
    monkeypatch.delenv("PF_DATA_DIR")
    with pytest.raises(SystemExit, match="PF_DATA_DIR is not set"):
        fx.main(["--start", "2025-01", "--end", "2025-01"])
    assert not data_dir.exists()


def test_script_refuses_the_real_data_dir(monkeypatch):
    monkeypatch.setenv("PF_DATA_DIR", str(config.PROJECT_ROOT / "data"))
    with pytest.raises(SystemExit, match="default real data directory"):
        fx.main(["--start", "2025-01", "--end", "2025-01"])


def test_script_refuses_to_mix_with_real_data(data_dir):
    real = config.month_partition_path(config.PRICES_RAW_DIR, "de_lu", "2025-01")
    real.parent.mkdir(parents=True)
    real.write_bytes(b"not a fixture")
    with pytest.raises(SystemExit, match="not written by this script"):
        fx.main(["--start", "2025-01", "--end", "2025-01"])


def test_script_writes_fixtures_and_can_rerun(data_dir):
    for _ in range(2):  # idempotent: the second run overwrites its own fixtures
        assert fx.main(["--start", "2025-01", "--end", "2025-01", "--market", "tr"]) == 0
    assert (data_dir / fx.MARKER_FILE).exists()
    assert config.month_partition_path(config.PRICES_RAW_DIR, "tr", "2025-01").exists()
    assert not (config.PRICES_RAW_DIR / "market=de_lu").exists()


# --- load_hourly -------------------------------------------------------------------------------


def _write(path: Path, rows: dict, schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(rows, schema=schema), path)


def test_load_hourly_averages_quarter_hours_and_keeps_latest_load(tmp_path, monkeypatch):
    prices, weather = tmp_path / "prices", tmp_path / "weather"
    monkeypatch.setattr(config, "PRICES_RAW_DIR", prices)
    monkeypatch.setattr(config, "WEATHER_RAW_DIR", weather)
    t0, old, new = datetime(2025, 10, 5, 10), datetime(2025, 10, 6), datetime(2025, 10, 7)
    starts = [t0 + timedelta(minutes=15 * q) for q in range(4)] + [t0 + timedelta(hours=1)] * 2
    _write(
        config.month_partition_path(prices, "de_lu", "2025-10"),
        {
            "delivery_start": starts,
            "resolution_minutes": [15] * 4 + [60, 60],
            "price": [10.0, 20.0, -30.0, 40.0, 99.0, 55.0],
            "price_eur": [10.0, 20.0, -30.0, 40.0, 99.0, 55.0],
            "currency": ["EUR"] * 6,
            "source": ["energy_charts"] * 6,
            "loaded_at": [old] * 4 + [old, new],  # the 11:00 price was reloaded
        },
        fx.PRICE_SCHEMA,
    )
    points = [p.name for p in config.MARKETS["de_lu"].weather_points]
    n = len(points) + 1
    _write(
        config.month_partition_path(weather, "de_lu", "2025-10"),
        {
            "point": [*points, "elsewhere"],
            "valid_time": [t0 + timedelta(hours=2)] * n,  # weather only: tomorrow-like hour
            "temperature_2m": [*[float(i) for i in range(len(points))], 1000.0],
            "wind_speed_100m": [1.0] * n,
            "shortwave_radiation": [0.0] * n,
            "cloud_cover": [50.0] * n,
            "loaded_at": [old] * n,
        },
        fx.WEATHER_SCHEMA,
    )
    hourly = load_hourly("de_lu").set_index("delivery_hour")
    assert list(hourly.index) == list(
        pd.date_range("2025-10-05 10:00", periods=3, freq="h", tz="UTC")
    )
    assert hourly["price_eur"].iloc[0] == pytest.approx(10.0)  # mean of four quarter-hours
    assert hourly["n_intervals"].iloc[0] == 4
    assert hourly["price_eur"].iloc[1] == 55.0  # latest loaded_at wins
    assert np.isnan(hourly["price_eur"].iloc[2])
    # Averaged over the configured points only.
    assert hourly["temperature_2m"].iloc[2] == pytest.approx(np.mean(range(len(points))))


def test_load_hourly_without_partitions_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PRICES_RAW_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="no price partitions"):
        load_hourly("tr")


def test_load_hourly_on_fixtures(hourly):
    for market, h in hourly.items():
        assert h["delivery_hour"].is_unique and h["delivery_hour"].is_monotonic_increasing
        assert str(h["delivery_hour"].dt.tz) == "UTC"
        assert (h["delivery_hour"].dt.minute == 0).all()
        assert set(h["n_intervals"]) == ({1, 4} if market == "de_lu" else {1})
        assert h[list(WEATHER_COLUMNS)].notna().all().all()


# --- the modelling table -----------------------------------------------------------------------


def test_modelling_table_columns_and_types(table):
    for market, t in table.items():
        assert tuple(t.columns) == MODEL_TABLE_COLUMNS
        assert (t["market"] == market).all()
        assert t["delivery_hour"].is_unique
        steps = t["delivery_hour"].diff().dropna()
        assert (steps == pd.Timedelta(hours=1)).all()  # a complete UTC grid
        assert t["hour_local"].between(0, 23).all()
        assert t["day_of_week"].between(1, 7).all()
        assert t["is_holiday"].dtype == bool
        assert all(isinstance(d, date) for d in t["delivery_date_local"].head())
        for col in ("price_eur", "price_d1_same_hour", "price_d7_same_hour", "price_d1_mean"):
            assert t[col].dtype == float


def test_calendar_columns(table):
    de, tr = table["de_lu"], table["tr"]
    assert _day(de, "2025-10-03")["is_holiday"].all()  # German Unity Day
    assert not _day(de, "2025-10-29")["is_holiday"].any()
    assert _day(tr, "2025-10-29")["is_holiday"].all()  # Republic Day
    assert (_day(de, "2025-06-02")["day_of_week"] == 1).all()  # a Monday
    assert (_day(tr, "2025-06-08")["day_of_week"] == 7).all()


def test_daylight_saving_days(table):
    de, tr = table["de_lu"], table["tr"]
    spring, autumn = _day(de, "2025-03-30"), _day(de, "2025-10-26")
    assert len(spring) == 23 and 2 not in set(spring["hour_local"])
    assert len(autumn) == 25 and (autumn["hour_local"] == 2).sum() == 2
    assert len(_day(de, "2025-10-01")) == 24  # first 15-minute day, averaged to hours
    days = tr.groupby("delivery_date_local").size()
    assert (days.iloc[1:-1] == 24).all()  # Turkey has no DST

    # The day after a DST switch still has every same-hour lag.
    after_spring, after_autumn = _day(de, "2025-03-31"), _day(de, "2025-10-27")
    assert after_spring["price_d1_same_hour"].notna().all()
    assert after_autumn["price_d1_same_hour"].notna().all()
    s = spring.set_index("hour_local")["price_eur"]
    lag = after_spring.set_index("hour_local")["price_d1_same_hour"]
    assert lag[2] == pytest.approx((s[1] + s[3]) / 2)  # the skipped hour: mean of neighbours
    repeated = autumn.loc[autumn["hour_local"] == 2, "price_eur"].mean()
    assert after_autumn.set_index("hour_local")["price_d1_same_hour"][2] == pytest.approx(repeated)


def test_table_extends_to_the_next_delivery_day(hourly):
    h = hourly["tr"]
    cut = pd.Timestamp("2025-06-10 21:00", tz="UTC")  # 2025-06-11 00:00 in Istanbul
    t = build_features(h[h["delivery_hour"] < cut], "tr")
    nxt = _day(t, "2025-06-11")
    assert len(nxt) == 24
    assert nxt["price_eur"].isna().all()
    assert nxt["price_d1_same_hour"].notna().all() and nxt["price_d7_same_hour"].notna().all()


def _constructed_hourly(days: int = 21) -> pd.DataFrame:
    """Turkey (no DST): price = 100 * day number + local hour."""
    start = pd.Timestamp("2025-06-01", tz=config.MARKETS["tr"].timezone).tz_convert("UTC")
    hours = pd.date_range(start, periods=24 * days, freq="h")
    local = hours.tz_convert(config.MARKETS["tr"].timezone)
    day_no = (local.normalize() - local[0].normalize()).days
    frame = pd.DataFrame(
        {"delivery_hour": hours, "price_eur": 100.0 * day_no + local.hour.to_numpy()}
    )
    for col in WEATHER_COLUMNS:
        frame[col] = 1.0
    return frame


def test_price_lags_on_constructed_data():
    t = build_features(_constructed_hourly(), "tr")
    t = t[t["price_eur"].notna()]
    day_no = (t["price_eur"] // 100).astype(int)
    ok7 = day_no >= 7
    np.testing.assert_array_equal(
        t.loc[day_no >= 1, "price_d1_same_hour"], t["price_eur"][day_no >= 1] - 100
    )
    np.testing.assert_array_equal(t.loc[ok7, "price_d7_same_hour"], t["price_eur"][ok7] - 700)
    np.testing.assert_allclose(
        t.loc[day_no >= 1, "price_d1_mean"], (day_no[day_no >= 1] - 1) * 100 + 11.5
    )
    assert t.loc[day_no == 0, "price_d1_same_hour"].isna().all()
    assert t.loc[day_no < 7, "price_d7_same_hour"].isna().all()


# --- no look-ahead -----------------------------------------------------------------------------


def _leaked_columns(build, hourly: pd.DataFrame, market: str, delivery_day: date) -> set[str]:
    """Columns of `delivery_day`'s rows that change when information that was not available at
    the day's issue time is altered (corrupted) or removed (truncated).

    Unavailable at issue time: prices of any day whose `price_known_at` is after the issue time,
    and weather beyond the end of the delivery day (the forecast in hand covers that day).
    """
    cutoff = issue_time(delivery_day, market)
    tz = config.MARKETS[market].timezone
    local_day = hourly["delivery_hour"].dt.tz_convert(tz).dt.date
    known_at = {d: price_known_at(d, market) for d in local_day.unique()}
    price_unknown = (local_day.map(known_at) > cutoff).to_numpy()
    weather_unknown = (
        hourly["delivery_hour"] >= price_known_at(delivery_day + timedelta(1), market)
    ).to_numpy()
    assert price_unknown.any() and (~price_unknown).any()

    rng = np.random.default_rng(1)
    corrupted = hourly.copy()
    noise = rng.normal(300.0, 400.0, price_unknown.sum())
    noise[::5] = np.nan
    corrupted.loc[price_unknown, "price_eur"] = noise
    for col in WEATHER_COLUMNS:
        corrupted.loc[weather_unknown, col] = rng.normal(0, 100, weather_unknown.sum())
    truncated = hourly[~weather_unknown].copy()
    truncated.loc[price_unknown[~weather_unknown], "price_eur"] = np.nan

    keys = ["delivery_hour", "delivery_date_local", "hour_local", *FEATURE_COLUMNS]
    reference = _day(build(hourly, market), delivery_day.isoformat())[keys].reset_index(drop=True)
    assert len(reference) >= 23
    leaked = set()
    for variant in (corrupted, truncated):
        other = _day(build(variant, market), delivery_day.isoformat())[keys].reset_index(drop=True)
        assert len(other) == len(reference)
        for col in keys:
            if not reference[col].equals(other[col]):
                leaked.add(col)
    return leaked


LOOKAHEAD_DAYS = [
    ("de_lu", "2025-03-30"),  # 23 hours
    ("de_lu", "2025-03-31"),  # after the spring switch
    ("de_lu", "2025-06-15"),
    ("de_lu", "2025-10-01"),  # first 15-minute day
    ("de_lu", "2025-10-26"),  # 25 hours
    ("de_lu", "2025-10-27"),
    ("tr", "2025-03-30"),
    ("tr", "2025-06-15"),
    ("tr", "2025-10-29"),  # holiday
]


@pytest.mark.parametrize(("market", "day"), LOOKAHEAD_DAYS)
def test_no_feature_uses_information_from_after_the_issue_time(hourly, market, day):
    assert _leaked_columns(build_features, hourly[market], market, date.fromisoformat(day)) == set()


def test_lookahead_check_catches_leaks(hourly):
    """The check above must fail for features that peek at the delivery day."""

    def same_day_lag(h, market):
        t = build_features(h, market)
        t["price_d1_same_hour"] = t["price_eur"]
        return t

    def same_day_mean(h, market):
        t = build_features(h, market)
        t["price_d1_mean"] = t.groupby("delivery_date_local")["price_eur"].transform("mean")
        return t

    def next_day_weather(h, market):
        t = build_features(h, market)
        t["wind_speed_100m"] = t["wind_speed_100m"].shift(-24)
        return t

    day = date(2025, 6, 15)
    assert _leaked_columns(same_day_lag, hourly["tr"], "tr", day) == {"price_d1_same_hour"}
    assert _leaked_columns(same_day_mean, hourly["tr"], "tr", day) == {"price_d1_mean"}
    assert _leaked_columns(next_day_weather, hourly["tr"], "tr", day) == {"wind_speed_100m"}


def test_issue_time_and_price_known_at():
    # 11:00 local on the day before; Berlin is UTC+2 in summer, Istanbul UTC+3 all year.
    assert issue_time(date(2025, 6, 15), "de_lu") == pd.Timestamp("2025-06-14 09:00", tz="UTC")
    assert issue_time(date(2025, 12, 15), "de_lu") == pd.Timestamp("2025-12-14 10:00", tz="UTC")
    assert issue_time(date(2025, 12, 15), "tr") == pd.Timestamp("2025-12-14 08:00", tz="UTC")
    d = date(2025, 6, 15)
    # The issue day's prices are known at issue time, the delivery day's are not.
    assert (
        price_known_at(d - timedelta(1), "de_lu")
        <= issue_time(d, "de_lu")
        < price_known_at(d, "de_lu")
    )
