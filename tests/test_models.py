"""Forecasters: the interface, exact naive benchmarks, and sane fits on fixture data."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from price_forecast import config
from price_forecast.features import WEATHER_COLUMNS, build_features, load_hourly
from price_forecast.models import MODEL_NAMES, make_model
from price_forecast.models._common import AsinhScaler
from price_forecast.models.lear import LEARForecaster
from price_forecast.models.naive import NaiveForecaster
from price_forecast.models.seasonal_naive import SeasonalNaiveForecaster

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_fixture_data.py"


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("make_fixture_data", SCRIPT)
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    raw = tmp_path_factory.mktemp("data") / "raw"
    fx.write_fixtures(raw / "prices", raw / "weather", "2025-01", "2025-11")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PRICES_RAW_DIR", raw / "prices")
        mp.setattr(config, "WEATHER_RAW_DIR", raw / "weather")
        return {m: build_features(load_hourly(m), m) for m in config.MARKETS}


def _split(t: pd.DataFrame, first_test_day: str, test_days: int = 7):
    """Train on every known day before `first_test_day`, test on the next `test_days` days."""
    day = pd.to_datetime(t["delivery_date_local"])
    start = pd.Timestamp(first_test_day)
    train = t[(day < start) & t["price_eur"].notna()]
    test = t[(day >= start) & (day < start + pd.Timedelta(days=test_days))]
    return train, test


def _mae(a, b) -> float:
    return float(np.nanmean(np.abs(np.asarray(a) - np.asarray(b))))


def _constructed_table() -> pd.DataFrame:
    """Turkey (no DST), 28 days, price = 100 * day number + local hour."""
    tz = config.MARKETS["tr"].timezone
    hours = pd.date_range(
        pd.Timestamp("2025-06-01", tz=tz).tz_convert("UTC"), periods=24 * 28, freq="h"
    )
    local = hours.tz_convert(tz)
    day_no = (local.normalize() - local[0].normalize()).days
    hourly = pd.DataFrame(
        {"delivery_hour": hours, "price_eur": 100.0 * day_no + local.hour.to_numpy()}
    )
    for col in WEATHER_COLUMNS:
        hourly[col] = 0.0
    return build_features(hourly, "tr")


def test_registry():
    assert MODEL_NAMES == ("naive", "seasonal_naive", "lear", "lightgbm", "median")
    for name in MODEL_NAMES:
        assert make_model(name).name == name
    assert make_model("lear", calibration_days=56).calibration_days == 56
    with pytest.raises(ValueError, match="unknown model"):
        make_model("prophet")


def test_naive_models_are_exact_on_constructed_data():
    t = _constructed_table()
    test = t[t["delivery_date_local"] == date(2025, 6, 20)].drop(columns="price_eur")
    truth = t.loc[test.index, "price_eur"]
    naive = NaiveForecaster().fit(t).predict(test)
    snaive = SeasonalNaiveForecaster().fit(t).predict(test)
    np.testing.assert_array_equal(naive, truth - 100)  # same hour on the issue day
    np.testing.assert_array_equal(snaive, truth - 700)  # same hour a week before delivery
    assert naive.index.equals(test.index) and snaive.index.equals(test.index)


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("market", ["de_lu", "tr"])
def test_forecaster_interface(table, name, market):
    train, test = _split(table[market], "2025-06-01")
    model = make_model(name)
    assert model.fit(train) is model
    inputs = test.drop(columns="price_eur")
    pred = model.predict(inputs)
    assert isinstance(pred, pd.Series)
    assert pred.index.equals(inputs.index)
    if name in ("naive", "seasonal_naive"):
        # Missing only where the lag is (Turkey has a few hours without an EUR price).
        lag = inputs[model.column]
        assert pred.isna().equals(lag.isna())
    else:
        assert pred.notna().all() and np.isfinite(pred).all()  # missing inputs are imputed
    # Predicting never needs the target, and does not change if the target is present.
    np.testing.assert_allclose(model.predict(test.assign(price_eur=1e6)), pred)


@pytest.mark.parametrize("market", ["de_lu", "tr"])
def test_fitted_models_beat_the_seasonal_naive(table, market):
    train, test = _split(table[market], "2025-07-01", test_days=28)
    truth = test["price_eur"]
    benchmark = _mae(truth, test["price_d7_same_hour"])
    for name in ("lear", "lightgbm"):
        pred = make_model(name).fit(train).predict(test.drop(columns="price_eur"))
        assert _mae(truth, pred) < 0.8 * benchmark, name


def test_lear_handles_dst_days_and_short_history(table):
    t = table["de_lu"]
    train, test = _split(t, "2025-10-26", test_days=1)  # the 25-hour day
    short = train[pd.to_datetime(train["delivery_date_local"]) >= pd.Timestamp("2025-10-12")]
    for frame in (train, short):  # 14 days < number of inputs: noise-variance fallback
        pred = LEARForecaster().fit(frame).predict(test.drop(columns="price_eur"))
        assert len(pred) == 25 and pred.notna().all()
        # The two 02:00 rows share the hour-2 model but have their own weather.
        assert pred[(test["hour_local"] == 2).to_numpy()].nunique() == 2


def test_lear_calibration_window_and_coefficients(table):
    train, test = _split(table["tr"], "2025-08-01")
    full = LEARForecaster().fit(train)
    recent = LEARForecaster(calibration_days=56).fit(train)
    inputs = test.drop(columns="price_eur")
    assert not np.allclose(full.predict(inputs), recent.predict(inputs))
    coefs = full.coefficients()
    assert coefs.shape == (24, len(full.columns_))
    assert (coefs != 0).any(axis=1).all()  # every hour keeps some inputs
    assert (coefs == 0).any(axis=1).all()  # and LASSO drops others
    # Same-hour yesterday is the strongest kind of input somewhere.
    assert (coefs.abs().idxmax(axis=1).str.startswith(("d1_", "d7_"))).any()


def test_lear_rejects_too_little_data(table):
    train, _ = _split(table["tr"], "2025-01-05")
    with pytest.raises(ValueError, match="at least"):
        LEARForecaster().fit(train)


def test_lightgbm_is_deterministic(table):
    train, test = _split(table["tr"], "2025-06-01")
    inputs = test.drop(columns="price_eur")
    a = make_model("lightgbm").fit(train).predict(inputs)
    b = make_model("lightgbm").fit(train).predict(inputs)
    pd.testing.assert_series_equal(a, b)


def test_asinh_scaler_round_trip_and_negative_prices():
    prices = np.array([-500.0, -20.0, 0.0, 35.0, 80.0, 90.0, 4000.0, np.nan])
    s = AsinhScaler().fit(prices)
    assert s.median_ == 35.0
    z = s.transform(prices)
    np.testing.assert_allclose(s.inverse_transform(z), prices)
    assert np.all(np.diff(z[:-1]) > 0)  # monotone, also through zero
    assert abs(z[-2]) < 0.1 * (4000 - 35) / s.scale_  # tails are compressed
    flat = AsinhScaler().fit([50.0] * 9 + [60.0])  # MAD = 0: falls back to the std
    assert flat.scale_ == pytest.approx(3.0)
    with pytest.raises(ValueError):
        AsinhScaler().fit([np.nan])


def test_median_takes_the_middle_forecast_every_hour():
    from price_forecast.models.median import MedianForecaster

    class Fixed:
        def __init__(self, values):
            self.values = values

        def fit(self, train):
            self.fitted = True
            return self

        def predict(self, features):
            return pd.Series(self.values, index=features.index)

    model = MedianForecaster()
    model.members = [Fixed([1.0, 50.0]), Fixed([2.0, 40.0]), Fixed([900.0, -5.0])]
    frame = pd.DataFrame({"x": [0, 0]}, index=[7, 8])
    assert model.fit(frame) is model
    assert all(m.fitted for m in model.members)
    result = model.predict(frame)
    assert result.tolist() == [2.0, 40.0]  # the outlier of the three is ignored
    assert list(result.index) == [7, 8] and result.name == "median"


def test_median_members_are_all_history_short_window_and_lightgbm():
    model = make_model("median_bounded")
    assert model.name == "median_bounded"
    assert [getattr(m, "calibration_days", "n/a") for m in model.inner.members] == [
        None,
        180,
        "n/a",
    ]
