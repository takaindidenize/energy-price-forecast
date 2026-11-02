"""Metrics on hand-computed examples, and the rolling-origin backtest's timing and ordering."""

from __future__ import annotations

import importlib.util
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from price_forecast import backtest, config, metrics
from price_forecast.features import build_features, issue_time, load_hourly, price_known_at
from price_forecast.models import MODEL_NAMES, make_model

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_fixture_data.py"


@pytest.fixture(scope="module")
def raw_dirs(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("make_fixture_data", SCRIPT)
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    raw = tmp_path_factory.mktemp("data") / "raw"
    fx.write_fixtures(raw / "prices", raw / "weather", "2025-01", "2025-11")
    return raw / "prices", raw / "weather"


@pytest.fixture
def on_fixtures(raw_dirs, monkeypatch):
    monkeypatch.setattr(config, "PRICES_RAW_DIR", raw_dirs[0])
    monkeypatch.setattr(config, "WEATHER_RAW_DIR", raw_dirs[1])


@pytest.fixture(scope="module")
def table(raw_dirs):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PRICES_RAW_DIR", raw_dirs[0])
        mp.setattr(config, "WEATHER_RAW_DIR", raw_dirs[1])
        return {m: build_features(load_hourly(m), m) for m in config.MARKETS}


# --- metrics -----------------------------------------------------------------------------------


def test_metrics_on_a_hand_computed_example():
    y = [-10.0, 0.0, 20.0, 50.0]  # prices cross zero
    pred = [-5.0, 0.0, 10.0, 30.0]  # errors 5, 0, 10, 20
    snaive = [0.0, 10.0, 20.0, 10.0]  # errors 10, 10, 0, 40
    assert metrics.mae(y, pred) == 8.75
    assert metrics.rmse(y, pred) == pytest.approx(math.sqrt((25 + 0 + 100 + 400) / 4))
    assert metrics.rmae(y, pred, snaive) == pytest.approx(8.75 / 15.0)
    assert metrics.rmae(y, snaive, snaive) == 1.0
    assert metrics.evaluate(y, pred, snaive) == {
        "n": 4,
        "mae": 8.75,
        "rmse": pytest.approx(math.sqrt(131.25)),
        "rmae": pytest.approx(8.75 / 15.0),
    }


def test_metrics_use_the_same_rows():
    y = [1.0, 2.0, np.nan, 4.0, 5.0]
    pred = [2.0, 2.0, 3.0, np.nan, 5.0]  # errors 1, 0, -, -, 0
    snaive = [1.0, np.nan, 3.0, 4.0, 3.0]  # usable together with pred: rows 0 and 4
    assert metrics.mae(y, pred) == pytest.approx(1 / 3)
    scores = metrics.evaluate(y, pred, snaive)
    assert scores["n"] == 2
    assert scores["mae"] == pytest.approx(0.5)  # rows 0 and 4 only
    assert scores["rmae"] == pytest.approx(0.5 / 1.0)  # benchmark errors 0 and 2
    assert math.isnan(metrics.mae([np.nan], [1.0]))
    assert math.isnan(metrics.rmae([1.0], [2.0], [1.0]))  # perfect benchmark: undefined
    with pytest.raises(ValueError, match="shape"):
        metrics.mae([1.0, 2.0], [1.0])


# --- backtest ----------------------------------------------------------------------------------


class Spy:
    """Records what it is trained on and asked to predict; predicts the issue-day mean."""

    def __init__(self, name: str = "spy") -> None:
        self.name = name
        self.fits: list[pd.DataFrame] = []
        self.calls: list[tuple[date, int, pd.DataFrame]] = []

    def fit(self, train: pd.DataFrame) -> Spy:
        self.fits.append(train)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        self.calls.append((features["delivery_date_local"].iloc[0], len(self.fits), features))
        return features["price_d1_mean"].rename(self.name)


def test_backtest_timing_and_ordering(table):
    spy = Spy()
    start, end = date(2025, 10, 20), date(2025, 11, 2)  # includes the 25-hour 2025-10-26
    out = backtest.run(
        "de_lu", [spy, make_model("naive")], start, end, features=table["de_lu"], refit_every=3
    )
    assert tuple(out.columns) == backtest.PREDICTION_COLUMNS
    n_hours = 14 * 24 + 1
    assert len(out) == 2 * n_hours
    assert out["issued_at"].is_monotonic_increasing
    assert (
        out.groupby("issued_at")["delivery_hour"].apply(lambda s: s.is_monotonic_increasing).all()
    )
    assert list(out["model"].iloc[:2]) == ["spy", "naive"]  # models in the given order
    assert not out.duplicated(["model", "delivery_hour"]).any()
    assert (out["issued_at"] < out["delivery_hour"]).all()
    for day, g in out.groupby("delivery_date_local"):
        assert (g["issued_at"] == issue_time(day, "de_lu")).all()
    assert out["delivery_date_local"].nunique() == 14

    # Refit every third day: on days 0, 3, 6, 9 and 12.
    assert len(spy.fits) == 5
    assert [fits for _, fits, _ in spy.calls] == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]
    for day, n_fits, inputs in spy.calls:
        train = spy.fits[n_fits - 1]
        last_trained = max(train["delivery_date_local"])
        assert last_trained < day  # never trained on the delivery day or later
        assert price_known_at(last_trained, "de_lu") <= issue_time(day, "de_lu")
        assert train["price_eur"].notna().all()
        assert "price_eur" not in inputs.columns  # the target is hidden at predict time
        assert (inputs["delivery_date_local"] == day).all()
    first_train = spy.fits[0]
    assert max(first_train["delivery_date_local"]) == start - timedelta(days=1)  # expanding


def test_backtest_rolling_window_and_daily_refit(table):
    spy = Spy()
    backtest.run(
        "tr", [spy], "2025-09-01", "2025-09-05", features=table["tr"], refit_every=1, window_days=30
    )
    assert len(spy.fits) == 5
    for i, train in enumerate(spy.fits):
        days = set(train["delivery_date_local"])
        assert len(days) == 30
        assert max(days) == date(2025, 8, 31) + timedelta(days=i)


def test_backtest_skips_days_without_enough_history(table):
    spy = Spy()
    out = backtest.run("tr", [spy], "2025-01-10", "2025-02-05", features=table["tr"])
    # Fixture prices start on 2025-01-01 (local), the first forecast needs 28 known days.
    assert min(out["delivery_date_local"]) == date(2025, 1, 29)
    empty = backtest.run("tr", [spy], "2025-01-10", "2025-01-12", features=table["tr"])
    assert empty.empty and tuple(empty.columns) == backtest.PREDICTION_COLUMNS


def test_backtest_rejects_bad_arguments(table):
    with pytest.raises(ValueError, match="unique"):
        backtest.run("tr", [Spy(), Spy()], "2025-09-01", "2025-09-02", features=table["tr"])
    with pytest.raises(ValueError, match="before"):
        backtest.run("tr", [Spy()], "2025-09-02", "2025-09-01", features=table["tr"])
    with pytest.raises(ValueError, match="refit_every"):
        backtest.run("tr", [Spy()], "2025-09-01", "2025-09-02", features=table["tr"], refit_every=0)


def test_backtest_end_to_end_from_raw_files(on_fixtures):
    models = [make_model(name) for name in MODEL_NAMES]
    out = backtest.run("de_lu", models, "2025-10-01", "2025-10-28", refit_every=7)
    summary = backtest.summarize(out)
    assert tuple(summary.columns) == backtest.SUMMARY_COLUMNS
    assert list(summary["model"]) == list(MODEL_NAMES)
    scores = summary.set_index("model")
    assert scores.loc["seasonal_naive", "rmae"] == 1.0
    assert (scores["n"] == 28 * 24 + 1).all()  # October 2025 has a 25-hour day
    assert scores.loc["lear", "rmae"] < 0.8
    assert scores.loc["lightgbm", "rmae"] < 0.8
    naive = out[out["model"] == "naive"]
    assert out["price_eur"].notna().all() and naive["price_eur_pred"].notna().all()


def test_summarize_on_constructed_predictions():
    hours = pd.date_range("2025-06-02", periods=2, freq="h", tz="UTC")
    rows = []
    for model, preds in (("a", [12.0, 18.0]), ("b", [10.0, 30.0])):
        for h, p, y, s in zip(hours, preds, [10.0, 20.0], [14.0, 20.0], strict=True):
            rows.append(
                {"market": "tr", "model": model, "delivery_hour": h, "price_eur": y}
                | {"price_eur_pred": p, "price_eur_snaive": s}
            )
    summary = backtest.summarize(pd.DataFrame(rows)).set_index("model")
    assert summary.loc["a", "mae"] == 2.0 and summary.loc["a", "rmae"] == 1.0
    assert summary.loc["b", "mae"] == 5.0 and summary.loc["b", "rmae"] == 2.5
    assert summary.loc["b", "rmse"] == pytest.approx(math.sqrt(50.0))
