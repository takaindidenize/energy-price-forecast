import dataclasses
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from price_forecast import bounds, config
from price_forecast.models import make_model


def test_cap_schedule_follows_effective_dates():
    assert bounds.cap_on("tr", date(2023, 3, 31)) == 3050.0
    assert bounds.cap_on("tr", date(2023, 4, 1)) == 2600.0
    assert bounds.cap_on("tr", date(2025, 4, 4)) == 3000.0
    assert bounds.cap_on("tr", date(2025, 4, 5)) == 3400.0
    assert bounds.cap_on("tr", date(2026, 4, 4)) == 4500.0
    assert bounds.cap_on("tr", date(2030, 1, 1)) == 4500.0  # the last cap stays in force
    assert bounds.cap_on("tr", date(2022, 12, 31)) is None
    assert bounds.cap_on("de_lu", date(2025, 6, 1)) is None


def test_limits_use_the_issue_days_exchange_rate():
    # The delivery day's own rate (0.5) must never be used: it is not known at issue time.
    rates = pd.Series({date(2025, 6, 9): 0.02, date(2025, 6, 10): 0.5})
    lower, upper = bounds.limits_eur("tr", pd.Series([date(2025, 6, 10)]), rates)
    assert lower[0] == 0.0
    assert upper[0] == pytest.approx(3400 * 0.02)


def test_limits_fall_back_to_the_latest_known_rate():
    rates = pd.Series({date(2025, 6, 1): 0.025})
    _, upper = bounds.limits_eur("tr", pd.Series([date(2025, 6, 10)]), rates)
    assert upper[0] == pytest.approx(3400 * 0.025)


def test_zero_floor_needs_no_rate_and_unknown_cap_is_open():
    lower, upper = bounds.limits_eur("tr", pd.Series([date(2025, 6, 10)]), pd.Series(dtype=float))
    assert lower[0] == 0.0
    assert upper[0] == np.inf


def test_markets_without_limits_are_untouched():
    preds = pd.Series([-400.0, 25.0, 900.0])
    days = pd.Series([date(2025, 6, 10)] * 3)
    assert bounds.clip("de_lu", days, preds, None).tolist() == [-400.0, 25.0, 900.0]


def test_clip_keeps_turkish_forecasts_within_zero_and_the_cap():
    rates = pd.Series({date(2025, 6, 9): 0.02})
    preds = pd.Series([-145.0, 40.0, 90.0], index=[7, 8, 9])
    days = pd.Series([date(2025, 6, 10)] * 3, index=preds.index)
    clipped = bounds.clip("tr", days, preds, rates)
    assert clipped.tolist() == [0.0, 40.0, pytest.approx(68.0)]
    assert list(clipped.index) == [7, 8, 9]


def test_eur_per_local_reads_the_raw_prices(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PRICES_RAW_DIR", tmp_path)
    path = config.month_partition_path(tmp_path, "tr", "2025-06")
    path.parent.mkdir(parents=True)
    starts = pd.to_datetime(["2025-06-09 21:00", "2025-06-09 22:00", "2025-06-09 23:00"])
    table = pa.table(
        {
            "delivery_start": starts,  # 00:00 to 02:00 on 10 June in Turkey
            "price": [1000.0, 0.0, 2000.0],  # a zero price carries no rate
            "price_eur": [20.0, 0.0, 44.0],
        }
    )
    pq.write_table(table, path)
    rates = bounds.eur_per_local("tr")
    assert list(rates.index) == [date(2025, 6, 10)]
    assert rates.iloc[0] == pytest.approx(0.021)
    assert bounds.eur_per_local("de_lu") is None


class _Constant:
    name = "constant"

    def __init__(self, values):
        self.values = values

    def fit(self, train):
        return self

    def predict(self, features):
        return pd.Series(self.values, index=features.index)


def test_bounded_forecaster_wraps_any_model(monkeypatch):
    from price_forecast.models.bounded import BoundedForecaster

    monkeypatch.setattr(bounds, "eur_per_local", lambda market: pd.Series({date(2025, 6, 9): 0.02}))
    features = pd.DataFrame(
        {"market": "tr", "delivery_date_local": [date(2025, 6, 10)] * 2}, index=[3, 4]
    )
    model = BoundedForecaster(_Constant([-50.0, 500.0]))
    assert model.name == "constant_bounded"
    assert model.fit(features) is model
    assert model.predict(features).tolist() == [0.0, pytest.approx(68.0)]


def test_make_model_builds_bounded_variants():
    model = make_model("lear_bounded")
    assert model.name == "lear_bounded"
    assert model.inner.name == "lear"
    with pytest.raises(ValueError):
        make_model("prophet_bounded")


def test_outdated_limits_are_detected(tmp_path, monkeypatch):
    """A realised price above the configured cap means the regulator changed it."""
    monkeypatch.setattr(config, "PRICES_RAW_DIR", tmp_path)
    path = config.month_partition_path(tmp_path, "tr", "2026-04")
    path.parent.mkdir(parents=True)
    starts = pd.to_datetime(["2026-04-02 12:00", "2026-04-04 12:00"])
    pq.write_table(pa.table({"delivery_start": starts, "price": [3400.0, 4500.0]}), path)
    assert bounds.outdated_limit_days("tr") == []
    old = config.PriceLimits(floor=0.0, caps=(("2025-04-05", 3400.0),))
    monkeypatch.setitem(config.MARKETS, "tr", dataclasses.replace(config.MARKETS["tr"], limits=old))
    assert bounds.outdated_limit_days("tr") == [date(2026, 4, 4)]
    assert bounds.outdated_limit_days("de_lu") == []
