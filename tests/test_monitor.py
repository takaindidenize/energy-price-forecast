from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from price_forecast import monitor, pipeline, store

MARKET = "de_lu"


def realised_hourly(start="2026-10-01", end="2026-10-22", seed=0) -> pd.DataFrame:
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    rng = np.random.default_rng(seed)
    price = 60 + 20 * np.sin(2 * np.pi * hours.hour / 24) + rng.normal(0, 10, len(hours))
    return pd.DataFrame({"delivery_hour": hours, "price_eur": price})


def forecasts_for(realised: pd.DataFrame, model: str, days: list[date], error) -> pd.DataFrame:
    """Forecast rows for local `days`: realised price plus `error(day)`."""
    frames = []
    lookup = realised.set_index("delivery_hour")["price_eur"]
    for day in days:
        hours = pipeline.delivery_hours_utc(MARKET, day)
        frames.append(
            pd.DataFrame(
                {
                    "market": MARKET,
                    "model": model,
                    "issued_at": pipeline.issue_time_utc(MARKET, day - timedelta(days=1)),
                    "delivery_hour": hours,
                    "price_eur_pred": lookup.reindex(hours).to_numpy() + error(day),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def days_between(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def test_seasonal_naive_uses_same_local_hour_across_dst():
    # 2026-10-25 is 25 hours long in Germany; local 12:00 is 11:00 UTC after it, 10:00 UTC before.
    realised = monitor.realised_prices(
        realised_hourly("2026-10-20", "2026-11-03").assign(market=MARKET), MARKET
    )
    snaive = monitor.seasonal_naive(realised, MARKET)
    lookup = dict(zip(realised["delivery_hour"], snaive, strict=True))
    prices = realised.set_index("delivery_hour")["price_eur"]
    assert lookup[pd.Timestamp("2026-10-30 11:00")] == prices[pd.Timestamp("2026-10-23 10:00")]
    assert np.isnan(lookup[pd.Timestamp("2026-10-20 00:00")])  # nothing a week earlier


def test_daily_accuracy_sums():
    realised = monitor.realised_prices(realised_hourly(), MARKET)
    forecasts = forecasts_for(realised, "plus_two", [date(2026, 10, 15)], lambda d: 2.0)
    acc = monitor.daily_accuracy(forecasts, realised, MARKET)
    row = acc.iloc[0]
    assert row["delivery_date_local"] == date(2026, 10, 15)
    assert row["n_hours"] == 24
    assert row["abs_err_sum"] == pytest.approx(48.0)
    assert row["sq_err_sum"] == pytest.approx(96.0)
    assert row["snaive_abs_err_sum"] > 0


def test_latest_issued_forecast_wins():
    realised = monitor.realised_prices(realised_hourly(), MARKET)
    day = date(2026, 10, 15)
    old = forecasts_for(realised, "m", [day], lambda d: 50.0)
    new = old.assign(
        issued_at=old["issued_at"] + pd.Timedelta(hours=1),
        price_eur_pred=old["price_eur_pred"] - 49.0,
    )
    acc = monitor.daily_accuracy(pd.concat([old, new]), realised, MARKET)
    assert acc["abs_err_sum"].iloc[0] == pytest.approx(24.0)


def run_monitor(tmp_path, forecasts, settings, as_of=None):
    warehouse = tmp_path / "w.duckdb"
    with store.session(warehouse) as con:
        store.write_forecasts(con, forecasts)
    return monitor.run_market(
        MARKET,
        settings,
        load_realised=lambda market: realised_hourly(),
        warehouse=warehouse,
        as_of=as_of,
    ), warehouse


def degradation_forecasts(bad_days: int, gap: date | None = None):
    realised = realised_hourly()
    days = days_between(date(2026, 10, 9), date(2026, 10, 20))
    last = days[-1]

    def bad_error(day):
        return 100.0 if (last - day).days < bad_days else 0.5

    good = forecasts_for(realised, "good", days, lambda d: 0.5)
    bad = forecasts_for(realised, "bad", [d for d in days if d != gap], bad_error)
    return pd.concat([good, bad], ignore_index=True)


def test_degraded_after_k_consecutive_worse_days(tmp_path):
    forecasts = degradation_forecasts(bad_days=3)
    status, _ = run_monitor(tmp_path, forecasts, monitor.MonitorSettings(30, 3))
    status = status.set_index("model")
    assert status.loc["bad", "worse_streak_days"] == 3
    assert status.loc["bad", "degraded"]
    assert status.loc["bad", "status"] == "degraded"
    assert not status.loc["good", "degraded"]
    assert status.loc["good", "window_rmae"] < 1
    assert status.loc["good", "evaluated_through"] == date(2026, 10, 20)


def test_not_degraded_below_k(tmp_path):
    status, _ = run_monitor(
        tmp_path, degradation_forecasts(bad_days=3), monitor.MonitorSettings(30, 4)
    )
    assert not status.set_index("model").loc["bad", "degraded"]


def test_missing_day_breaks_the_streak(tmp_path):
    forecasts = degradation_forecasts(bad_days=5, gap=date(2026, 10, 18))
    status, _ = run_monitor(tmp_path, forecasts, monitor.MonitorSettings(30, 3))
    assert status.set_index("model").loc["bad", "worse_streak_days"] == 2


def test_as_of_limits_the_evaluation(tmp_path):
    forecasts = degradation_forecasts(bad_days=3)
    status, _ = run_monitor(
        tmp_path, forecasts, monitor.MonitorSettings(30, 3), as_of=date(2026, 10, 17)
    )
    bad = status.set_index("model").loc["bad"]
    assert not bad["degraded"]
    assert bad["evaluated_through"] == date(2026, 10, 17)


def test_window_limits_rolling_metrics(tmp_path):
    forecasts = degradation_forecasts(bad_days=2)
    status, _ = run_monitor(tmp_path, forecasts, monitor.MonitorSettings(2, 3))
    bad = status.set_index("model").loc["bad"]
    assert bad["window_n_days"] == 2
    assert bad["window_mae"] == pytest.approx(100.0)


def test_monitor_writes_status_idempotently(tmp_path):
    forecasts = degradation_forecasts(bad_days=3)
    _, warehouse = run_monitor(tmp_path, forecasts, monitor.MonitorSettings(30, 3))
    monitor.run_market(
        MARKET,
        monitor.MonitorSettings(30, 3),
        load_realised=lambda m: realised_hourly(),
        warehouse=warehouse,
    )
    with store.session(warehouse) as con:
        assert con.execute("SELECT count(*) FROM model_status").fetchone()[0] == 2
        assert con.execute("SELECT count(*) FROM daily_accuracy").fetchone()[0] == 12 + 12
        n_realised = con.execute("SELECT count(*) FROM realised_prices").fetchone()[0]
    assert n_realised == len(realised_hourly())


def test_settings_validate():
    with pytest.raises(ValueError):
        monitor.MonitorSettings(window_days=0)


def test_run_isolates_market_failures(tmp_path):
    def load(market):
        if market == "tr":
            raise ConnectionError("down")
        return realised_hourly()

    ok = monitor.run(["tr", MARKET], load_realised=load, warehouse=tmp_path / "w.duckdb")
    assert not ok
    with store.session(tmp_path / "w.duckdb") as con:
        assert con.execute("SELECT count(*) FROM realised_prices").fetchone()[0] > 0
