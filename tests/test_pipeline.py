import sys
import types
from datetime import date

import numpy as np
import pandas as pd
import pytest

from price_forecast import pipeline, store

# --- fakes that follow the module interfaces in CLAUDE.md -------------------------------------


def fake_hourly(market: str, start="2026-09-01", end="2026-11-06") -> pd.DataFrame:
    """Hourly prices including hours *after* the issue day, as in a historical re-run."""
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    price = 60 + 20 * np.sin(2 * np.pi * hours.hour / 24) + (5 if market == "tr" else 0)
    return pd.DataFrame({"market": market, "delivery_hour": hours, "price_eur": price})


def fake_build_features(hourly: pd.DataFrame, market: str) -> pd.DataFrame:
    tz = pipeline.config.MARKETS[market].timezone
    out = hourly.sort_values("delivery_hour").reset_index(drop=True).copy()
    local = out["delivery_hour"].dt.tz_localize("UTC").dt.tz_convert(tz)
    out["market"] = market
    out["delivery_date_local"] = local.dt.date
    out["hour_local"] = local.dt.hour
    lookup = out.set_index("delivery_hour")["price_eur"]
    out["price_d1_same_hour"] = (out["delivery_hour"] - pd.Timedelta(days=1)).map(lookup)
    return out


class MeanModel:
    name = "mean"

    def fit(self, train):
        self.seen = train
        self.value = train["price_eur"].mean()
        return self

    def predict(self, features):
        self.predicted = features
        return pd.Series(self.value, index=features.index)


class LagModel:
    name = "lag"

    def fit(self, train):
        return self

    def predict(self, features):
        return features["price_d1_same_hour"]


class BrokenModel:
    name = "broken"

    def fit(self, train):
        raise RuntimeError("cannot fit")

    def predict(self, features):  # pragma: no cover
        raise AssertionError


def make_deps(*, fail_load_for=(), ingested=None, models=None):
    models = models if models is not None else {}

    def load_hourly(market):
        if market in fail_load_for:
            raise ConnectionError(f"raw data for {market} unavailable")
        return fake_hourly(market)

    def make_model(name):
        cls = {"mean": MeanModel, "lag": LagModel, "broken": BrokenModel}[name]
        models[name] = cls()
        return models[name]

    def ingest(kind):
        def fn(market, month):
            if ingested is not None:
                ingested.append((kind, market, month))
            return None

        return fn

    return pipeline.PipelineDeps(
        ingest_prices=ingest("prices"),
        ingest_weather=ingest("weather"),
        load_hourly=load_hourly,
        build_features=fake_build_features,
        make_model=make_model,
    )


@pytest.fixture
def warehouse(tmp_path):
    return tmp_path / "warehouse.duckdb"


def read(warehouse, sql):
    with store.session(warehouse) as con:
        return con.execute(sql).df()


# --- time helpers -------------------------------------------------------------------------------


def test_issue_time_is_local_issue_hour_in_utc():
    assert pipeline.issue_time_utc("de_lu", date(2026, 7, 1)) == pd.Timestamp("2026-07-01 09:00")
    assert pipeline.issue_time_utc("de_lu", date(2026, 11, 2)) == pd.Timestamp("2026-11-02 10:00")
    assert pipeline.issue_time_utc("tr", date(2026, 11, 2)) == pd.Timestamp("2026-11-02 08:00")


@pytest.mark.parametrize(
    "market, day, hours",
    [
        ("de_lu", date(2026, 3, 29), 23),
        ("de_lu", date(2026, 10, 25), 25),
        ("de_lu", date(2026, 11, 3), 24),
        ("tr", date(2026, 3, 29), 24),
    ],
)
def test_delivery_day_hours_follow_dst(market, day, hours):
    assert len(pipeline.delivery_hours_utc(market, day)) == hours


def test_months_to_ingest_cover_month_boundaries():
    assert pipeline.months_to_ingest(date(2026, 11, 15)) == ["2026-11"]
    assert pipeline.months_to_ingest(date(2026, 11, 1)) == ["2026-10", "2026-11"]
    assert pipeline.months_to_ingest(date(2026, 11, 30)) == ["2026-11", "2026-12"]


# --- run ----------------------------------------------------------------------------------------


def test_run_market_stores_d_plus_1_forecasts(warehouse):
    ingested = []
    deps = make_deps(ingested=ingested)
    result = pipeline.run_market(
        "de_lu", date(2026, 11, 2), ["mean", "lag"], deps=deps, warehouse=warehouse
    )
    assert result.status == "ok", result.error
    assert result.models_ok == ["mean", "lag"]
    assert result.n_forecasts == 48
    assert ("prices", "de_lu", "2026-11") in ingested
    assert ("weather", "de_lu", "2026-11") in ingested

    rows = read(warehouse, "SELECT * FROM forecasts ORDER BY model, delivery_hour")
    assert set(rows["issued_at"]) == {pd.Timestamp("2026-11-02 10:00")}
    lag = rows[rows["model"] == "lag"]
    # Local 2026-11-03 00:00 CET is 2026-11-02 23:00 UTC.
    assert lag["delivery_hour"].iloc[0] == pd.Timestamp("2026-11-02 23:00")
    assert lag["delivery_hour"].iloc[-1] == pd.Timestamp("2026-11-03 22:00")

    runs = read(warehouse, "SELECT * FROM pipeline_runs")
    assert runs[["market", "status", "n_forecasts"]].values.tolist() == [["de_lu", "ok", 48]]


def test_training_uses_only_prices_known_at_issue_time(warehouse):
    models = {}
    deps = make_deps(models=models)
    pipeline.run_market("de_lu", date(2026, 11, 2), ["mean"], deps=deps, warehouse=warehouse)
    model = models["mean"]
    # Prices are known through the end of local day D (2026-11-02 23:00 UTC is 11-03 00:00 CET).
    assert model.seen["delivery_hour"].max() == pd.Timestamp("2026-11-02 22:00")
    assert model.seen["price_eur"].notna().all()
    assert set(model.predicted["delivery_date_local"]) == {date(2026, 11, 3)}


def test_dst_delivery_day_has_25_forecasts(warehouse):
    deps = make_deps()
    result = pipeline.run_market(
        "de_lu", date(2026, 10, 24), ["lag"], deps=deps, warehouse=warehouse
    )
    assert result.status == "ok"
    assert result.n_forecasts == 25


def test_future_rows_are_added_when_hourly_stops_at_last_price(warehouse):
    deps = make_deps()
    deps.load_hourly = lambda market: fake_hourly(market, end="2026-11-03")  # ends with day D (UTC)
    result = pipeline.run_market("tr", date(2026, 11, 2), ["mean"], deps=deps, warehouse=warehouse)
    assert result.status == "ok", result.error
    assert result.n_forecasts == 24


def test_failing_model_marks_market_partial(warehouse):
    result = pipeline.run_market(
        "de_lu", date(2026, 11, 2), ["broken", "mean"], deps=make_deps(), warehouse=warehouse
    )
    assert result.status == "partial"
    assert result.models_ok == ["mean"]
    assert "cannot fit" in result.models_failed["broken"]
    assert read(warehouse, "SELECT DISTINCT model FROM forecasts")["model"].tolist() == ["mean"]


def test_one_market_failing_does_not_stop_the_other(warehouse):
    deps = make_deps(fail_load_for=("de_lu",))
    results = pipeline.run(
        ["de_lu", "tr"], date(2026, 11, 2), ["mean"], deps=deps, warehouse=warehouse
    )
    by_market = {r.market: r for r in results}
    assert by_market["de_lu"].status == "failed"
    assert "raw data for de_lu unavailable" in by_market["de_lu"].error
    assert by_market["tr"].status == "ok"
    assert read(warehouse, "SELECT DISTINCT market FROM forecasts")["market"].tolist() == ["tr"]
    runs = read(warehouse, "SELECT market, status FROM pipeline_runs ORDER BY market")
    assert runs.values.tolist() == [["de_lu", "failed"], ["tr", "ok"]]
    text = pipeline.summary(results)
    assert "FAILED" in text and "OK" in text


def test_rerun_is_idempotent(warehouse):
    for _ in range(2):
        pipeline.run_market(
            "tr", date(2026, 11, 2), ["mean"], deps=make_deps(), warehouse=warehouse
        )
    assert read(warehouse, "SELECT count(*) AS n FROM forecasts")["n"].iloc[0] == 24
    assert read(warehouse, "SELECT count(*) AS n FROM pipeline_runs")["n"].iloc[0] == 1


def test_default_deps_import_other_areas_lazily(monkeypatch, warehouse):
    """The real wiring resolves the module interfaces from CLAUDE.md at call time."""
    calls = []

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    module(
        "price_forecast.ingestion.energy_charts",
        ingest_month=lambda month, market="de_lu": calls.append(("ec", month, market)),
    )
    module(
        "price_forecast.ingestion.weather",
        ingest_month=lambda market, month: calls.append(("weather", market, month)),
    )
    module(
        "price_forecast.features",
        load_hourly=fake_hourly,
        build_features=fake_build_features,
    )
    lag_module = module("price_forecast.models.lag_test")
    lag_class = type(
        "LagTest", (LagModel,), {"name": "lag_test", "__module__": lag_module.__name__}
    )
    lag_module.LagTest = lag_class

    result = pipeline.run_market("de_lu", date(2026, 11, 2), ["lag_test"], warehouse=warehouse)
    assert result.status == "ok", result.error
    assert ("ec", "2026-11", "de_lu") in calls
    assert ("weather", "de_lu", "2026-11") in calls


def test_make_model_rejects_bad_names():
    with pytest.raises(ValueError):
        pipeline.make_model("../etc")
    with pytest.raises(ValueError):
        pipeline.make_model("base")


def test_cli_exit_code_reflects_failures(monkeypatch, warehouse, capsys):
    deps = make_deps(fail_load_for=("tr",))
    monkeypatch.setattr(pipeline, "default_deps", lambda: deps)
    code = pipeline.main(
        [
            "--market",
            "de_lu,tr",
            "--date",
            "2026-11-02",
            "--models",
            "mean",
            "--warehouse",
            str(warehouse),
            "--no-ingest",
        ]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "de_lu" in out and "tr" in out
    assert (
        pipeline.main(
            [
                "--market",
                "de_lu",
                "--date",
                "2026-11-02",
                "--models",
                "mean",
                "--warehouse",
                str(warehouse),
            ]
        )
        == 0
    )


def test_cli_rejects_unknown_market():
    with pytest.raises(SystemExit):
        pipeline.main(["--market", "fr"])
