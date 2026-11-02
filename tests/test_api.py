from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from price_forecast import api, monitor, pipeline, store


def forecast_rows(market: str, model: str, day: date, offset: float = 0.0) -> pd.DataFrame:
    hours = pipeline.delivery_hours_utc(market, day)
    return pd.DataFrame(
        {
            "market": market,
            "model": model,
            "issued_at": pipeline.issue_time_utc(market, day - timedelta(days=1)),
            "delivery_hour": hours,
            "price_eur_pred": 50.0 + offset + np.arange(len(hours)),
        }
    )


def realised(start="2026-10-01", end="2026-10-22") -> pd.DataFrame:
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    rng = np.random.default_rng(1)
    return pd.DataFrame({"delivery_hour": hours, "price_eur": 50 + rng.normal(0, 10, len(hours))})


@pytest.fixture
def warehouse(tmp_path):
    path = tmp_path / "w.duckdb"
    rows = [
        forecast_rows("de_lu", "seasonal_naive", date(2026, 10, 25)),  # 25 hours
        forecast_rows("de_lu", "lightgbm", date(2026, 10, 25), offset=1),
        forecast_rows("de_lu", "lightgbm", date(2026, 3, 29)),  # 23 hours
        forecast_rows("tr", "lightgbm", date(2026, 3, 29)),  # Turkey has no DST
    ]
    # Accuracy history for de_lu: lightgbm good, seasonal_naive copy off by 30.
    for day in [date(2026, 10, 10) + timedelta(days=i) for i in range(11)]:
        hours = pipeline.delivery_hours_utc("de_lu", day)
        prices = realised().set_index("delivery_hour")["price_eur"].reindex(hours).to_numpy()
        for model, err in (("lightgbm", 1.0), ("stale", 30.0)):
            rows.append(
                pd.DataFrame(
                    {
                        "market": "de_lu",
                        "model": model,
                        "issued_at": pipeline.issue_time_utc("de_lu", day - timedelta(days=1)),
                        "delivery_hour": hours,
                        "price_eur_pred": prices + err,
                    }
                )
            )
    with store.session(path) as con:
        store.write_forecasts(con, pd.concat(rows, ignore_index=True))
        store.upsert(
            con,
            "pipeline_runs",
            pd.DataFrame(
                [
                    {
                        "market": "tr",
                        "issue_date": date(2026, 3, 28),
                        "status": "failed",
                        "started_at": pd.Timestamp("2026-03-28 08:00"),
                        "finished_at": pd.Timestamp("2026-03-28 08:01"),
                        "models_ok": "",
                        "models_failed": "",
                        "n_forecasts": 0,
                        "error": "RuntimeError: EPIAS_USERNAME is not set",
                    }
                ]
            ),
        )
    monitor.run_market(
        "de_lu",
        monitor.MonitorSettings(30, 3),
        load_realised=lambda m: realised(),
        warehouse=path,
    )
    return path


@pytest.fixture
def client(warehouse):
    return TestClient(api.create_app(warehouse))


def test_health_and_markets(client):
    assert client.get("/health").json() == {"status": "ok", "warehouse_exists": True}
    markets = client.get("/markets").json()
    assert {m["code"] for m in markets} == {"de_lu", "tr"}
    de = next(m for m in markets if m["code"] == "de_lu")
    assert de["timezone"] == "Europe/Berlin" and len(de["weather_points"]) == 4


def test_forecast_on_fall_back_day_has_25_hours(client):
    body = client.get("/forecast/de_lu", params={"date": "2026-10-25", "model": "lightgbm"}).json()
    assert body["expected_hours"] == 25
    (fc,) = body["forecasts"]
    assert fc["model"] == "lightgbm" and fc["n_hours"] == 25
    values = fc["values"]
    assert values[0]["delivery_hour_utc"] == "2026-10-24T22:00:00Z"
    assert values[0]["delivery_hour_local"] == "2026-10-25T00:00:00+02:00"
    # Local 02:00 appears twice, once in CEST and once in CET.
    assert [v["hour_local"] for v in values].count(2) == 2
    assert values[-1]["delivery_hour_local"] == "2026-10-25T23:00:00+01:00"
    assert fc["issued_at"] == "2026-10-24T09:00:00Z"


def test_forecast_on_spring_forward_day_has_23_hours(client):
    body = client.get("/forecast/de_lu", params={"date": "2026-03-29"}).json()
    (fc,) = body["forecasts"]
    assert body["expected_hours"] == 23 and fc["n_hours"] == 23
    assert 2 not in [v["hour_local"] for v in fc["values"]]


def test_turkey_has_24_hours_on_european_dst_day(client):
    body = client.get("/forecast/tr", params={"date": "2026-03-29"}).json()
    assert body["forecasts"][0]["n_hours"] == 24
    assert body["forecasts"][0]["values"][0]["delivery_hour_local"] == "2026-03-29T00:00:00+03:00"


def test_forecast_all_models_and_actuals(client):
    body = client.get("/forecast/de_lu", params={"date": "2026-10-15"}).json()
    models = [f["model"] for f in body["forecasts"]]
    assert models == ["lightgbm", "stale"]
    assert all(v["price_eur_actual"] is not None for v in body["forecasts"][0]["values"])
    body = client.get("/forecast/de_lu", params={"date": "2026-10-25"}).json()
    assert body["forecasts"][0]["values"][0]["price_eur_actual"] is None


def test_forecast_defaults_to_latest_day(client):
    body = client.get("/forecast/de_lu").json()
    assert body["delivery_date"] == "2026-10-25"


@pytest.mark.parametrize(
    "path, params",
    [
        ("/forecast/fr", {}),
        ("/forecast/de_lu", {"date": "2026-01-01"}),
        ("/forecast/de_lu", {"date": "2026-10-25", "model": "prophet"}),
        ("/accuracy/fr", {}),
        ("/accuracy/tr", {}),
    ],
)
def test_not_found(client, path, params):
    response = client.get(path, params=params)
    assert response.status_code == 404
    assert response.json()["detail"]


@pytest.mark.parametrize(
    "path, params",
    [
        ("/forecast/de_lu", {"date": "2026-13-01"}),
        ("/forecast/de_lu", {"date": "tomorrow"}),
        ("/forecast/de_lu", {"model": "DROP TABLE"}),
        ("/accuracy/de_lu", {"days": 0}),
        ("/accuracy/de_lu", {"days": 1000}),
    ],
)
def test_invalid_input(client, path, params):
    assert client.get(path, params=params).status_code == 422


def test_accuracy(client):
    body = client.get("/accuracy/de_lu", params={"days": 5}).json()
    assert body["date_to"] == "2026-10-20" and body["date_from"] == "2026-10-16"
    by_model = {m["model"]: m for m in body["models"]}
    assert by_model["lightgbm"]["n_days"] == 5
    assert by_model["lightgbm"]["mae"] == pytest.approx(1.0)
    assert by_model["lightgbm"]["rmse"] == pytest.approx(1.0)
    assert by_model["lightgbm"]["rmae"] < 1 < by_model["stale"]["rmae"]
    assert len(by_model["stale"]["daily"]) == 5


def test_status(client):
    body = client.get("/status").json()
    assert body["status"] == "degraded"
    markets = {m["market"]: m for m in body["markets"]}
    de = {m["model"]: m for m in markets["de_lu"]["models"]}
    assert de["stale"]["degraded"] and de["stale"]["status"] == "degraded"
    assert not de["lightgbm"]["degraded"]
    assert markets["de_lu"]["status"] == "degraded"
    assert markets["tr"]["status"] == "failed"
    assert markets["tr"]["last_run"]["error"].startswith("RuntimeError")


def test_empty_warehouse(tmp_path):
    client = TestClient(api.create_app(tmp_path / "none.duckdb"))
    assert client.get("/health").json()["warehouse_exists"] is False
    assert client.get("/forecast/de_lu").status_code == 404
    assert client.get("/status").json()["status"] == "no_data"
    assert not (tmp_path / "none.duckdb").exists()


def test_openapi_docs(client):
    schema = client.get("/openapi.json").json()
    assert {"/health", "/markets", "/forecast/{market}", "/accuracy/{market}", "/status"} <= set(
        schema["paths"]
    )
    assert client.get("/docs").status_code == 200
