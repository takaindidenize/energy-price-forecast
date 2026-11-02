import pandas as pd
import pytest

from price_forecast import store


def forecast_rows(pred: float, tz=None) -> pd.DataFrame:
    hours = pd.date_range("2026-11-02 23:00", periods=24, freq="h", tz=tz)
    return pd.DataFrame(
        {
            "market": "de_lu",
            "model": "m",
            "issued_at": pd.Timestamp("2026-11-02 10:00", tz=tz),
            "delivery_hour": hours,
            "price_eur_pred": pred,
        }
    )


def test_schema_matches_contract(tmp_path):
    with store.session(tmp_path / "w.duckdb") as con:
        cols = con.execute("DESCRIBE forecasts").df().set_index("column_name")["column_type"]
        assert cols.to_dict() == {
            "market": "VARCHAR",
            "model": "VARCHAR",
            "issued_at": "TIMESTAMP",
            "delivery_hour": "TIMESTAMP",
            "price_eur_pred": "DOUBLE",
        }
        cols = con.execute("DESCRIBE backtest_results").df()["column_name"].tolist()
        assert cols == ["market", "model", "delivery_hour", "price_eur", "price_eur_pred", "run_id"]


def test_forecast_upsert_is_idempotent(tmp_path):
    with store.session(tmp_path / "w.duckdb") as con:
        assert store.write_forecasts(con, forecast_rows(1.0)) == 24
        assert store.write_forecasts(con, forecast_rows(2.0)) == 24
        rows = store.read_forecasts(con, "de_lu")
    assert len(rows) == 24
    assert set(rows["price_eur_pred"]) == {2.0}


def test_duplicate_keys_in_one_batch_keep_last(tmp_path):
    batch = pd.concat([forecast_rows(1.0), forecast_rows(3.0)])
    with store.session(tmp_path / "w.duckdb") as con:
        assert store.write_forecasts(con, batch) == 24
        assert set(store.read_forecasts(con)["price_eur_pred"]) == {3.0}


def test_aware_timestamps_are_stored_as_naive_utc(tmp_path):
    rows = forecast_rows(1.0, tz="UTC")
    rows["delivery_hour"] = rows["delivery_hour"].dt.tz_convert("Europe/Berlin")
    with store.session(tmp_path / "w.duckdb") as con:
        store.write_forecasts(con, rows)
        stored = store.read_forecasts(con)
    assert stored["delivery_hour"].dt.tz is None
    assert stored["delivery_hour"].min() == pd.Timestamp("2026-11-02 23:00")
    assert stored["issued_at"].iloc[0] == pd.Timestamp("2026-11-02 10:00")


def test_backtest_results_upsert_by_run(tmp_path):
    rows = pd.DataFrame(
        {
            "market": "tr",
            "model": "m",
            "delivery_hour": pd.date_range("2025-01-01", periods=3, freq="h"),
            "price_eur": [1.0, 2.0, 3.0],
            "price_eur_pred": [1.5, 2.5, 3.5],
        }
    )
    with store.session(tmp_path / "w.duckdb") as con:
        store.write_backtest_results(con, rows, run_id="a")
        store.write_backtest_results(con, rows, run_id="a")
        store.write_backtest_results(con, rows, run_id="b")
        counts = con.execute(
            "SELECT run_id, count(*) FROM backtest_results GROUP BY run_id ORDER BY run_id"
        ).fetchall()
        with pytest.raises(ValueError, match="run_id"):
            store.write_backtest_results(con, rows)
    assert counts == [("a", 3), ("b", 3)]


def test_missing_key_column_is_rejected(tmp_path):
    with store.session(tmp_path / "w.duckdb") as con, pytest.raises(ValueError, match="key"):
        store.write_forecasts(con, forecast_rows(1.0).drop(columns="issued_at"))


def test_readonly_on_missing_warehouse_is_empty_and_creates_nothing(tmp_path):
    path = tmp_path / "missing.duckdb"
    con = store.connect_readonly(path)
    assert con.execute("SELECT count(*) FROM forecasts").fetchone()[0] == 0
    con.close()
    assert not path.exists()
