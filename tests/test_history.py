import pandas as pd
import pytest

from price_forecast import history, store


def _forecasts(prices: list[float], issued: str = "2026-11-03 10:00") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": "de_lu",
            "model": "median_bounded",
            "issued_at": pd.Timestamp(issued),
            "delivery_hour": pd.date_range("2026-11-03 23:00", periods=len(prices), freq="h"),
            "price_eur_pred": prices,
        }
    )


def _runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": ["de_lu", "tr"],
            "issue_date": pd.to_datetime(["2026-11-03", "2026-11-03"]).date,
            "status": ["ok", "failed"],
            "started_at": pd.to_datetime(["2026-11-03 10:20", "2026-11-03 10:21"]),
            "finished_at": pd.to_datetime(["2026-11-03 10:24", "2026-11-03 10:22"]),
            "models_ok": ["seasonal_naive,median_bounded", ""],  # empty, not missing
            "models_failed": ["", "median_bounded"],
            "n_forecasts": [48, 0],
            "error": [None, 'EPİAŞ said "no", twice, with a comma'],
        }
    )


def _fill(warehouse, forecasts: pd.DataFrame, runs: pd.DataFrame | None = None) -> None:
    with store.session(warehouse) as con:
        store.write_forecasts(con, forecasts)
        if runs is not None:
            store.upsert(con, "pipeline_runs", runs)


def _table(warehouse, table: str) -> pd.DataFrame:
    with store.session(warehouse) as con:
        keys = ", ".join(store.KEYS[table])
        return con.execute(f"SELECT * FROM {table} ORDER BY {keys}").df()


# Values that a careless text format would round or mangle.
AWKWARD = [0.1 + 0.2, -12.345678901234567, 1e-17, 3400.0 / 37.3]


def test_round_trip_is_exact(tmp_path):
    source, target = tmp_path / "a.duckdb", tmp_path / "b.duckdb"
    _fill(source, _forecasts(AWKWARD), _runs())
    history.export(tmp_path / "h", source)
    added = history.restore(tmp_path / "h", target)
    assert added == {"forecasts": 4, "pipeline_runs": 2}
    for table in history.TABLES:
        pd.testing.assert_frame_equal(_table(source, table), _table(target, table))
    runs = _table(target, "pipeline_runs")
    assert runs["models_ok"].tolist() == ["seasonal_naive,median_bounded", ""]
    assert runs["error"].isna().tolist() == [True, False]


def test_export_never_shrinks_the_history(tmp_path):
    out = tmp_path / "h"
    old = tmp_path / "old.duckdb"
    _fill(old, _forecasts([1.0, 2.0]), _runs())
    history.export(out, old)
    before = (out / "forecasts.csv").read_text()

    # A run that starts with an empty cache, or none at all, must not erase anything.
    assert history.export(out, tmp_path / "missing.duckdb")["forecasts"] == 2
    empty = tmp_path / "empty.duckdb"
    store.connect(empty).close()
    history.export(out, empty)
    assert (out / "forecasts.csv").read_text() == before


def test_export_adds_new_rows_and_the_warehouse_wins_on_a_key(tmp_path):
    out = tmp_path / "h"
    first = tmp_path / "first.duckdb"
    _fill(first, _forecasts([1.0, 2.0]))
    history.export(out, first)

    second = tmp_path / "second.duckdb"
    rerun = _forecasts([9.0])  # same key as the first hour, new value
    later = _forecasts([5.0, 6.0], issued="2026-11-04 10:00")
    _fill(second, pd.concat([rerun, later]))
    counts = history.export(out, second)

    assert counts["forecasts"] == 4
    restored = tmp_path / "restored.duckdb"
    history.restore(out, restored)
    prices = _table(restored, "forecasts")["price_eur_pred"].tolist()
    assert prices == [9.0, 2.0, 5.0, 6.0]


def test_restore_keeps_rows_the_warehouse_already_has(tmp_path):
    out = tmp_path / "h"
    old = tmp_path / "old.duckdb"
    _fill(old, _forecasts([1.0, 2.0]))
    history.export(out, old)

    current = tmp_path / "current.duckdb"
    _fill(current, _forecasts([7.0]))
    assert history.restore(out, current)["forecasts"] == 1
    assert _table(current, "forecasts")["price_eur_pred"].tolist() == [7.0, 2.0]


def test_files_are_sorted_and_stable(tmp_path):
    warehouse = tmp_path / "w.duckdb"
    later = _forecasts([5.0], issued="2026-11-04 10:00")
    _fill(warehouse, pd.concat([later, _forecasts([1.0, 2.0])]), _runs())
    history.export(tmp_path / "a", warehouse)
    history.export(tmp_path / "b", warehouse)
    history.export(tmp_path / "b", warehouse)  # exporting again changes nothing
    for table in history.TABLES:
        a = (tmp_path / "a" / f"{table}.csv").read_bytes()
        assert a == (tmp_path / "b" / f"{table}.csv").read_bytes()
    lines = (tmp_path / "a" / "forecasts.csv").read_text().splitlines()
    assert lines[0].startswith("market,model,issued_at,delivery_hour")
    assert [line.split(",")[2] for line in lines[1:]] == sorted(
        line.split(",")[2] for line in lines[1:]
    )


def test_restore_without_files_adds_nothing(tmp_path):
    added = history.restore(tmp_path / "nothing", tmp_path / "w.duckdb")
    assert added == {"forecasts": 0, "pipeline_runs": 0}


@pytest.mark.parametrize("command", ["export", "restore"])
def test_command_line(tmp_path, command):
    warehouse = tmp_path / "w.duckdb"
    _fill(warehouse, _forecasts([1.0]))
    history.main(["export", "--out", str(tmp_path / "h"), "--warehouse", str(warehouse)])
    if command == "restore":
        target = tmp_path / "t.duckdb"
        history.main(["restore", "--from", str(tmp_path / "h"), "--warehouse", str(target)])
        assert len(_table(target, "forecasts")) == 1
    assert (tmp_path / "h" / "README.md").exists()
