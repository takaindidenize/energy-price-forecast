"""Keep the live forecast history outside the Actions cache.

The daily job carries `data/` (raw data and the warehouse) between runs in the Actions cache. Raw
data can be downloaded again, but the forecasts made each day cannot be made again as they were. So
after each run the irreplaceable tables are written as CSV to the repository's `data` branch, and a
run that starts with an empty cache restores them from there.

- `export` merges: rows already in the files stay, rows from the warehouse are added or replace
  rows with the same key. The history can grow, never shrink, so a run that starts empty cannot
  erase it.
- `restore` never overwrites: rows already in the warehouse win.

Files are sorted by key, so a day's commit only adds lines.

    python -m price_forecast.history export --out <dir>
    python -m price_forecast.history restore --from <dir>
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import duckdb

from price_forecast import config, store

log = logging.getLogger(__name__)

# The other tables are rebuilt by the monitor from raw data and these.
TABLES = ("forecasts", "pipeline_runs")

README = """# Forecast history

Written by the daily job of https://github.com/takaindidenize/energy-price-forecast
(`python -m price_forecast.history`). One CSV per warehouse table, sorted by key, timestamps in
naive UTC. `forecasts` holds every live forecast as it was made; `pipeline_runs` holds when each
daily run finished. The job restores these when its cache is empty.
"""


# A missing value is written as \N, so it cannot be confused with an empty string.
_NULL = "'\\N'"


def _quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _read_csv(con: duckdb.DuckDBPyConnection, table: str, path: Path) -> str:
    """A `read_csv` call that reads `path` with exactly the table's column types."""
    types = con.execute(f"DESCRIBE {table}").fetchall()
    columns = ", ".join(f"'{name}': '{dtype}'" for name, dtype, *_ in types)
    return (
        f"read_csv({_quote(path)}, header = true, auto_detect = false, nullstr = {_NULL}, "
        f"columns = {{{columns}}})"
    )


def _count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def export(out_dir: Path, warehouse: Path | None = None) -> dict[str, int]:
    """Merge the warehouse's history tables into the CSV files in `out_dir`. Returns row counts."""
    warehouse = Path(warehouse) if warehouse is not None else config.WAREHOUSE_PATH
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    store.ensure_schema(con)
    if warehouse.exists():
        con.execute(f"ATTACH {_quote(warehouse)} AS wh (READ_ONLY)")
    else:
        log.warning("history: no warehouse at %s, keeping the files as they are", warehouse)
    counts = {}
    for table in TABLES:
        path = out_dir / f"{table}.csv"
        columns = ", ".join(store.table_columns(con, table))
        if path.exists():
            con.execute(f"INSERT INTO {table} SELECT {columns} FROM {_read_csv(con, table, path)}")
        before = _count(con, table)
        if warehouse.exists():
            con.execute(
                f"INSERT OR REPLACE INTO {table} ({columns}) SELECT {columns} FROM wh.{table}"
            )
        counts[table] = _count(con, table)
        order = ", ".join(store.KEYS[table])
        tmp = path.with_suffix(".csv.tmp")
        con.execute(
            f"COPY (SELECT {columns} FROM {table} ORDER BY {order}) TO {_quote(tmp)} "
            f"(HEADER, DELIMITER ',', NULL {_NULL})"
        )
        tmp.replace(path)
        log.info("history: %s has %d rows (%+d)", table, counts[table], counts[table] - before)
    (out_dir / "README.md").write_text(README, encoding="utf-8")
    con.close()
    return counts


def restore(in_dir: Path, warehouse: Path | None = None) -> dict[str, int]:
    """Add the rows in `in_dir`'s CSV files that the warehouse does not have. Returns rows added."""
    added = {}
    with store.session(warehouse) as con:
        for table in TABLES:
            path = in_dir / f"{table}.csv"
            if not path.exists():
                log.warning("history: %s not found, nothing restored for %s", path, table)
                added[table] = 0
                continue
            columns = ", ".join(store.table_columns(con, table))
            before = _count(con, table)
            con.execute(
                f"INSERT OR IGNORE INTO {table} ({columns}) "
                f"SELECT {columns} FROM {_read_csv(con, table, path)}"
            )
            added[table] = _count(con, table) - before
            log.info("history: restored %d rows into %s", added[table], table)
    return added


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="merge the warehouse history into CSV files")
    exp.add_argument("--out", type=Path, required=True)
    res = sub.add_parser("restore", help="add missing history rows to the warehouse")
    res.add_argument("--from", dest="source", type=Path, required=True)
    for p in (exp, res):
        p.add_argument("--warehouse", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "export":
        export(args.out, args.warehouse)
    else:
        restore(args.source, args.warehouse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
