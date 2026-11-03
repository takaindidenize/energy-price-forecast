from datetime import UTC, datetime

import pandas as pd

from price_forecast import report, store


def _hours(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h")


def _fill(warehouse) -> None:
    hours = _hours("2026-11-01 23:00", 72)  # 2–4 November, Berlin time
    rows = []
    for i, h in enumerate(hours):
        issued = pd.Timestamp("2026-11-01 10:00") if i < 48 else pd.Timestamp("2026-11-03 10:00")
        rows.append(("de_lu", "lear_bounded", issued, h, 100.0 + i))
        # An older forecast for the same hour must be ignored in favour of the latest one.
        rows.append(("de_lu", "lear_bounded", issued - pd.Timedelta(days=1), h, -999.0))
        rows.append(("de_lu", "seasonal_naive", issued, h, 50.0))
    forecasts = pd.DataFrame(
        rows, columns=["market", "model", "issued_at", "delivery_hour", "price_eur_pred"]
    )
    realised = pd.DataFrame(
        {
            "market": "de_lu",
            "delivery_hour": hours[:48],
            "price_eur": [110.0 + i for i in range(48)],
            "loaded_at": pd.Timestamp("2026-11-03 12:00"),
        }
    )
    status = pd.DataFrame(
        [
            ("de_lu", m, "ok", False, 0, 3, 30, 2, mae, mae / 20, "2026-11-03", "2026-11-03")
            for m, mae in (("lear_bounded", 10.0), ("seasonal_naive", 20.0))
        ],
        columns=[
            "market",
            "model",
            "status",
            "degraded",
            "worse_streak_days",
            "degraded_after_days",
            "window_days",
            "window_n_days",
            "window_mae",
            "window_rmae",
            "evaluated_through",
            "updated_at",
        ],
    )
    status["evaluated_through"] = pd.to_datetime(status["evaluated_through"]).dt.date
    status["updated_at"] = pd.to_datetime(status["updated_at"])
    with store.session(warehouse) as con:
        store.write_forecasts(con, forecasts)
        store.upsert(con, "realised_prices", realised)
        store.upsert(con, "model_status", status)


def test_report_shows_tomorrow_history_and_accuracy(tmp_path):
    warehouse = tmp_path / "w.duckdb"
    _fill(warehouse)
    path = report.build(tmp_path / "site", warehouse, now=datetime(2026, 11, 3, 12, tzinfo=UTC))
    page = path.read_text(encoding="utf-8")

    assert "Germany-Luxembourg" in page
    assert "Turkey" not in page.split("<main>")[1].split("<footer>")[0].split("</p>", 1)[1]
    assert "delivery day Wednesday 04 November 2026" in page
    assert "159.5 EUR/MWh" in page  # mean of 148..171, the latest forecasts for 4 November
    assert "171.0 at 23:00" in page and "148.0 at 00:00" in page
    assert "10.0 (naive 20.0)" in page
    assert "-999" not in page
    assert page.count("<svg") == 1
    assert "Updated 2026-11-03 12:00 UTC" in page


def test_report_without_forecasts(tmp_path):
    path = report.build(tmp_path / "site", tmp_path / "empty.duckdb")
    assert "No forecasts yet." in path.read_text(encoding="utf-8")


def test_nice_ticks_cover_the_range():
    ticks = report._nice_ticks(-12.0, 87.0)
    assert ticks[0] <= -12.0 and ticks[-1] >= 87.0
    assert len(set(round(b - a, 6) for a, b in zip(ticks, ticks[1:], strict=False))) == 1


def test_latest_svg_is_a_standalone_image(tmp_path):
    import xml.dom.minidom

    warehouse = tmp_path / "w.duckdb"
    _fill(warehouse)
    report.build(tmp_path / "site", warehouse, now=datetime(2026, 11, 3, 12, tzinfo=UTC))
    svg = (tmp_path / "site" / "latest.svg").read_text(encoding="utf-8")

    doc = xml.dom.minidom.parseString(svg)  # well-formed, so browsers render it as an <img>
    assert doc.documentElement.getAttribute("xmlns") == "http://www.w3.org/2000/svg"
    assert "<style>" in svg and "prefers-color-scheme: dark" in svg  # no external CSS
    assert "Germany-Luxembourg" in svg
    assert "Tomorrow, 04 Nov: 159.5 EUR/MWh" in svg
    assert "Updated 2026-11-03 12:00 UTC" in svg


def test_page_charts_carry_the_hourly_readout(tmp_path):
    """Hovering or tapping a chart shows the hour's date, realised and forecast price."""
    import html as html_lib
    import json
    import re

    warehouse = tmp_path / "w.duckdb"
    _fill(warehouse)
    page = report.build(tmp_path / "site", warehouse).read_text(encoding="utf-8")
    svg = (tmp_path / "site" / "latest.svg").read_text(encoding="utf-8")

    points = json.loads(html_lib.unescape(re.search(r'data-points="([^"]*)"', page).group(1)))
    heights = json.loads(html_lib.unescape(re.search(r'data-y="([^"]*)"', page).group(1)))
    assert len(points) == len(heights) == 72
    assert [p["x"] for p in points] == sorted(p["x"] for p in points)
    assert points[0] == {"x": points[0]["x"], "t": "Mon 02 Nov, 00:00", "a": "110.0", "f": "100.0"}
    assert points[-1]["t"] == "Wed 04 Nov, 23:00" and points[-1]["a"] == "–"
    assert heights[-1]["ya"] is None and heights[-1]["yf"] is not None
    assert 'class="cross"' in page and "<script>" in page
    assert "data-points" not in svg and "<script" not in svg  # the README image stays static
