"""A static HTML page with the latest forecasts and their accuracy, for GitHub Pages.

Reads the warehouse (forecasts, realised prices, model status) and writes `<out>/index.html`: per
market, the latest forecast hour by hour, the last `history_days` days of forecast against the
realised price, and the rolling accuracy against the seasonal naive. Inline SVG and one small
script, no external assets, light and dark themes.

    python -m price_forecast.report --out site
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from price_forecast import config, store

log = logging.getLogger("price_forecast.report")

MODEL = "lear_bounded"
BASELINE = "seasonal_naive"
HISTORY_DAYS = 14
REPO_URL = "https://github.com/takaindidenize/energy-price-forecast"

_W, _H = 760, 240
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 48, 12, 12, 28


def latest_forecasts(con, model: str) -> pd.DataFrame:
    """The most recently issued forecast of `model` for every market and delivery hour."""
    return con.execute(
        """
        SELECT market, delivery_hour, price_eur_pred, issued_at
        FROM forecasts
        WHERE model = ?
        QUALIFY row_number() OVER (
            PARTITION BY market, delivery_hour ORDER BY issued_at DESC) = 1
        """,
        [model],
    ).df()


def _local(values: pd.Series, tz: str) -> pd.Series:
    return pd.to_datetime(values).dt.tz_localize("UTC").dt.tz_convert(tz)


def market_view(
    market: str,
    forecasts: pd.DataFrame,
    realised: pd.DataFrame,
    status: pd.DataFrame,
    history_days: int = HISTORY_DAYS,
    runs: pd.DataFrame | None = None,
) -> dict | None:
    """Everything one market's section shows, or None when it has no forecasts yet."""
    tz = config.MARKETS[market].timezone
    f = forecasts[forecasts["market"] == market].copy()
    if f.empty:
        return None
    f["local"] = _local(f["delivery_hour"], tz)
    next_day = f["local"].dt.date.max()
    first_day = next_day - timedelta(days=history_days)
    f = f[f["local"].dt.date >= first_day].sort_values("local")

    r = realised[realised["market"] == market].copy()
    r["local"] = _local(r["delivery_hour"], tz)
    r = r[r["local"].dt.date >= first_day].sort_values("local")

    tomorrow = f[f["local"].dt.date == next_day]
    joined = f.merge(r[["delivery_hour", "price_eur"]], on="delivery_hour", how="inner")
    st = status[status["market"] == market].set_index("model")
    computed_at = None
    if runs is not None and not runs.empty:
        run = runs[
            (runs["market"] == market)
            & (pd.to_datetime(runs["issue_date"]).dt.date == next_day - timedelta(days=1))
        ]
        computed_at = pd.to_datetime(run["finished_at"]).max() if not run.empty else None
    return {
        "market": market,
        "name": config.MARKETS[market].name,
        "timezone": tz,
        "next_day": next_day,
        "tomorrow": tomorrow,
        "next_realised": r[r["local"].dt.date == next_day],
        "computed_at": computed_at,
        "forecasts": f,
        "realised": r,
        "history_mae": (joined["price_eur_pred"] - joined["price_eur"]).abs().mean()
        if not joined.empty
        else None,
        "history_days_scored": joined["local"].dt.date.nunique() if not joined.empty else 0,
        "status": st,
    }


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = max(hi - lo, 1e-9)
    raw = span / n
    magnitude = 10 ** int(f"{raw:e}".split("e")[1])
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), default=raw)
    ticks = [(lo // step) * step]
    while ticks[-1] < hi - 1e-9:  # the last tick must reach the highest value
        ticks.append(ticks[-1] + step)
    return [round(t, 6) for t in ticks]


def chart_svg(view: dict, interactive: bool = False, one_day: bool = False) -> str:
    """The chart as an inline SVG.

    `one_day` draws only the latest forecast day, with an hourly axis.

    `interactive` adds what the page's script needs for the hover/tap readout: the hourly points
    as JSON in `data-points` and a hidden crosshair. The README image (`latest.svg`) is shown as
    a plain <img>, where no script runs, so it is drawn without them.
    """
    f, r = view["forecasts"], view["realised"]
    if one_day:
        f, r = view["tomorrow"], view["next_realised"]
    times = pd.concat([f["local"], r["local"]])
    t0, t1 = times.min(), times.max() + pd.Timedelta(hours=1)
    values = pd.concat([f["price_eur_pred"], r["price_eur"]]).dropna()
    ticks = _nice_ticks(min(0.0, values.min()), values.max())
    y0, y1 = ticks[0], ticks[-1]

    def x(t: pd.Timestamp) -> float:
        return _PAD_L + (t - t0) / (t1 - t0) * (_W - _PAD_L - _PAD_R)

    def y(v: float) -> float:
        return _PAD_T + (y1 - v) / (y1 - y0) * (_H - _PAD_T - _PAD_B)

    def path(times: pd.Series, vals: pd.Series) -> str:
        points = [f"{x(t):.1f},{y(v):.1f}" for t, v in zip(times, vals, strict=True) if pd.notna(v)]
        return "M" + " L".join(points) if points else ""

    data = ""
    if interactive:
        points = _hourly_points(f, r)
        coords = [
            {"x": round(x(t), 1), "t": f"{t:%a %d %b, %H:%M}", "a": _fmt(a), "f": _fmt(p)}
            for t, a, p in points
        ]
        heights = [{"ya": _y_or_none(y, a), "yf": _y_or_none(y, p)} for _, a, p in points]
        data = (
            f' data-points="{html.escape(json.dumps(coords, ensure_ascii=False))}"'
            f' data-y="{html.escape(json.dumps(heights))}"'
        )
    parts = [
        f'<svg viewBox="0 0 {_W} {_H}" role="img" aria-label="Forecast and realised price, '
        f'{html.escape(view["name"])}"{data}>'
    ]
    if not one_day:
        next_start = pd.Timestamp(view["next_day"]).tz_localize(view["timezone"])
        parts.append(
            f'<rect class="next" x="{x(next_start):.1f}" y="{_PAD_T}" '
            f'width="{x(t1) - x(next_start):.1f}" height="{_H - _PAD_T - _PAD_B}"/>'
        )
    for v in ticks:
        parts.append(
            f'<line class="grid" x1="{_PAD_L}" x2="{_W - _PAD_R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
            f'<text class="tick" x="{_PAD_L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>'
        )
    tick = t0.normalize()
    step, label = (pd.Timedelta(hours=3), "%H:%M") if one_day else (pd.Timedelta(days=2), "%d %b")
    while tick < t1:  # the end of the axis has no label of its own: it would be cut off
        if tick >= t0:
            parts.append(
                f'<text class="tick" x="{x(tick):.1f}" y="{_H - 8}" text-anchor="middle">'
                f"{tick.strftime(label)}</text>"
            )
        tick += step
    parts.append(f'<path class="actual" d="{path(r["local"], r["price_eur"])}"/>')
    parts.append(f'<path class="forecast" d="{path(f["local"], f["price_eur_pred"])}"/>')
    if interactive:
        parts.append(
            f'<line class="cross" x1="0" x2="0" y1="{_PAD_T}" y2="{_H - _PAD_B}"/>'
            f'<circle class="dot actual-dot" r="4" cx="-10" cy="-10"/>'
            f'<circle class="dot forecast-dot" r="4" cx="-10" cy="-10"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _y_or_none(y, value) -> float | None:
    return None if pd.isna(value) else round(y(value), 1)


def _hourly_points(f: pd.DataFrame, r: pd.DataFrame) -> list[tuple[pd.Timestamp, float, float]]:
    """(local hour, realised, forecast) for every hour that has either, oldest first."""
    joined = pd.merge(
        r[["local", "price_eur"]],
        f[["local", "price_eur_pred"]],
        on="local",
        how="outer",
    ).sort_values("local")
    return list(zip(joined["local"], joined["price_eur"], joined["price_eur_pred"], strict=True))


def _fmt(v, digits: int = 1) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:,.{digits}f}"


def section_html(view: dict) -> str:
    t = view["tomorrow"]
    st = view["status"]
    peak = t.loc[t["price_eur_pred"].idxmax()] if not t.empty else None
    low = t.loc[t["price_eur_pred"].idxmin()] if not t.empty else None
    model_row = st.loc[MODEL] if MODEL in st.index else None
    base_row = st.loc[BASELINE] if BASELINE in st.index else None
    degraded = model_row is not None and bool(model_row["degraded"])
    status_text = (
        "degraded: worse than the seasonal naive for several days"
        if degraded
        else ("ok" if model_row is not None else "no accuracy yet")
    )
    stats = [
        ("Average", f"{_fmt(t['price_eur_pred'].mean())} EUR/MWh"),
        (
            "Highest hour",
            f"{_fmt(peak['price_eur_pred'])} at {peak['local']:%H:%M}" if peak is not None else "–",
        ),
        (
            "Lowest hour",
            f"{_fmt(low['price_eur_pred'])} at {low['local']:%H:%M}" if low is not None else "–",
        ),
        (
            f"MAE, last {int(model_row['window_n_days']) if model_row is not None else 0} days",
            f"{_fmt(model_row['window_mae'] if model_row is not None else None)} "
            f"(naive {_fmt(base_row['window_mae'] if base_row is not None else None)})",
        ),
    ]
    tiles = "".join(
        f'<div class="tile"><div class="label">{html.escape(k)}</div>'
        f'<div class="value">{html.escape(v)}</div></div>'
        for k, v in stats
    )
    status_html = html.escape(status_text)
    return f"""
<section>
  <h2>{html.escape(view["name"])}</h2>
  <h3>Forecast for {view["next_day"]:%A %d %B %Y}</h3>
  <p class="note">{html.escape(_forecast_note(view))}</p>
  <div class="tiles">{tiles}</div>
  <div class="chart">{chart_svg(view, True, one_day=True)}<div class="tip" hidden></div></div>
  {_hours_table(view)}
  <h3>Forecast against realised prices, last {HISTORY_DAYS} days</h3>
  <div class="chart">{chart_svg(view, interactive=True)}<div class="tip" hidden></div></div>
  <p class="legend"><span class="swatch actual"></span>Realised price
    <span class="swatch forecast"></span>Forecast (LEAR within the market's price limits)
    <span class="swatch next"></span>Latest forecast day · model status: {status_html}</p>
</section>"""


def _forecast_note(view: dict) -> str:
    computed = view["computed_at"]
    when = f"Computed {computed:%d %b %H:%M} UTC. " if computed is not None else ""
    if view["next_realised"].empty:
        return when + "Actual prices for this day were not published when this page was built."
    return (
        when + "Actual prices were already published when this page was built, because the daily "
        "job started late, so they are shown too. The forecast did not use them."
    )


def _hours_table(view: dict) -> str:
    """Every hour of the latest forecast day, for reading exact values."""
    points = _hourly_points(view["tomorrow"], view["next_realised"])
    show_actual = not view["next_realised"].empty
    head = "<th>Hour</th><th>Forecast</th>" + ("<th>Actual</th>" if show_actual else "")
    rows = "".join(
        f"<tr><td>{t:%H:%M}</td><td>{_fmt(p)}</td>"
        + (f"<td>{_fmt(a)}</td>" if show_actual else "")
        + "</tr>"
        for t, a, p in points
    )
    return (
        "<details><summary>All hours as a table (EUR/MWh)</summary>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></details>"
    )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Energy Price Forecast</title>
<style>
:root {{ --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e4e3df;
  --forecast:#2a78d6; --next:#2a78d614; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#383835;
    --forecast:#3987e5; --next:#3987e51f; }}
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:820px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:24px; margin:0 0 4px; }}
.sub {{ color:var(--ink2); margin:0 0 24px; }}
section {{ background:var(--surface); border:1px solid var(--grid); border-radius:12px;
  padding:16px; margin-bottom:20px; }}
h2 {{ font-size:18px; margin:0 0 12px; }}
h3 {{ font-size:15px; margin:20px 0 4px; }}
h2 + h3 {{ margin-top:0; }}
.note {{ color:var(--ink2); font-size:13px; margin:0 0 12px; }}
details {{ margin-top:8px; font-size:13px; }}
summary {{ cursor:pointer; color:var(--ink2); }}
table {{ border-collapse:collapse; margin-top:8px; font-variant-numeric:tabular-nums; }}
th, td {{ padding:2px 20px 2px 0; text-align:right; border-bottom:1px solid var(--grid); }}
th {{ color:var(--ink2); font-weight:500; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:8px;
  margin-bottom:12px; }}
.tile .label {{ color:var(--ink2); font-size:12px; }}
.tile .value {{ font-size:17px; font-weight:600; font-variant-numeric:tabular-nums; }}
svg {{ width:100%; height:auto; display:block; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.tick {{ fill:var(--ink2); font-size:11px; }}
.actual {{ fill:none; stroke:var(--ink); stroke-width:2; }}
.forecast {{ fill:none; stroke:var(--forecast); stroke-width:2; }}
.next {{ fill:var(--next); }}
.chart {{ position:relative; }}
.chart svg {{ touch-action:pan-y; cursor:crosshair; }}
.cross {{ stroke:var(--ink2); stroke-width:1; stroke-dasharray:3 3; visibility:hidden; }}
.dot {{ stroke:var(--surface); stroke-width:2; }}
.actual-dot {{ fill:var(--ink); }} .forecast-dot {{ fill:var(--forecast); }}
.tip {{ position:absolute; top:4px; pointer-events:none; background:var(--surface);
  border:1px solid var(--grid); border-radius:8px; padding:6px 10px; font-size:12px;
  line-height:1.5; white-space:nowrap; box-shadow:0 2px 8px #0000001f;
  font-variant-numeric:tabular-nums; }}
.tip b {{ display:block; color:var(--ink); }}
.tip .k {{ display:inline-block; width:10px; height:3px; vertical-align:middle;
  margin-right:6px; }}
.legend {{ color:var(--ink2); font-size:13px; margin:8px 0 0; }}
.swatch {{ display:inline-block; width:14px; height:3px; vertical-align:middle;
  margin:0 4px 0 10px; }}
.swatch.actual {{ background:var(--ink); }} .swatch.forecast {{ background:var(--forecast); }}
.swatch.next {{ background:var(--next); height:10px; border:1px solid var(--grid); }}
footer {{ color:var(--ink2); font-size:13px; }}
a {{ color:var(--forecast); }}
</style>
</head>
<body><main>
<h1>Day-ahead electricity price forecast</h1>
<p class="sub">Germany-Luxembourg and Turkey, EUR/MWh, local time. Forecasts are made each morning
before the day-ahead auction closes, from that day's prices and weather forecasts.</p>
{sections}
<footer>Updated {updated} UTC · <a href="{repo}">source and method</a> · Data: Energy-Charts
(Bundesnetzagentur | SMARD.de, CC BY 4.0), EPİAŞ Transparency Platform, Open-Meteo.</footer>
</main>
<script>
// Hover or tap a chart: crosshair on the nearest hour, with its date, realised and forecast price.
document.querySelectorAll(".chart").forEach(function (box) {{
  var svg = box.querySelector("svg"), tip = box.querySelector(".tip");
  var cross = svg.querySelector(".cross"), da = svg.querySelector(".actual-dot");
  var df = svg.querySelector(".forecast-dot");
  var pts = JSON.parse(svg.dataset.points), ys = JSON.parse(svg.dataset.y);
  var width = svg.viewBox.baseVal.width;
  function place(dot, y, x) {{
    dot.setAttribute("cx", y === null ? -10 : x); dot.setAttribute("cy", y === null ? -10 : y);
  }}
  function show(e) {{
    var r = svg.getBoundingClientRect(), vx = (e.clientX - r.left) * width / r.width;
    var lo = 0, hi = pts.length - 1;
    while (hi - lo > 1) {{
      var mid = (lo + hi) >> 1;
      if (pts[mid].x < vx) lo = mid; else hi = mid;
    }}
    var i = Math.abs(pts[lo].x - vx) <= Math.abs(pts[hi].x - vx) ? lo : hi, p = pts[i];
    cross.setAttribute("x1", p.x); cross.setAttribute("x2", p.x);
    cross.style.visibility = "visible";
    place(da, ys[i].ya, p.x); place(df, ys[i].yf, p.x);
    tip.innerHTML = "<b>" + p.t + "</b>" +
      "<span class='k' style='background:var(--ink)'></span>Realised " +
      (p.a === "–" ? "not published yet" : p.a) + "<br>" +
      "<span class='k' style='background:var(--forecast)'></span>Forecast " + p.f +
      " <span style='color:var(--ink2)'>EUR/MWh</span>";
    tip.hidden = false;
    var px = p.x * r.width / width, w = tip.offsetWidth;
    tip.style.left = (px + 12 + w > r.width ? Math.max(0, px - 12 - w) : px + 12) + "px";
  }}
  function hide() {{
    tip.hidden = true; cross.style.visibility = "hidden"; place(da, null); place(df, null);
  }}
  svg.addEventListener("pointermove", show);
  svg.addEventListener("pointerdown", show);
  // A mouse leaving hides the readout; on touch screens lifting the finger also "leaves", so
  // there the readout stays until the next tap outside the chart.
  svg.addEventListener("pointerleave", function (e) {{ if (e.pointerType === "mouse") hide(); }});
  document.addEventListener("pointerdown", function (e) {{ if (!box.contains(e.target)) hide(); }});
}});
</script>
</body>
</html>
"""


def build(
    out_dir: Path,
    warehouse: Path | str | None = None,
    history_days: int = HISTORY_DAYS,
    now: datetime | None = None,
) -> Path:
    with store.session(warehouse) as con:
        forecasts = latest_forecasts(con, MODEL)
        realised = con.execute("SELECT market, delivery_hour, price_eur FROM realised_prices").df()
        status = con.execute("SELECT * FROM model_status").df()
        runs = con.execute("SELECT market, issue_date, finished_at FROM pipeline_runs").df()
    views = [
        v
        for m in config.MARKETS
        if (v := market_view(m, forecasts, realised, status, history_days, runs)) is not None
    ]
    sections = "".join(section_html(v) for v in views) or "<p>No forecasts yet.</p>"
    updated = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(
        PAGE.format(sections=sections, updated=updated, repo=REPO_URL), encoding="utf-8"
    )
    (out_dir / "latest.svg").write_text(latest_svg(views, updated), encoding="utf-8")
    log.info("wrote %s and latest.svg (%d market(s))", path, len(views))
    return path


SVG_STYLE = """<style>
.bg { fill:#fcfcfb; stroke:#e4e3df; } .h1 { fill:#0b0b0b; font-size:18px; font-weight:600; }
.h2 { fill:#0b0b0b; font-size:15px; font-weight:600; } .sub { fill:#52514e; font-size:12px; }
.grid { stroke:#e4e3df; stroke-width:1; } .tick { fill:#52514e; font-size:11px; }
.actual { fill:none; stroke:#0b0b0b; stroke-width:2; }
.forecast { fill:none; stroke:#2a78d6; stroke-width:2; } .forecast-key { fill:#2a78d6; }
.actual-key { fill:#0b0b0b; } .next { fill:#2a78d614; } .hit { fill:transparent; }
@media (prefers-color-scheme: dark) {
  .bg { fill:#1a1a19; stroke:#383835; } .h1, .h2 { fill:#f0efec; } .sub, .tick { fill:#c3c2b7; }
  .grid { stroke:#383835; } .actual { stroke:#f0efec; } .actual-key { fill:#f0efec; }
  .forecast { stroke:#3987e5; } .forecast-key { fill:#3987e5; } .next { fill:#3987e51f; }
}
</style>"""


def latest_svg(views: list[dict], updated: str) -> str:
    """One self-contained SVG with every market's chart, for embedding as an image in the README.

    GitHub shows README images through a caching proxy, so the image is at most a few hours old.
    """
    header, panel = 40, 44 + _H
    height = header + len(views) * (panel + 12) + 30
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {height}" width="{_W}" '
        f'height="{height}" font-family="system-ui,-apple-system,Segoe UI,sans-serif" '
        'role="img" aria-label="Latest day-ahead price forecasts">',
        SVG_STYLE,
        f'<rect class="bg" x="0.5" y="0.5" width="{_W - 1}" height="{height - 1}" rx="12"/>',
        '<text class="h1" x="16" y="28">Day-ahead electricity price forecast</text>',
    ]
    y = header
    for view in views:
        t = view["tomorrow"]
        mae, naive, days = _accuracy(view)
        summary = (
            f"Forecast for {view['next_day']:%d %b}: {_fmt(t['price_eur_pred'].mean())} EUR/MWh "
            f"on average · MAE over the last {days} days {_fmt(mae)} (seasonal naive {_fmt(naive)})"
        )
        parts.append(f'<text class="h2" x="16" y="{y + 18}">{html.escape(view["name"])}</text>')
        parts.append(f'<text class="sub" x="16" y="{y + 36}">{html.escape(summary)}</text>')
        chart = chart_svg(view).replace(
            "<svg viewBox=", f'<svg x="0" y="{y + 44}" width="{_W}" height="{_H}" viewBox=', 1
        )
        parts.append(chart)
        y += panel + 12
    parts.append(
        f'<rect class="actual-key" x="16" y="{y + 4}" width="14" height="3"/>'
        f'<text class="sub" x="36" y="{y + 9}">Realised</text>'
        f'<rect class="forecast-key" x="100" y="{y + 4}" width="14" height="3"/>'
        f'<text class="sub" x="120" y="{y + 9}">Forecast (EUR/MWh, local time)</text>'
        f'<text class="sub" x="{_W - 16}" y="{y + 9}" text-anchor="end">'
        f"Updated {updated} UTC</text>"
    )
    if not views:
        parts.append(f'<text class="sub" x="16" y="{header + 20}">No forecasts yet.</text>')
    parts.append("</svg>")
    return "".join(parts)


def _accuracy(view: dict) -> tuple[float | None, float | None, int]:
    st = view["status"]
    model = st.loc[MODEL] if MODEL in st.index else None
    base = st.loc[BASELINE] if BASELINE in st.index else None
    return (
        model["window_mae"] if model is not None else None,
        base["window_mae"] if base is not None else None,
        int(model["window_n_days"]) if model is not None else 0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("site"))
    parser.add_argument("--days", type=int, default=HISTORY_DAYS, help="days of history to show")
    parser.add_argument("--warehouse", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(args.out, args.warehouse, args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
