"""A static HTML page with the latest forecasts and their accuracy, for GitHub Pages.

Reads the warehouse (forecasts, realised prices, model status) and writes `<out>/index.html`: per
market, tomorrow's forecast, the last `history_days` days of forecast against the realised price,
and the rolling accuracy against the seasonal naive. Inline SVG, no scripts, no external assets,
light and dark themes.

    python -m price_forecast.report --out site
"""

from __future__ import annotations

import argparse
import html
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
    return {
        "market": market,
        "name": config.MARKETS[market].name,
        "timezone": tz,
        "next_day": next_day,
        "tomorrow": tomorrow,
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


def chart_svg(view: dict) -> str:
    f, r = view["forecasts"], view["realised"]
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

    parts = [
        f'<svg viewBox="0 0 {_W} {_H}" role="img" aria-label="Forecast and realised price, '
        f'{html.escape(view["name"])}">'
    ]
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
    day = t0.normalize()
    while day <= t1:
        if day >= t0:
            parts.append(
                f'<text class="tick" x="{x(day):.1f}" y="{_H - 8}" text-anchor="middle">'
                f"{day:%d %b}</text>"
            )
        day += pd.Timedelta(days=2)
    parts.append(f'<path class="actual" d="{path(r["local"], r["price_eur"])}"/>')
    parts.append(f'<path class="forecast" d="{path(f["local"], f["price_eur_pred"])}"/>')
    for t, v in zip(f["local"], f["price_eur_pred"], strict=True):
        parts.append(
            f'<circle class="hit" cx="{x(t):.1f}" cy="{y(v):.1f}" r="4">'
            f"<title>{t:%d %b %H:%M}: forecast {v:.1f} EUR/MWh</title></circle>"
        )
    parts.append("</svg>")
    return "".join(parts)


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
        ("Tomorrow, average", f"{_fmt(t['price_eur_pred'].mean())} EUR/MWh"),
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
    day = f"delivery day {view['next_day']:%A %d %B %Y}"
    return f"""
<section>
  <h2>{html.escape(view["name"])} <span class="day">{day}</span></h2>
  <div class="tiles">{tiles}</div>
  {chart_svg(view)}
  <p class="legend"><span class="swatch actual"></span>Realised price
    <span class="swatch forecast"></span>Forecast (LEAR within the market's price limits)
    <span class="swatch next"></span>Tomorrow · model status: {html.escape(status_text)}</p>
</section>"""


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
.day {{ color:var(--ink2); font-weight:400; font-size:14px; margin-left:6px; }}
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
.hit {{ fill:transparent; }}
.hit:hover {{ fill:var(--forecast); }}
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
</main></body>
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
    views = [
        v
        for m in config.MARKETS
        if (v := market_view(m, forecasts, realised, status, history_days)) is not None
    ]
    sections = "".join(section_html(v) for v in views) or "<p>No forecasts yet.</p>"
    updated = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(
        PAGE.format(sections=sections, updated=updated, repo=REPO_URL), encoding="utf-8"
    )
    log.info("wrote %s (%d market(s))", path, len(views))
    return path


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
