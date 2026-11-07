"""What is the forecast worth? A battery scheduled from each forecast, paid at realised prices.

Fixed before the results were seen:
- Battery: 1 MW / 2 MWh, 85% round trip, at most one full cycle a day, empty at the start and end of
  each delivery day. Price taker, no fees and no wear. An illustration, not a business case.
- Ceiling: the same battery with the realised prices known in advance.
- Rule: the median forecast "adds value" only if it earns more than each simple benchmark (issue
  day's prices, same day last week, 28-day hourly profile) in all four market-years, reliably
  (paired t above 2 on daily profit).

2025 and 2026 were both used to choose the model, so this is not an independent test of accuracy.

Writes `docs/analysis/battery.md`. Needs the raw data to 4 November 2026. The first run takes about
ten minutes on one core and keeps the forecasts in `data/battery_<market>_<year>.parquet`.

    python analysis/battery.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ensemble import BASE, PERIODS, TRIO, wide_predictions

from price_forecast import backtest, config, features
from price_forecast.battery import Battery, profit, schedule
from price_forecast.models import make_model

OUT_MD = config.PROJECT_ROOT / "docs" / "analysis" / "battery.md"
MODEL = "median"
BENCHMARKS = {
    "naive": "Issue day's prices",
    "seasonal_naive": "Same day last week",
    "profile": "Hourly profile, last 28 days",
}
STRATEGIES = {MODEL: "Median of three", BASE: "LEAR (bounded)", **BENCHMARKS}
PROFILE_DAYS = 28
BASE_BATTERY = Battery()
SIZES = (1.0, 2.0, 4.0)
RECENT_START = "2026-10-28"  # German prices jumped; the forecast has been too low since


def _profile(market: str, hourly: pd.DataFrame) -> pd.DataFrame:
    """Mean price per local hour over the 28 days before each day: all known at issue time, since
    the issue day's prices were published the day before it."""
    tz = config.MARKETS[market].timezone
    local = pd.to_datetime(hourly["delivery_hour"], utc=True).dt.tz_convert(tz)
    table = pd.DataFrame(
        {"day": local.dt.tz_localize(None).dt.normalize(), "hour": local.dt.hour}
    ).assign(price=hourly["price_eur"].to_numpy())
    wide = table.pivot_table(index="day", columns="hour", values="price")
    return wide.rolling(PROFILE_DAYS, min_periods=14).mean().shift(1)


def predictions(market: str, feats: pd.DataFrame, label: str, start: str, end: str) -> pd.DataFrame:
    path = config.DATA_DIR / f"battery_{market}_{label}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    w = wide_predictions(market, feats, start, end)
    simple = backtest.run(
        market, [make_model("naive"), make_model("seasonal_naive")], start, end, features=feats
    ).pivot_table(index="delivery_hour", columns="model", values="price_eur_pred")
    w = w[w["day"] >= start].join(simple)
    w[MODEL] = w[TRIO].median(axis=1)

    tz = config.MARKETS[market].timezone
    hours = pd.DatetimeIndex(pd.to_datetime(w.index, utc=True)).tz_convert(tz).hour
    profile = _profile(market, features.load_hourly(market))
    stacked = profile.stack()
    w["profile"] = stacked.reindex(pd.MultiIndex.from_arrays([w["day"], hours])).to_numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    w.to_parquet(path)
    return w


def daily_profits(w: pd.DataFrame, battery: Battery, columns: list[str]) -> pd.DataFrame:
    """One row per delivery day, one column per strategy, plus `ceiling`. Days where any strategy
    has no forecast are dropped, so every column covers the same days."""
    rows = {}
    for day, group in w.groupby("day"):
        day_data = group[["price", *columns]].astype(float)
        if day_data.isna().any().any():
            continue
        realised = day_data["price"].to_numpy()
        row = {c: profit(schedule(day_data[c].to_numpy(), battery), realised) for c in columns}
        row["ceiling"] = profit(schedule(realised, battery), realised)
        rows[day] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def paired_t(a: pd.Series, b: pd.Series) -> float:
    diff = a - b
    return diff.mean() / (diff.std() / np.sqrt(len(diff))) if diff.std() > 0 else 0.0


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    columns = list(STRATEGIES)
    results: dict[str, pd.DataFrame] = {}
    wides: dict[str, pd.DataFrame] = {}
    for market in ("de_lu", "tr"):
        feats = features.build_features(features.load_hourly(market), market)
        for label, start, end in PERIODS:
            key = f"{config.MARKETS[market].name} {label}"
            wides[key] = predictions(market, feats, label, start, end)
            results[key] = daily_profits(wides[key], BASE_BATTERY, columns)
            print(key, "days", len(results[key]), flush=True)

    head = "| Strategy | MAE | EUR per MW per day | Share of the ceiling | t against median |"
    lines = []
    failed = []
    for key, daily in results.items():
        w = wides[key]
        lines += [f"\n### {key} ({len(daily)} days)\n", head, "|---|---:|---:|---:|---:|"]
        ceiling = daily["ceiling"].mean()
        lines.append(f"| Prices known in advance (ceiling) | | {ceiling:.0f} | 100% | |")
        for c, name in STRATEGIES.items():
            mae = (w[c] - w["price"]).abs().mean()
            t = "" if c == MODEL else f"{paired_t(daily[MODEL], daily[c]):.2f}"
            lines.append(
                f"| {name} | {mae:.1f} | {daily[c].mean():.0f} | "
                f"{daily[c].mean() / ceiling:.0%} | {t} |"
            )
            if c in BENCHMARKS and paired_t(daily[MODEL], daily[c]) <= 2:
                failed.append(f"{key} against {name.lower()} (t = {t})")

    sizes = ["\n## Battery size\n", "Share of the ceiling, median forecast, 1 MW.\n"]
    sizes += ["| | " + " | ".join(f"{s:.0f} MWh" for s in SIZES) + " |"]
    sizes += ["|---|" + "---:|" * len(SIZES)]
    for key, w in wides.items():
        cells = []
        for size in SIZES:
            battery = Battery(capacity_mwh=size)
            d = (
                results[key]
                if battery == BASE_BATTERY
                else daily_profits(w, battery, [MODEL, *BENCHMARKS])
            )
            cells.append(f"{d[MODEL].mean() / d['ceiling'].mean():.0%}")
        sizes.append(f"| {key} | " + " | ".join(cells) + " |")

    recent_key = f"{config.MARKETS['de_lu'].name} 2026"
    w = wides[recent_key]
    recent = results[recent_key].loc[RECENT_START:]
    window = w[w["day"] >= RECENT_START]
    bias = (window[MODEL] - window["price"]).mean()
    others = "; ".join(f"{STRATEGIES[c]}: {recent[c].mean():.0f}" for c in columns[1:])
    recent_text = (
        f"\n## Germany since {RECENT_START} ({len(recent)} days)\n\n"
        f"The median forecast's mean error over these days is {bias:+.0f} EUR/MWh. It earns "
        f"{recent[MODEL].mean():.0f} of the {recent['ceiling'].mean():.0f} EUR per MW per day "
        f"available; {others}. These are few days: read it as an example.\n"
    )

    verdict = "passes" if not failed else "fails"
    verdict_text = (
        "**The median passes.**"
        if not failed
        else "**The median fails**, on: " + "; ".join(failed) + "."
    )
    OUT_MD.write_text(
        f"""# What the forecast is worth: a battery

Generated by `analysis/battery.py`. A 1 MW / 2 MWh battery (85% round trip, one cycle a day, empty
at the start and end of each day) is scheduled from each forecast and paid at realised hourly
prices. Price taker, no fees and no wear, so these are not expected earnings. Germany has traded in
15-minute products since 1 October 2025; hourly averages hide part of that spread.

The rule, fixed before the results: the median earns more than each simple benchmark in all four
market-years, with paired t above 2 on daily profit. {verdict_text}

## Results
{chr(10).join(lines)}
{chr(10).join(sizes)}
{recent_text}""",
        encoding="utf-8",
    )
    print(f"wrote {OUT_MD}; rule {verdict}")


if __name__ == "__main__":
    main()
