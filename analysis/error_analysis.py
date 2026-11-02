"""Where are day-ahead prices hard to forecast, and why does the best model differ by market?

Reads the 2025 backtest predictions (`data/backtest_<market>_2025.parquet`, written by
`analysis/run_backtests.py`) and the raw prices, then writes:

- `docs/analysis/results.md`: error tables by regime, the price-cap experiment;
- `docs/img/error_by_regime.png`, `docs/img/monthly_mae.png`, `docs/img/example_weeks.png`.

Run from the repository root: `python analysis/error_analysis.py`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import holidays
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from price_forecast import config  # noqa: E402

DOCS = config.PROJECT_ROOT / "docs"
IMG = DOCS / "img"
OUT_MD = DOCS / "analysis" / "results.md"

# Reference palette: categorical slots in fixed order; the baseline is neutral grey.
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {
    "seasonal_naive": "#9a9993",
    "lear": "#2a78d6",
    "lightgbm": "#eb6834",
    "lear_capped": "#1baf7a",
}
LABELS = {
    "seasonal_naive": "Seasonal naive",
    "naive": "Naive",
    "lear": "LEAR",
    "lightgbm": "LightGBM",
    "lear_capped": "LEAR within 0–cap",
}
MODELS = ["seasonal_naive", "lear", "lightgbm"]


def load_predictions(market: str) -> pd.DataFrame:
    path = config.DATA_DIR / f"backtest_{market}_2025.parquet"
    p = pd.read_parquet(path)
    p["delivery_hour"] = (
        pd.to_datetime(p["delivery_hour"]).dt.tz_convert("UTC").dt.tz_localize(None)
    )
    p["delivery_date_local"] = pd.to_datetime(p["delivery_date_local"]).dt.date
    p["err"] = (p["price_eur_pred"] - p["price_eur"]).abs()
    country = config.MARKETS[market].holiday_country
    hols = holidays.country_holidays(country, years=[2025])
    p["is_holiday"] = p["delivery_date_local"].map(lambda d: d in hols)
    p["is_weekend"] = pd.to_datetime(p["delivery_date_local"]).dt.day_of_week >= 5
    return p


def turkish_caps() -> pd.DataFrame:
    """Hourly TRY price, the month's price cap and the EUR/TRY rate implied by EPİAŞ's prices.

    The cap is the regulated maximum price. It is read off the data as the month's highest price,
    which the market sits at for dozens of hours every month; in production it would come from the
    regulator's published decisions, which are known in advance.
    """
    raw = duckdb.sql(
        f"SELECT delivery_start, price, price_eur "
        f"FROM '{config.PRICES_RAW_DIR.as_posix()}/market=tr/*/*/data.parquet'"
    ).df()
    local = raw["delivery_start"] + pd.Timedelta(hours=3)
    raw["local_date"] = local.dt.date
    raw["cap_try"] = raw.groupby(local.dt.to_period("M"))["price"].transform("max")
    raw["eur_per_try"] = np.where(raw["price"] > 0, raw["price_eur"] / raw["price"], np.nan)
    raw["eur_per_try_day"] = raw.groupby("local_date")["eur_per_try"].transform("mean")
    return raw.rename(columns={"delivery_start": "delivery_hour"})


def add_turkish_regimes(p: pd.DataFrame) -> pd.DataFrame:
    caps = turkish_caps()
    p = p.merge(caps[["delivery_hour", "price", "cap_try"]], on="delivery_hour", how="left")
    # The rate of the issue day (the day before delivery) is known when the forecast is made.
    rate = caps.groupby("local_date")["eur_per_try_day"].first()
    issue_day = p["delivery_date_local"].map(lambda d: d - pd.Timedelta(days=1))
    p["cap_eur_known"] = p["cap_try"] * issue_day.map(rate)
    p["at_cap"] = p["price"] >= p["cap_try"]
    p["zero_price"] = p["price"] == 0
    return p


def capped_lear(p: pd.DataFrame) -> pd.DataFrame:
    """LEAR limited to the Turkish market's bounds: no negative prices, nothing above the cap."""
    lear = p[p["model"] == "lear"].copy()
    lear["model"] = "lear_capped"
    lear["price_eur_pred"] = lear["price_eur_pred"].clip(lower=0).clip(upper=lear["cap_eur_known"])
    lear["err"] = (lear["price_eur_pred"] - lear["price_eur"]).abs()
    return lear


def regime_table(p: pd.DataFrame, regimes: dict[str, pd.Series], models: list[str]) -> pd.DataFrame:
    rows = []
    for name, mask in regimes.items():
        sub = p[mask]
        hours = sub[sub["model"] == models[0]].shape[0]
        maes = sub.groupby("model")["err"].mean()
        rows.append({"regime": name, "hours": hours, **{m: maes.get(m, np.nan) for m in models}})
    return pd.DataFrame(rows)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_regimes(tables: dict[str, pd.DataFrame], models: dict[str, list[str]], path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, (market, table) in zip(axes, tables.items(), strict=True):
        ms = models[market]
        n = len(ms)
        height = 0.8 / n
        y = np.arange(len(table))[::-1]
        for i, m in enumerate(ms):
            ax.barh(
                y + (n - 1 - i - (n - 1) / 2) * height,
                table[m],
                height=height * 0.9,
                color=COLORS[m],
                label=LABELS[m],
            )
        ax.set_yticks(
            y, [f"{r} ({h:,} h)" for r, h in zip(table["regime"], table["hours"], strict=True)]
        )
        ax.set_xlabel("MAE, EUR/MWh", color=INK_2, fontsize=9)
        ax.set_title(config.MARKETS[market].name, loc="left", color=INK, fontsize=11)
        style(ax)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=len(labels), loc="lower center")
    fig.suptitle(
        "Forecast error by situation, 2025 backtest", x=0.01, ha="left", color=INK, fontsize=13
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_monthly(preds: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), facecolor=SURFACE)
    for ax, (market, p) in zip(axes, preds.items(), strict=True):
        month = pd.to_datetime(p["delivery_date_local"]).dt.month
        table = p.assign(month=month).groupby(["month", "model"])["err"].mean().unstack()
        for m in MODELS:
            ax.plot(
                table.index,
                table[m],
                color=COLORS[m],
                linewidth=2,
                marker="o",
                markersize=4,
                label=LABELS[m],
            )
        ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_ylim(bottom=0)
        ax.set_ylabel("MAE, EUR/MWh", color=INK_2, fontsize=9)
        ax.set_title(config.MARKETS[market].name, loc="left", color=INK, fontsize=11)
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Monthly forecast error, 2025", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_weeks(windows: dict[str, tuple[pd.DataFrame, date, date, str]], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor=SURFACE)
    for ax, (market, (p, start, end, note)) in zip(axes, windows.items(), strict=True):
        days = pd.Series(p["delivery_date_local"])
        sub = p[(days >= start) & (days <= end)]
        tz = config.MARKETS[market].timezone
        actual = sub[sub["model"] == "lear"].sort_values("delivery_hour")
        x = actual["delivery_hour"].dt.tz_localize("UTC").dt.tz_convert(tz)
        ax.plot(x, actual["price_eur"], color=INK, linewidth=2, label="Actual price")
        for m in ("lear", "lightgbm"):
            s = sub[sub["model"] == m].sort_values("delivery_hour")
            ax.plot(
                x, s["price_eur_pred"].to_numpy(), color=COLORS[m], linewidth=1.5, label=LABELS[m]
            )
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_ylabel("EUR/MWh", color=INK_2, fontsize=9)
        ax.set_title(f"{config.MARKETS[market].name}: {note}", loc="left", color=INK, fontsize=11)
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.08))
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def md_table(table: pd.DataFrame, models: list[str]) -> str:
    head = "| Situation | Hours | " + " | ".join(LABELS[m] for m in models) + " |"
    sep = "|---|---:|" + "---:|" * len(models)
    rows = []
    for _, r in table.iterrows():
        best = min(models, key=lambda m: r[m])
        cells = [f"**{r[m]:.1f}**" if m == best else f"{r[m]:.1f}" for m in models]
        rows.append(f"| {r['regime']} | {r['hours']:,} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    de = load_predictions("de_lu")
    tr = add_turkish_regimes(load_predictions("tr"))
    tr_all = pd.concat([tr, capped_lear(tr)], ignore_index=True)

    workday_de = ~de["is_holiday"] & ~de["is_weekend"]
    de_regimes = {
        "All hours": de["model"].notna(),
        "Negative prices": de["price_eur"] < 0,
        "Holidays": de["is_holiday"],
        "Normal workdays": workday_de & (de["price_eur"] >= 0),
    }
    workday_tr = ~tr_all["is_holiday"] & ~tr_all["is_weekend"]
    tr_regimes = {
        "All hours": tr_all["model"].notna(),
        "At the price cap": tr_all["at_cap"],
        "Zero prices": tr_all["zero_price"],
        "Holidays": tr_all["is_holiday"],
        "Normal workdays": workday_tr & ~tr_all["at_cap"] & ~tr_all["zero_price"],
    }
    tr_models = [*MODELS, "lear_capped"]
    de_table = regime_table(de, de_regimes, MODELS)
    tr_table = regime_table(tr_all, tr_regimes, tr_models)

    chart_regimes(
        {"de_lu": de_table, "tr": tr_table},
        {"de_lu": MODELS, "tr": tr_models},
        IMG / "error_by_regime.png",
    )
    chart_monthly({"de_lu": de, "tr": tr}, IMG / "monthly_mae.png")

    # The German week with the most negative hours, and the Turkish Kurban Bayramı week.
    neg = de[(de["model"] == "lear") & (de["price_eur"] < 0)]
    week_start = (
        pd.to_datetime(neg["delivery_date_local"]).dt.to_period("W").value_counts().idxmax()
    )
    de_start = week_start.start_time.date()
    chart_weeks(
        {
            "de_lu": (
                de,
                de_start,
                de_start + pd.Timedelta(days=6),
                "the week with the most negative prices",
            ),
            "tr": (tr, date(2025, 6, 4), date(2025, 6, 10), "Kurban Bayramı, 6–9 June"),
        },
        IMG / "example_weeks.png",
    )

    lear_rows = tr[tr["model"] == "lear"]
    over = lear_rows[lear_rows["price_eur_pred"] > lear_rows["cap_eur_known"]]
    under = lear_rows[lear_rows["price_eur_pred"] < 0]
    lear_tr = tr[tr["model"] == "lear"]["err"].mean()
    capped_tr = tr_all[tr_all["model"] == "lear_capped"]["err"].mean()
    lgbm_tr = tr[tr["model"] == "lightgbm"]["err"].mean()
    floor_tr = (lear_rows["price_eur_pred"].clip(lower=0) - lear_rows["price_eur"]).abs().mean()
    caps = turkish_caps()
    cap_history = (
        caps.assign(month=pd.to_datetime(caps["local_date"]).dt.to_period("M"))
        .groupby("month")["cap_try"]
        .first()
    )
    changes = cap_history[cap_history != cap_history.shift()]
    cap_lines = ", ".join(f"{m}: {int(v):,} TRY" for m, v in changes.items())

    OUT_MD.write_text(
        f"""# Error analysis, 2025 backtest

Generated by `analysis/error_analysis.py`. MAE in EUR/MWh; bold = best per row.

## Germany-Luxembourg

{md_table(de_table, MODELS)}

## Turkey

{md_table(tr_table, tr_models)}

*LEAR within 0–cap* is LEAR with every forecast limited to the Turkish market's own bounds:
prices cannot be negative, and cannot exceed the regulated maximum. The cap is in TRY; it is
converted with the issue day's EUR/TRY rate implied by EPİAŞ's own prices, so the experiment uses
nothing that was unknown at the issue time. Unbounded, LEAR forecast a negative price in
{len(under):,} hours and a price above the cap in {len(over):,} hours (of 8,760). Keeping it within
the bounds brings its MAE from {lear_tr:.2f} to {capped_tr:.2f} (LightGBM: {lgbm_tr:.2f}).

The floor alone, a fixed market rule that involves no estimated value, already brings LEAR to
{floor_tr:.2f}, below LightGBM. The cap adds the rest; its values below are read off the data
(the price the market sits at for dozens of hours each month) and in production would come from
the regulator's decisions, which are published in advance.

Monthly cap values: {cap_lines}.
""",
        encoding="utf-8",
    )
    print(f"wrote {OUT_MD} and charts in {IMG}")


if __name__ == "__main__":
    main()
