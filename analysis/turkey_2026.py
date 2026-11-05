"""Out-of-sample check on 2026, and whether more Turkish data helps the forecast.

Three questions, each answered with the same rolling backtest (weekly refit, issue day D, delivery
day D+1) as the 2025 results in the README:

1. Do the 2025 results hold on 2026, which played no part in choosing the model?
2. Does the system operator's load plan, or the realised generation mix, improve the Turkish
   forecast? Both come from the EPİAŞ Transparency Platform.
3. Does a shorter training window help when the market changes?

Look-ahead: the generation mix is taken from the last day complete at issue time (the delivery day
minus two). The load plan is for the delivery hour itself. EPİAŞ describes it only as published
the day before, and it was seen available at 12:50 Turkish time, after the 11:00 issue time, so
its result is an upper bound: the plan may not be out yet when the forecast is made.

Needs the raw data for 2023 to 2026-11-04 and, for the first run, `EPIAS_USERNAME` and
`EPIAS_PASSWORD`. Takes about ten minutes. Writes `docs/analysis/turkey_2026.md` and
`docs/img/turkey_2026.png`.

    pip install -e ".[analysis]"
    python analysis/turkey_2026.py
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from price_forecast import backtest, config, features  # noqa: E402
from price_forecast.ingestion import epias  # noqa: E402
from price_forecast.models.bounded import BoundedForecaster  # noqa: E402
from price_forecast.models.lear import LEARForecaster  # noqa: E402
from price_forecast.models.lightgbm import LightGBMForecaster  # noqa: E402

EXTRA_DIR = config.DATA_DIR / "experiment"
OUT_MD = config.PROJECT_ROOT / "docs" / "analysis" / "turkey_2026.md"
OUT_PNG = config.PROJECT_ROOT / "docs" / "img" / "turkey_2026.png"

FIRST_DAY, LAST_DAY = date(2023, 1, 1), date(2026, 11, 4)
PERIODS = (("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", LAST_DAY.isoformat()))
LOAD_COLUMNS = ["load_plan"]
MIX_COLUMNS = [
    "hydro_t2",
    "gas_t2",
    "wind_t2",
    "river_t2",
    "coal_t2",
    "share_gas_t2",
    "share_hydro_t2",
]
SERIES = {
    "load_plan": "/v1/consumption/data/load-estimation-plan",
    "generation": "/v1/generation/data/realtime-generation",
}

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, BASELINE_GREY = "#2a78d6", "#9a9993"


def download_extra_data() -> None:
    """Fetch each EPİAŞ series month by month, unless it is already on disk."""
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    missing = {n: p for n, p in SERIES.items() if not (EXTRA_DIR / f"{n}.parquet").exists()}
    if not missing:
        return
    with epias.EpiasClient() as client:
        for name, path in missing.items():
            rows = []
            for month in pd.period_range(FIRST_DAY, LAST_DAY, freq="M"):
                body = {
                    "startDate": epias._local_midnight(month.start_time.date()),
                    "endDate": epias._local_midnight(min(month.end_time.date(), LAST_DAY)),
                }
                rows += client.post_json(config.EPIAS_BASE_URL + path, body).get("items") or []
            df = pd.DataFrame(rows)
            hour = df["hour" if "hour" in df else "time"].str[:2].astype(int)
            df["local"] = pd.to_datetime(df["date"].str[:10]) + pd.to_timedelta(hour, unit="h")
            df = df.drop(columns="date").drop_duplicates("local").sort_values("local")
            df.to_parquet(EXTRA_DIR / f"{name}.parquet")


def generation() -> pd.DataFrame:
    return pd.read_parquet(EXTRA_DIR / "generation.parquet")


def turkish_features() -> pd.DataFrame:
    """The usual modelling table plus the load plan and the lagged generation mix."""
    feats = features.build_features(features.load_hourly("tr"), "tr")
    day = pd.to_datetime(feats["delivery_date_local"])
    feats = feats.assign(_day=day, _local=day + pd.to_timedelta(feats["hour_local"], unit="h"))
    load = pd.read_parquet(EXTRA_DIR / "load_plan.parquet")
    load = load[["local", "lep"]].rename(columns={"local": "_local", "lep": "load_plan"})
    feats = feats.merge(load, on="_local", how="left")

    gen = generation()
    daily = gen.groupby(gen["local"].dt.normalize()).mean(numeric_only=True)
    mix = pd.DataFrame(
        {
            "hydro_t2": daily["dammedHydro"],
            "gas_t2": daily["naturalGas"],
            "wind_t2": daily["wind"],
            "river_t2": daily["river"],
            "coal_t2": daily["importCoal"] + daily["lignite"],
            "share_gas_t2": daily["naturalGas"] / daily["total"],
            "share_hydro_t2": daily["dammedHydro"] / daily["total"],
        }
    )
    mix.index = mix.index + timedelta(days=2)
    feats = feats.merge(mix, left_on="_day", right_index=True, how="left")
    return feats.drop(columns=["_day", "_local"])


class LearWithExtras(LEARForecaster):
    """LEAR with extra inputs: `hourly` also get a day-mean column, `daily` are used as they are."""

    def __init__(self, tag: str, hourly=(), daily=(), calibration_days: int | None = None):
        super().__init__(calibration_days=calibration_days)
        self.name = f"lear_{tag}"
        self.hourly, self.daily = list(hourly), list(daily)

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = [super()._design(df)]
        if self.hourly:
            values = df[self.hourly].astype(float)
            day_mean = values.groupby(df["delivery_date_local"]).transform("mean")
            parts += [values, day_mean.add_suffix("_day")]
        if self.daily:
            parts.append(df[self.daily].astype(float))
        return pd.concat(parts, axis=1)


class LightGBMWithExtras(LightGBMForecaster):
    def __init__(self, tag: str, extra=()):
        super().__init__()
        self.name = f"lightgbm_{tag}"
        self.extra = list(extra)

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        base = LightGBMForecaster._design(df)
        return pd.concat([base, df[self.extra].astype(float)], axis=1) if self.extra else base


DATA_LABELS = {
    "lear_base_bounded": "LEAR",
    "lear_load_bounded": "LEAR + load plan",
    "lear_mix_bounded": "LEAR + generation mix",
    "lear_both_bounded": "LEAR + both",
    "lightgbm_base_bounded": "LightGBM",
    "lightgbm_both_bounded": "LightGBM + both",
}
WINDOW_LABELS = {
    "lear_all_bounded": "LEAR, all history",
    "lear_365_bounded": "LEAR, last 365 days",
    "lear_180_bounded": "LEAR, last 180 days",
    "lightgbm_all_bounded": "LightGBM, all history",
}


def data_models() -> list[BoundedForecaster]:
    return [
        BoundedForecaster(LearWithExtras("base")),
        BoundedForecaster(LearWithExtras("load", hourly=LOAD_COLUMNS)),
        BoundedForecaster(LearWithExtras("mix", daily=MIX_COLUMNS)),
        BoundedForecaster(LearWithExtras("both", hourly=LOAD_COLUMNS, daily=MIX_COLUMNS)),
        BoundedForecaster(LightGBMWithExtras("base")),
        BoundedForecaster(LightGBMWithExtras("both", extra=LOAD_COLUMNS + MIX_COLUMNS)),
    ]


def window_models() -> list[BoundedForecaster]:
    return [
        BoundedForecaster(LearWithExtras("all")),
        BoundedForecaster(LearWithExtras("365", calibration_days=365)),
        BoundedForecaster(LearWithExtras("180", calibration_days=180)),
        BoundedForecaster(LightGBMWithExtras("all")),
    ]


def run(market: str, models: list, feats: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    preds = backtest.run(market, models, start, end, features=feats, refit_every=7)
    preds["err"] = (preds["price_eur_pred"] - preds["price_eur"]).abs()
    return preds


def table(rows: list[list[str]], header: list[str]) -> str:
    align = "|---|" + "---:|" * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", align]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def paired_t(preds: pd.DataFrame, base: str, other: str) -> float:
    """t statistic of the daily MAE difference; above about 2, `other` is reliably better."""
    wide = preds.pivot_table(index="delivery_hour", columns="model", values="err")
    days = preds.drop_duplicates("delivery_hour").set_index("delivery_hour")["delivery_date_local"]
    diff = (wide[base] - wide[other]).groupby(days).mean()
    return float(diff.mean() / (diff.std() / np.sqrt(len(diff))))


def data_section(results: dict[str, pd.DataFrame]) -> str:
    rows = []
    for model, label in DATA_LABELS.items():
        row = [label]
        for period in results:
            preds = results[period]
            mae = preds[preds["model"] == model]["err"].mean()
            base = "lightgbm_base_bounded" if model.startswith("lightgbm") else "lear_base_bounded"
            t = "" if model == base else f" (t = {paired_t(preds, base, model):.1f})"
            row.append(f"{mae:.2f}{t}")
        rows.append(row)
    return table(rows, ["Model", *[f"MAE {p}" for p in results]])


def window_section(results: dict[str, dict[str, pd.DataFrame]]) -> str:
    rows = []
    for model, label in WINDOW_LABELS.items():
        row = [label]
        for by_period in results.values():
            for preds in by_period.values():
                row.append(f"{preds[preds['model'] == model]['err'].mean():.2f}")
        rows.append(row)
    header = ["Model"] + [f"{m} {p}" for m, by in results.items() for p in by]
    return table(rows, header)


def year_section(results: dict[str, dict[str, pd.DataFrame]]) -> str:
    rows = []
    for market, by_period in results.items():
        for period, preds in by_period.items():
            lear = preds[preds["model"] == "lear_all_bounded"]
            mae, mean = lear["err"].mean(), lear["price_eur"].mean()
            snaive = (lear["price_eur_snaive"] - lear["price_eur"]).abs().mean()
            rows.append(
                [
                    f"{config.MARKETS[market].name} {period}",
                    f"{mean:.1f}",
                    f"{mae:.1f}",
                    f"{mae / abs(mean) * 100:.1f}%",
                    f"{mae / snaive:.2f}",
                ]
            )
    return table(rows, ["Period", "Mean price", "MAE", "MAE / mean price", "rMAE"])


def monthly_context(lear: pd.DataFrame) -> pd.DataFrame:
    """Per year and month: mean price, zero-price share, hydro share and LEAR's error."""
    pattern = (config.PRICES_RAW_DIR / "market=tr" / "*" / "*" / "data.parquet").as_posix()
    prices = duckdb.sql(
        f"SELECT delivery_start + INTERVAL 3 HOUR AS local, price, price_eur "
        f"FROM read_parquet('{pattern}')"
    ).df()
    prices = prices[prices["local"] < pd.Timestamp(LAST_DAY) + pd.Timedelta(days=1)]
    gen = generation()
    gen = gen[gen["local"] < pd.Timestamp(LAST_DAY) + pd.Timedelta(days=1)]
    hydro = gen["dammedHydro"] + gen["river"]

    def by_month(values: pd.Series, when: pd.Series) -> pd.Series:
        return values.groupby([when.dt.year.rename("year"), when.dt.month.rename("month")]).mean()

    days = pd.to_datetime(lear["delivery_date_local"])
    out = pd.DataFrame(
        {
            "price": by_month(prices["price_eur"], prices["local"]),
            "zero_share": by_month((prices["price"] == 0).astype(float), prices["local"]) * 100,
            "hydro_share": by_month(hydro / gen["total"], gen["local"]) * 100,
            "lear_mae": by_month(lear["err"], days),
        }
    )
    return out.loc[[2025, 2026]]


def chart(context: pd.DataFrame) -> None:
    rows = (
        ("price", "Mean price, EUR/MWh"),
        ("hydro_share", "Hydro share of generation, %"),
        ("lear_mae", "LEAR forecast error (MAE), EUR/MWh"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, facecolor=SURFACE)
    for ax, (column, title) in zip(axes, rows, strict=True):
        for year, color, width in ((2025, BASELINE_GREY, 2), (2026, BLUE, 2.5)):
            series = context.loc[year][column]
            ax.plot(series.index, series, color=color, linewidth=width, marker="o", markersize=4)
            ax.annotate(
                str(year),
                (series.index[-1], series.iloc[-1]),
                xytext=(7, 0),
                textcoords="offset points",
                color=INK_2,
                fontsize=9,
                va="center",
            )
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_ylim(bottom=0)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    fig.suptitle(
        "Turkey: spring 2026 had far more hydro, far lower prices and the largest errors",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    download_extra_data()
    tr_feats = turkish_features()

    data_results = {
        label: run("tr", data_models(), tr_feats, start, end) for label, start, end in PERIODS
    }
    window_results: dict[str, dict[str, pd.DataFrame]] = {}
    for market in ("de_lu", "tr"):
        feats = features.build_features(features.load_hourly(market), market)
        window_results[market] = {
            label: run(market, window_models(), feats, start, end) for label, start, end in PERIODS
        }

    tr = window_results["tr"]
    lear = pd.concat([p[p["model"] == "lear_all_bounded"] for p in tr.values()], ignore_index=True)
    context = monthly_context(lear)
    chart(context)

    spring = context.loc[(context.index.get_level_values("month")).isin([4, 5])].reset_index()
    spring_rows = [
        [
            f"{int(r.year)}-{int(r.month):02d}",
            f"{r.price:.0f}",
            f"{r.zero_share:.0f}%",
            f"{r.hydro_share:.0f}%",
            f"{r.lear_mae:.1f}",
        ]
        for r in spring.itertuples()
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(
        f"""# Out-of-sample check on 2026, and more data for Turkey

Generated by `analysis/turkey_2026.py`. MAE in EUR/MWh. All forecasts are kept within the market's
price limits. 2026 runs from 1 January to {LAST_DAY:%d %B}.

## Do the 2025 results hold?

{year_section(window_results)}

rMAE is the MAE relative to the seasonal naive on the same hours; lower is better.

## Does more data help in Turkey?

{data_section(data_results)}

The t statistic compares daily errors with the same model without the extra data; above about 2,
the extra data is reliably better, below about -2, reliably worse. The load plan is an upper
bound (see the script's docstring). Neither source helps consistently across both years, so
neither was added to the model.

## Does a shorter training window help?

{window_section(window_results)}

## Spring 2026 in Turkey

{table(spring_rows, ["Month", "Mean price", "Zero-price hours", "Hydro share", "LEAR MAE"])}

These move together; the data does not prove the cause.
""",
        encoding="utf-8",
    )
    print(f"wrote {OUT_MD} and {OUT_PNG}")


if __name__ == "__main__":
    main()
