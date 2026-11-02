"""LEAR-style benchmark: one LASSO-estimated autoregressive model per local hour of the day.

In the spirit of LEAR from Lago, Marcjasz, De Schutter and Weron (2021), "Forecasting day-ahead
electricity prices: A review of state-of-the-art algorithms, best practices and an open-access
benchmark". As there:

- 24 separate linear models, one per delivery hour, each seeing the whole daily price profile
  (so hour 18 can use yesterday's hour 12, etc.).
- Prices are median/MAD normalised and passed through asinh (variance-stabilising transform),
  inputs and target alike; forecasts are transformed back.
- The L1 penalty is chosen per hour by an information criterion (LassoLarsIC, AIC), which is fast
  enough to refit daily.

Differences, forced by the modelling table: price inputs are the issue day's and the week-before
day's full 24-hour profiles (LEAR also uses D-2 and D-3), recovered by pivoting
`price_d1_same_hour` / `price_d7_same_hour` over the delivery day's rows; exogenous inputs are the
delivery hour's weather and the delivery day's mean weather instead of load and renewable
forecasts. Only one calibration window is used (LEAR averages four).

Because the profiles come from the frame's own rows, `predict` expects whole delivery days (as the
backtest and the daily forecast pass them); hours missing from a day are imputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from price_forecast.features import TARGET_COLUMN, WEATHER_COLUMNS
from price_forecast.models._common import AsinhScaler, with_target

MIN_ROWS_PER_HOUR = 7


class LEARForecaster:
    name = "lear"

    def __init__(self, calibration_days: int | None = None) -> None:
        """`calibration_days`: fit on only the most recent days of the training frame."""
        self.calibration_days = calibration_days

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        day = df["delivery_date_local"]
        parts = []
        for col, prefix in (("price_d1_same_hour", "d1"), ("price_d7_same_hour", "d7")):
            profile = (
                df.groupby(["delivery_date_local", "hour_local"])[col]
                .mean()
                .unstack("hour_local")
                .reindex(columns=range(24))
            )
            values = self.scaler_.transform(profile.reindex(day).to_numpy())
            parts.append(
                pd.DataFrame(
                    values, index=df.index, columns=[f"{prefix}_h{h:02d}" for h in range(24)]
                )
            )
        weather = df[list(WEATHER_COLUMNS)].astype(float)
        daily_weather = weather.groupby(day).transform("mean").add_suffix("_day_mean")
        dow = pd.DataFrame(
            {f"dow_{d}": (df["day_of_week"] == d).astype(float) for d in range(1, 8)},
            index=df.index,
        )
        holiday = df["is_holiday"].astype(float).rename("is_holiday")
        return pd.concat([*parts, weather, daily_weather, dow, holiday], axis=1)

    def _new_hour_model(self, n_rows: int, n_features: int, y: np.ndarray) -> Pipeline:
        # LassoLarsIC estimates the noise variance from an OLS fit, which needs more rows than
        # features. With a short history fall back to the (larger) target variance, which only
        # makes the selected model sparser.
        noise_variance = None if n_rows > n_features + 1 else max(float(np.var(y)), 1e-6)
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LassoLarsIC(criterion="aic", noise_variance=noise_variance),
        )

    def fit(self, train: pd.DataFrame) -> LEARForecaster:
        # 24 small least-squares problems: multithreaded BLAS only adds overhead (about 7x here).
        with threadpool_limits(limits=1, user_api="blas"):
            return self._fit(train)

    def _fit(self, train: pd.DataFrame) -> LEARForecaster:
        train = with_target(train)
        if self.calibration_days:
            days = pd.to_datetime(train["delivery_date_local"])
            train = train[days > days.max() - pd.Timedelta(days=self.calibration_days)]
        self.scaler_ = AsinhScaler().fit(train[TARGET_COLUMN])
        X = self._design(train)
        y = self.scaler_.transform(train[TARGET_COLUMN])
        self.columns_ = list(X.columns)
        # Each hour's fitted pipeline is collapsed to arrays: impute NaN with `fill_`, then
        # z = intercept_ + ((x - mean_) / scale_) . coef_. Predicting a day is one array op.
        n_inputs = X.shape[1]
        self.fill_, self.mean_ = np.zeros((24, n_inputs)), np.zeros((24, n_inputs))
        self.scale_, self.coef_ = np.ones((24, n_inputs)), np.zeros((24, n_inputs))
        self.intercept_ = np.zeros(24)
        for hour in range(24):
            mask = (train["hour_local"] == hour).to_numpy()
            if mask.sum() < MIN_ROWS_PER_HOUR:
                raise ValueError(
                    f"LEAR needs at least {MIN_ROWS_PER_HOUR} training rows per hour, "
                    f"got {mask.sum()} for hour {hour}"
                )
            pipe = self._new_hour_model(int(mask.sum()), n_inputs, y[mask]).fit(X[mask], y[mask])
            imputer, std, lasso = pipe[0], pipe[1], pipe[2]
            self.fill_[hour], self.mean_[hour] = imputer.statistics_, std.mean_
            self.scale_[hour], self.coef_[hour] = std.scale_, lasso.coef_
            self.intercept_[hour] = lasso.intercept_
        return self

    def coefficients(self) -> pd.DataFrame:
        """Hour x input matrix of LASSO coefficients on standardised inputs (0 = dropped)."""
        return pd.DataFrame(
            self.coef_, index=pd.RangeIndex(24, name="hour_local"), columns=self.columns_
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        X = self._design(features)[self.columns_].to_numpy()
        h = features["hour_local"].to_numpy()
        X = np.where(np.isnan(X), self.fill_[h], X)
        z = self.intercept_[h] + np.sum(
            (X - self.mean_[h]) / self.scale_[h] * self.coef_[h], axis=1
        )
        return pd.Series(self.scaler_.inverse_transform(z), index=features.index, name=self.name)
