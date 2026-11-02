"""Point-forecast errors in EUR/MWh.

MAPE is deliberately absent: prices cross zero, so percentage errors explode or flip sign.
rMAE (Lago et al. 2021) divides a model's MAE by the MAE of the weekly seasonal naive forecast on
the same rows, so it is comparable across markets with very different price levels: below 1 beats
the benchmark.

Every function ignores rows where any of its inputs is missing, so all of them see the same rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def _aligned(*arrays: ArrayLike) -> list[np.ndarray]:
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    if len({a.shape for a in arrs}) != 1:
        raise ValueError(f"inputs differ in shape: {[a.shape for a in arrs]}")
    mask = np.logical_and.reduce([np.isfinite(a) for a in arrs])
    return [a[mask] for a in arrs]


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    t, p = _aligned(y_true, y_pred)
    return float(np.mean(np.abs(t - p))) if t.size else float("nan")


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    t, p = _aligned(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2))) if t.size else float("nan")


def rmae(y_true: ArrayLike, y_pred: ArrayLike, y_seasonal_naive: ArrayLike) -> float:
    """MAE of `y_pred` divided by the MAE of `y_seasonal_naive`, both on the same rows."""
    t, p, s = _aligned(y_true, y_pred, y_seasonal_naive)
    denom = mae(t, s)
    if not t.size or denom == 0:
        return float("nan")
    return mae(t, p) / denom


def evaluate(y_true: ArrayLike, y_pred: ArrayLike, y_seasonal_naive: ArrayLike) -> dict[str, float]:
    """n, MAE, RMSE and rMAE, all on the rows where truth, forecast and benchmark are known."""
    t, p, s = _aligned(y_true, y_pred, y_seasonal_naive)
    return {"n": int(t.size), "mae": mae(t, p), "rmse": rmse(t, p), "rmae": rmae(t, p, s)}
