"""A price-taking battery on the day-ahead market: what is a price forecast worth in money?

A schedule for one delivery day is chosen from forecast prices and paid at realised prices. The
battery starts and ends each day empty, so days are independent, and it never charges and
discharges in the same hour.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

# Without a small cost per MWh, hours that gain nothing (a flat forecast at zero) could be traded
# at random. Small enough never to change a schedule that gains anything real.
_THROUGHPUT_COST = 1e-6


@dataclass(frozen=True)
class Battery:
    power_mw: float = 1.0
    capacity_mwh: float = 2.0
    round_trip_efficiency: float = 0.85
    cycles_per_day: float = 1.0


def schedule(prices: np.ndarray, battery: Battery | None = None) -> np.ndarray:
    """Net energy sold to the market per hour (MWh; negative = charging) that maximises profit at
    `prices`. Hourly steps, one value per price, so 23- and 25-hour days work too."""
    battery = battery or Battery()
    p = np.asarray(prices, dtype=float)
    if np.isnan(p).any():
        raise ValueError("prices contain NaN")
    n = len(p)
    one_way = sqrt(battery.round_trip_efficiency)
    # Variables, hour by hour: charge[n], discharge[n], is_charging[n].
    objective = np.concatenate(
        [p + _THROUGHPUT_COST, -p + _THROUGHPUT_COST, np.zeros(n)]
    )  # milp minimises

    # State of charge after each hour stays within 0..capacity.
    soc = lil_matrix((n, 3 * n))
    for t in range(n):
        for i in range(t + 1):
            soc[t, i] = one_way
            soc[t, n + i] = -1.0 / one_way
    lower = np.zeros(n)
    upper = np.full(n, battery.capacity_mwh)
    upper[-1] = 0.0  # empty at the end of the day

    # Energy taken out of the battery over the day is limited by the cycle allowance.
    cycles = lil_matrix((1, 3 * n))
    for i in range(n):
        cycles[0, n + i] = 1.0 / one_way

    # charge <= power * is_charging, discharge <= power * (1 - is_charging)
    exclusive = lil_matrix((2 * n, 3 * n))
    for t in range(n):
        exclusive[t, t] = 1.0
        exclusive[t, 2 * n + t] = -battery.power_mw
        exclusive[n + t, n + t] = 1.0
        exclusive[n + t, 2 * n + t] = battery.power_mw
    exclusive_upper = np.concatenate([np.zeros(n), np.full(n, battery.power_mw)])

    result = milp(
        c=objective,
        constraints=[
            LinearConstraint(soc.tocsr(), lower, upper),
            LinearConstraint(cycles.tocsr(), 0.0, battery.cycles_per_day * battery.capacity_mwh),
            LinearConstraint(exclusive.tocsr(), -np.inf, exclusive_upper),
        ],
        integrality=np.concatenate([np.zeros(2 * n), np.ones(n)]),
        bounds=Bounds(np.zeros(3 * n), np.concatenate([np.full(2 * n, np.inf), np.ones(n)])),
    )
    if not result.success:
        raise RuntimeError(f"battery schedule not solved: {result.message}")
    return np.round(result.x[n : 2 * n] - result.x[:n], 9)


def profit(net_sold: np.ndarray, realised: np.ndarray) -> float:
    """Money earned by a schedule at realised prices (negative prices pay for charging)."""
    return float(np.dot(net_sold, realised))
