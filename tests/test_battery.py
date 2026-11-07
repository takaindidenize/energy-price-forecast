from math import sqrt

import numpy as np
import pytest

from price_forecast.battery import Battery, profit, schedule

EFF = 0.85
ONE_WAY = sqrt(EFF)


def _soc(net: np.ndarray) -> np.ndarray:
    """State of charge after each hour, from net energy sold."""
    charged = np.clip(-net, 0, None) * ONE_WAY
    taken = np.clip(net, 0, None) / ONE_WAY
    return np.cumsum(charged - taken)


def test_buys_cheap_sells_dear_within_power_and_capacity():
    prices = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 100.0])
    net = schedule(prices)
    assert net[:3].sum() < 0 and net[3:].sum() > 0
    assert np.abs(net).max() <= 1.0 + 1e-6
    soc = _soc(net)
    assert soc.min() >= -1e-6 and soc.max() <= 2.0 + 1e-6
    assert soc[-1] == pytest.approx(0.0, abs=1e-6)


def test_profit_matches_a_hand_calculation():
    # Two hours of charging at 1 MW buy 2 MWh, of which 85% comes back: 1.7 MWh sold.
    prices = np.array([10.0, 10.0, 100.0, 100.0])
    net = schedule(prices)
    assert profit(net, prices) == pytest.approx(1.7 * 100 - 2 * 10, abs=1e-4)


def test_does_not_trade_when_the_spread_does_not_cover_the_losses():
    # 85% round trip needs the sell price above 100 / 0.85 = 117.6.
    assert np.abs(schedule(np.array([100.0, 100.0, 110.0, 110.0]))).sum() == 0
    assert np.abs(schedule(np.zeros(24))).sum() == 0


def test_one_cycle_a_day_limits_two_price_peaks():
    prices = np.array([0.0, 0.0, 200.0, 200.0, 0.0, 0.0, 200.0, 200.0])
    one = schedule(prices, Battery(cycles_per_day=1.0))
    two = schedule(prices, Battery(cycles_per_day=2.0))
    # Energy taken out of the battery over the day, against 2 MWh per cycle.
    assert (np.clip(one, 0, None) / ONE_WAY).sum() <= 2.0 + 1e-6
    assert profit(two, prices) > profit(one, prices)


def test_never_charges_and_discharges_at_once_even_when_paid_to_burn_energy():
    # In a single hour, charging and discharging at once would earn money by burning energy at a
    # negative price. One direction per hour leaves nothing to do, since it must end empty.
    assert schedule(np.array([-500.0])).tolist() == [0.0]


def test_daylight_saving_days_have_other_lengths():
    for hours in (23, 25):
        prices = np.resize([10.0, 200.0], hours)
        assert len(schedule(prices)) == hours


def test_realised_prices_decide_the_money_not_the_forecast():
    forecast = np.array([10.0, 10.0, 100.0, 100.0])
    wrong_way = np.array([100.0, 100.0, 10.0, 10.0])
    assert profit(schedule(forecast), wrong_way) < 0


def test_nan_is_refused():
    with pytest.raises(ValueError):
        schedule(np.array([1.0, np.nan]))
