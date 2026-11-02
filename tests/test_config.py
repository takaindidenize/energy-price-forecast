from pathlib import Path

from price_forecast import config


def test_markets_are_consistent():
    for code, market in config.MARKETS.items():
        assert market.code == code
        assert market.weather_points
        assert 0 <= market.issue_hour_local < 12


def test_month_partition_path():
    path = config.month_partition_path(Path("base"), "tr", "2025-03")
    assert path == Path("base/market=tr/year=2025/month=03/data.parquet")
