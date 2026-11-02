"""EPİAŞ ingestion, fully mocked with respx. Credentials here are fake."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from urllib.parse import parse_qs

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tenacity import wait_none

from price_forecast import config
from price_forecast.ingestion import epias

FAKE_USER = "fake-user@example.com"
FAKE_PASSWORD = "fake-password-123"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv(epias.USERNAME_ENV, FAKE_USER)
    monkeypatch.setenv(epias.PASSWORD_ENV, FAKE_PASSWORD)
    monkeypatch.setattr(config, "PRICES_RAW_DIR", tmp_path / "prices")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_client(clock=None) -> epias.EpiasClient:
    kwargs = {"wait": wait_none()}
    if clock is not None:
        kwargs["clock"] = clock
    return epias.EpiasClient(**kwargs)


def items_for(start: date, end: date, *, eur: bool = True) -> list[dict]:
    """24 hourly items per local day, shaped like the documented MCP response."""
    items = []
    day = start
    while day <= end:
        for hour in range(24):
            price = 1000.0 + day.day * 10 + hour
            items.append(
                {
                    "date": f"{day.isoformat()}T{hour:02d}:00:00+03:00",
                    "hour": f"{hour:02d}:00",
                    "price": price,
                    "priceUsd": round(price / 30, 2),
                    "priceEur": round(price / 33, 2) if eur else None,
                }
            )
        day += timedelta(days=1)
    return items


def mcp_from_body(request: httpx.Request) -> httpx.Response:
    """Answer an MCP request with full data for the requested dates (endDate inclusive)."""
    body = json.loads(request.content)
    start = datetime.fromisoformat(body["startDate"]).date()
    end = datetime.fromisoformat(body["endDate"]).date()
    return httpx.Response(200, json={"items": items_for(start, end)})


def ticket(value: str = "TGT-1-fake") -> httpx.Response:
    return httpx.Response(201, text=value)


@pytest.fixture
def ticket_route(respx_mock):
    return respx_mock.post(config.EPIAS_TICKET_URL).mock(return_value=ticket())


@pytest.fixture
def mcp_route(respx_mock):
    return respx_mock.post(epias.MCP_URL).mock(side_effect=mcp_from_body)


# -- tickets -----------------------------------------------------------------------------------


def test_ticket_creation_sends_form_credentials_and_uses_tgt_header(ticket_route, mcp_route):
    with make_client() as client:
        items = client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))

    assert len(items) == 24
    login = ticket_route.calls.last.request
    assert parse_qs(login.content.decode()) == {
        "username": [FAKE_USER],
        "password": [FAKE_PASSWORD],
    }
    assert login.headers["Accept"] == "text/plain"
    data_request = mcp_route.calls.last.request
    assert data_request.headers["TGT"] == "TGT-1-fake"
    assert json.loads(data_request.content) == {
        "startDate": "2024-01-01T00:00:00+03:00",
        "endDate": "2024-01-01T00:00:00+03:00",
    }


def test_ticket_from_location_header(respx_mock, mcp_route):
    respx_mock.post(config.EPIAS_TICKET_URL).mock(
        return_value=httpx.Response(
            201,
            text="<html>created</html>",
            headers={"Location": f"{config.EPIAS_TICKET_URL}/TGT-9-loc"},
        )
    )
    with make_client() as client:
        client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))
    assert mcp_route.calls.last.request.headers["TGT"] == "TGT-9-loc"


def test_ticket_is_reused_until_near_expiry(ticket_route, mcp_route):
    ticket_route.mock(side_effect=[ticket("TGT-1-fake"), ticket("TGT-2-fake")])
    clock = FakeClock()
    with make_client(clock) as client:
        client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))
        clock.now += 60 * 60  # 1 h: still fresh
        client.fetch_mcp(date(2024, 1, 2), date(2024, 1, 2))
        assert ticket_route.call_count == 1

        clock.now += 50 * 60  # 1 h 50 min: inside the renewal margin of the 2 h lifetime
        client.fetch_mcp(date(2024, 1, 3), date(2024, 1, 3))

    assert ticket_route.call_count == 2
    assert [c.request.headers["TGT"] for c in mcp_route.calls] == [
        "TGT-1-fake",
        "TGT-1-fake",
        "TGT-2-fake",
    ]


def test_ticket_renewed_once_after_401(ticket_route, mcp_route):
    ticket_route.mock(side_effect=[ticket("TGT-1-fake"), ticket("TGT-2-fake")])
    mcp_route.mock(
        side_effect=[
            httpx.Response(401),
            httpx.Response(200, json={"items": items_for(date(2024, 1, 1), date(2024, 1, 1))}),
        ]
    )
    with make_client() as client:
        items = client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))

    assert len(items) == 24
    assert ticket_route.call_count == 2
    assert [c.request.headers["TGT"] for c in mcp_route.calls] == ["TGT-1-fake", "TGT-2-fake"]


def test_second_401_with_fresh_ticket_fails(ticket_route, mcp_route):
    mcp_route.mock(return_value=httpx.Response(401))
    with make_client() as client, pytest.raises(epias.EpiasAuthError, match="fresh ticket"):
        client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))
    assert ticket_route.call_count == 2
    assert mcp_route.call_count == 2


def test_rejected_login_is_clear_and_does_not_leak(respx_mock, caplog):
    respx_mock.post(config.EPIAS_TICKET_URL).mock(
        return_value=httpx.Response(401, text=f"bad credentials for {FAKE_USER}")
    )
    caplog.set_level(logging.DEBUG)
    with make_client() as client, pytest.raises(epias.EpiasAuthError) as excinfo:
        client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))

    message = str(excinfo.value)
    assert "EPIAS_USERNAME" in message and "401" in message
    for secret in (FAKE_USER, FAKE_PASSWORD):
        assert secret not in message
        assert secret not in caplog.text
    assert FAKE_USER not in repr(client) and FAKE_PASSWORD not in repr(client)


def test_201_without_ticket_is_an_error(respx_mock):
    respx_mock.post(config.EPIAS_TICKET_URL).mock(return_value=httpx.Response(201, text="ok"))
    with make_client() as client, pytest.raises(epias.EpiasAuthError, match="no TGT"):
        client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))


@pytest.mark.parametrize("missing", [epias.USERNAME_ENV, epias.PASSWORD_ENV])
def test_missing_credentials(monkeypatch, respx_mock, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(epias.EpiasCredentialsError, match=missing):
        epias.ingest_month("2024-01")
    assert not respx_mock.calls


def test_empty_credentials_count_as_missing(monkeypatch):
    monkeypatch.setenv(epias.PASSWORD_ENV, "")
    with pytest.raises(epias.EpiasCredentialsError, match=epias.PASSWORD_ENV):
        epias.EpiasClient()


# -- retries -----------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retries_transient_errors(ticket_route, mcp_route, status):
    mcp_route.mock(side_effect=[httpx.Response(status), httpx.Response(status), mcp_from_body])
    with make_client() as client:
        assert len(client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))) == 24
    assert mcp_route.call_count == 3


def test_retries_network_errors(ticket_route, mcp_route):
    mcp_route.mock(side_effect=[httpx.ConnectError("boom"), mcp_from_body])
    with make_client() as client:
        assert len(client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))) == 24


def test_gives_up_after_max_attempts(ticket_route, mcp_route):
    mcp_route.mock(return_value=httpx.Response(503))
    with make_client() as client, pytest.raises(epias.EpiasError, match="HTTP 503"):
        client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))
    assert mcp_route.call_count == epias.MAX_ATTEMPTS


def test_rate_limit_waits_as_long_as_the_server_says(ticket_route, mcp_route):
    """The live API answers 429 with `RateLimit-Reset` (seconds until the 60-second window ends)."""
    limited = httpx.Response(429, headers={"RateLimit-Limit": "60;w=60", "RateLimit-Reset": "7"})
    mcp_route.mock(side_effect=[limited, mcp_from_body])
    sleeps: list[float] = []
    with epias.EpiasClient(wait=wait_none(), sleep=sleeps.append) as client:
        assert len(client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))) == 24
    assert sleeps == [8.0]  # the server's 7 s plus a one-second margin


def test_retry_after_header_is_preferred_and_capped(ticket_route, mcp_route):
    limited = httpx.Response(429, headers={"Retry-After": "500", "RateLimit-Reset": "3"})
    mcp_route.mock(side_effect=[limited, mcp_from_body])
    sleeps: list[float] = []
    with epias.EpiasClient(wait=wait_none(), sleep=sleeps.append) as client:
        client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))
    assert sleeps == [epias.MAX_SERVER_WAIT_SECONDS]


def test_pauses_when_the_rate_limit_window_is_used_up(ticket_route, mcp_route):
    def last_request_in_window(request: httpx.Request) -> httpx.Response:
        response = mcp_from_body(request)
        response.headers.update({"RateLimit-Remaining": "0", "RateLimit-Reset": "12"})
        return response

    mcp_route.mock(side_effect=last_request_in_window)
    sleeps: list[float] = []
    with epias.EpiasClient(wait=wait_none(), sleep=sleeps.append) as client:
        client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))
    assert sleeps == [13.0]


def test_client_error_is_not_retried(ticket_route, mcp_route):
    mcp_route.mock(return_value=httpx.Response(400, json={"message": "bad request"}))
    with make_client() as client, pytest.raises(epias.EpiasError, match="HTTP 400"):
        client.fetch_mcp(date(2024, 1, 1), date(2024, 1, 1))
    assert mcp_route.call_count == 1


# -- ingest_month ------------------------------------------------------------------------------


def read_partition(month: str) -> pa.Table:
    return pq.read_table(config.month_partition_path(config.PRICES_RAW_DIR, "tr", month))


def test_chunks_cover_the_range_without_gaps():
    chunks = list(epias._chunks(date(2024, 2, 1), date(2024, 2, 29), 7))
    assert chunks[0] == (date(2024, 2, 1), date(2024, 2, 7))
    assert chunks[-1] == (date(2024, 2, 29), date(2024, 2, 29))
    assert all(b[0] == a[1] + timedelta(days=1) for a, b in zip(chunks, chunks[1:], strict=False))


def test_month_is_one_request(ticket_route, mcp_route):
    """The live API returns a whole year in one request, so a month needs exactly one."""
    with make_client() as client:
        path = epias.ingest_month("2024-02", client=client)

    assert path == config.month_partition_path(config.PRICES_RAW_DIR, "tr", "2024-02")
    bodies = [json.loads(c.request.content) for c in mcp_route.calls]
    assert [(b["startDate"][:10], b["endDate"][:10]) for b in bodies] == [
        ("2024-02-01", "2024-02-29"),
    ]
    assert ticket_route.call_count == 1
    table = read_partition("2024-02")
    assert table.num_rows == 29 * 24
    starts = table.column("delivery_start").to_pylist()
    assert len(set(starts)) == len(starts)
    assert starts == sorted(starts)


def test_utc_conversion_and_contract(ticket_route, mcp_route):
    with make_client() as client:
        epias.ingest_month("2024-01", client=client)

    table = read_partition("2024-01")
    assert table.schema.names == [
        "delivery_start",
        "resolution_minutes",
        "price",
        "price_eur",
        "currency",
        "source",
        "loaded_at",
    ]
    assert table.schema.field("delivery_start").type == pa.timestamp("us")
    assert table.schema.field("delivery_start").type.tz is None
    assert table.schema.field("resolution_minutes").type == pa.int16()
    assert table.schema.field("loaded_at").type == pa.timestamp("us")

    rows = table.to_pylist()
    # Local 2024-01-01 00:00 (+03:00) is 2023-12-31 21:00 UTC; local 2024-01-31 23:00 is 20:00 UTC.
    assert rows[0]["delivery_start"] == datetime(2023, 12, 31, 21)
    assert rows[-1]["delivery_start"] == datetime(2024, 1, 31, 20)
    assert rows[0]["price"] == 1010.0  # TRY, from `price`
    assert rows[0]["price_eur"] == round(1010.0 / 33, 2)  # from `priceEur`, not `priceUsd`
    assert {r["currency"] for r in rows} == {"TRY"}
    assert {r["source"] for r in rows} == {"epias"}
    assert {r["resolution_minutes"] for r in rows} == {60}
    steps = {
        b["delivery_start"] - a["delivery_start"] for a, b in zip(rows, rows[1:], strict=False)
    }
    assert steps == {timedelta(hours=1)}


def test_date_without_hour_uses_hour_field():
    frame = epias.parse_mcp_items(
        [{"date": "2024-03-05T00:00:00+03:00", "hour": "13:00", "price": 1.0, "priceEur": None}]
    )
    assert frame.loc[0, "delivery_start"] == datetime(2024, 3, 5, 10)


def test_naive_date_is_read_as_turkish_local_time():
    frame = epias.parse_mcp_items([{"date": "2024-03-05T13:00:00", "price": 1.0}])
    assert frame.loc[0, "delivery_start"] == datetime(2024, 3, 5, 10)


def test_missing_eur_price_is_null(ticket_route, mcp_route):
    def without_eur(request):
        body = json.loads(request.content)
        start = datetime.fromisoformat(body["startDate"]).date()
        end = datetime.fromisoformat(body["endDate"]).date()
        return httpx.Response(200, json={"items": items_for(start, end, eur=False)})

    mcp_route.mock(side_effect=without_eur)
    with make_client() as client:
        epias.ingest_month("2024-01", client=client)
    table = read_partition("2024-01")
    assert table.column("price_eur").null_count == table.num_rows
    assert table.column("price").null_count == 0


def test_rows_outside_requested_dates_are_dropped(ticket_route, mcp_route):
    def one_day_too_many(request):
        body = json.loads(request.content)
        start = datetime.fromisoformat(body["startDate"]).date()
        end = datetime.fromisoformat(body["endDate"]).date()
        return httpx.Response(200, json={"items": items_for(start, end + timedelta(days=1))})

    mcp_route.mock(side_effect=one_day_too_many)
    with make_client() as client:
        epias.ingest_month("2024-01", client=client)
    table = read_partition("2024-01")
    assert table.num_rows == 31 * 24


def test_empty_month_is_skipped(ticket_route, mcp_route, caplog):
    mcp_route.mock(return_value=httpx.Response(200, json={"items": []}))
    caplog.set_level(logging.INFO)
    with make_client() as client:
        assert epias.ingest_month("2030-01", client=client) is None
    assert not config.PRICES_RAW_DIR.exists()
    assert "No EPİAŞ PTF data for 2030-01" in caplog.text


def test_partial_month_is_written_with_warning(ticket_route, mcp_route, caplog):
    def first_ten_days(request):
        body = json.loads(request.content)
        start = datetime.fromisoformat(body["startDate"]).date()
        end = min(datetime.fromisoformat(body["endDate"]).date(), date(2024, 1, 10))
        return httpx.Response(200, json={"items": items_for(start, end) if start <= end else []})

    mcp_route.mock(side_effect=first_ten_days)
    with make_client() as client:
        epias.ingest_month("2024-01", client=client)
    assert read_partition("2024-01").num_rows == 10 * 24
    assert "240 of 744 hours present" in caplog.text


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"data": []}),
        httpx.Response(200, json=[]),
        httpx.Response(200, text="<html>maintenance</html>"),
        httpx.Response(200, json={"items": [{"hour": "00:00", "price": 1.0}]}),
        httpx.Response(200, json={"items": [{"date": "2024-01-01T00:00:00+03:00"}]}),
        httpx.Response(200, json={"items": [{"date": "yesterday", "price": 1.0}]}),
        httpx.Response(
            200, json={"items": [{"date": "2024-01-01T00:00:00+03:00", "price": "1,0"}]}
        ),
        httpx.Response(200, json={"items": [{"date": "2024-01-01T00:00:00+03:00", "price": None}]}),
        httpx.Response(
            200,
            json={"items": [{"date": "2024-01-01T05:00:00+03:00", "hour": "06:00", "price": 1.0}]},
        ),
        httpx.Response(200, json={"items": ["not an object"]}),
    ],
)
def test_malformed_response_raises_and_writes_nothing(ticket_route, mcp_route, response):
    mcp_route.mock(return_value=response)
    with make_client() as client, pytest.raises(epias.EpiasResponseError):
        epias.ingest_month("2024-01", client=client)
    assert not config.PRICES_RAW_DIR.exists()


def test_rerun_overwrites_partition_without_leftovers(ticket_route, mcp_route):
    with make_client() as client:
        first = epias.ingest_month("2024-01", client=client)
        mcp_route.mock(
            side_effect=lambda request: httpx.Response(
                200,
                json={
                    "items": [
                        {**item, "price": item["price"] + 1}
                        for item in mcp_from_body(request).json()["items"]
                    ]
                },
            )
        )
        second = epias.ingest_month("2024-01", client=client)

    assert first == second
    assert sorted(p.name for p in first.parent.iterdir()) == ["data.parquet"]
    assert read_partition("2024-01").to_pylist()[0]["price"] == 1011.0


@pytest.mark.parametrize("month", ["2024-1", "2024-13", "24-01", "2024/01"])
def test_invalid_month(month):
    with pytest.raises(ValueError, match="YYYY-MM"):
        epias.ingest_month(month)


# -- CLI ---------------------------------------------------------------------------------------


def test_iter_months_crosses_years():
    assert list(epias.iter_months("2023-11", "2024-02")) == [
        "2023-11",
        "2023-12",
        "2024-01",
        "2024-02",
    ]


def test_cli_ingests_each_month_with_one_ticket(ticket_route, mcp_route):
    assert epias.main(["--start", "2023-12", "--end", "2024-01"]) == 0
    assert read_partition("2023-12").num_rows == 31 * 24
    assert read_partition("2024-01").num_rows == 31 * 24
    assert ticket_route.call_count == 1


def test_cli_missing_credentials(monkeypatch, capsys, respx_mock):
    monkeypatch.delenv(epias.USERNAME_ENV)
    monkeypatch.delenv(epias.PASSWORD_ENV)
    assert epias.main(["--start", "2024-01", "--end", "2024-01"]) == 2
    err = capsys.readouterr().err
    assert "EPIAS_USERNAME" in err and "EPIAS_PASSWORD" in err
    assert not respx_mock.calls


def test_cli_rejects_reversed_range():
    with pytest.raises(SystemExit):
        epias.main(["--start", "2024-02", "--end", "2024-01"])
