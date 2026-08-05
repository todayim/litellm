"""
Unit tests for the auto-router per-session benchmarks rollup writer.

The classification SQL itself runs against a real Postgres in
tests/proxy_behavior/spend/test_autorouter_session_rollup.py; these tests cover the
request-time transaction builder and the flush contract with an injected fake client.
"""

import asyncio
import json
from datetime import datetime

import httpx
import pytest

from litellm.proxy.db.autorouter_session_rollup import (
    AutoRouterTurnTransaction,
    UPSERT_AUTOROUTER_SESSION_SQL,
    build_autorouter_turn_transaction,
    flush_autorouter_turn_transactions,
)

ROUTING_DECISION = {"router_model_name": "live-auto", "router_type": "complexity", "routed_model": "haiku"}


def _payload(**overrides: object) -> dict:
    base: dict = {
        "status": "success",
        "api_key": "hashed-key",
        "session_id": "session-1",
        "model": "bedrock/haiku",
        "model_group": "live-auto",
        "startTime": "2026-08-01T12:00:00",
        "spend": 0.01,
        "prompt_tokens": 90,
        "completion_tokens": 10,
    }
    base.update(overrides)
    return base


def _metadata(**overrides: object) -> dict:
    base: dict = {"routing_decision": dict(ROUTING_DECISION), "usage_object": {"prompt_tokens": 90}}
    base.update(overrides)
    return base


def _build(payload: dict | None = None, metadata: dict | None = None):
    return build_autorouter_turn_transaction(
        payload=payload if payload is not None else _payload(),
        metadata=metadata if metadata is not None else _metadata(),
        saved_spend=0.02,
    )


class TestBuildTransaction:
    def test_successful_auto_routed_turn_builds_every_field(self):
        transaction = _build(
            metadata=_metadata(
                usage_object={"prompt_tokens": 90, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 7}
            )
        )
        assert transaction == AutoRouterTurnTransaction(
            api_key="hashed-key",
            session_id="session-1",
            router_name="live-auto",
            router_type="complexity",
            model="bedrock/haiku",
            turn_at=datetime(2026, 8, 1, 12, 0, 0),
            total_tokens=100,
            spend=0.01,
            saved_spend=0.02,
            covered=True,
            cache_hit=True,
            cache_ttl_seconds=300,
        )

    @pytest.mark.parametrize(
        "payload_overrides",
        [
            {"status": "failure"},
            {"api_key": ""},
            {"session_id": None},
            {"model": ""},
            {"startTime": "not-a-time"},
        ],
    )
    def test_incomplete_payloads_are_skipped(self, payload_overrides: dict):
        assert _build(payload=_payload(**payload_overrides)) is None

    @pytest.mark.parametrize("metadata", [{}, {"routing_decision": None}, {"routing_decision": {}}])
    def test_requests_without_a_routing_decision_are_skipped(self, metadata: dict):
        assert _build(metadata=metadata) is None

    def test_router_name_falls_back_to_the_payload_model_group(self):
        transaction = _build(metadata=_metadata(routing_decision={"router_type": "complexity"}))
        assert transaction is not None and transaction.router_name == "live-auto"

    def test_one_hour_ttl_detail_beats_the_five_minute_default(self):
        metadata = _metadata(
            usage_object={
                "prompt_tokens": 90,
                "cache_creation_input_tokens": 4,
                "prompt_tokens_details": {"cache_creation_token_details": {"ephemeral_1h_input_tokens": 4}},
            }
        )
        transaction = _build(metadata=metadata)
        assert transaction is not None and transaction.cache_ttl_seconds == 3600

    def test_a_cache_write_without_ttl_detail_is_the_provider_default_five_minutes(self):
        transaction = _build(metadata=_metadata(usage_object={"prompt_tokens": 90, "cache_creation_input_tokens": 12}))
        assert transaction is not None and transaction.cache_ttl_seconds == 300

    def test_a_turn_that_wrote_nothing_records_no_ttl(self):
        transaction = _build()
        assert transaction is not None and transaction.cache_ttl_seconds is None

    def test_a_turn_without_usage_telemetry_is_uncovered(self):
        transaction = _build(metadata=_metadata(usage_object={}))
        assert transaction is not None
        assert transaction.covered is False
        assert transaction.cache_ttl_seconds is None

    def test_timezone_aware_start_times_normalize_to_utc(self):
        transaction = _build(payload=_payload(startTime="2026-08-01T14:00:00+02:00"))
        assert transaction is not None and transaction.turn_at == datetime(2026, 8, 1, 12, 0, 0)


class _FakeDB:
    def __init__(self, failures: "list[Exception] | None" = None):
        self.calls: list[tuple] = []
        self._failures = list(failures or [])

    async def execute_raw(self, sql: str, *params: object) -> int:
        if self._failures:
            raise self._failures.pop(0)
        self.calls.append((sql, params))
        return 1


class _FakeClient:
    def __init__(self, failures: "list[Exception] | None" = None):
        self.db = _FakeDB(failures)


def _transaction(session_id: str = "s1", at: datetime = datetime(2026, 8, 1, 12, 0, 0)) -> AutoRouterTurnTransaction:
    return AutoRouterTurnTransaction(
        api_key="k1",
        session_id=session_id,
        router_name="live-auto",
        router_type="complexity",
        model="bedrock/haiku",
        turn_at=at,
        total_tokens=100,
        spend=0.01,
        saved_spend=0.02,
        covered=True,
        cache_hit=False,
        cache_ttl_seconds=None,
    )


class TestFlush:
    def test_turns_replay_in_per_session_event_order(self):
        client = _FakeClient()
        first = _transaction(at=datetime(2026, 8, 1, 12, 0, 0))
        second = _transaction(at=datetime(2026, 8, 1, 12, 0, 10))
        asyncio.run(flush_autorouter_turn_transactions(client, [second, first]))
        sent_times = [params[5] for _, params in client.db.calls]
        assert sent_times == ["2026-08-01T12:00:00", "2026-08-01T12:00:10"]

    def test_params_marshal_in_statement_order(self):
        client = _FakeClient()
        asyncio.run(flush_autorouter_turn_transactions(client, [_transaction()]))
        sql, params = client.db.calls[0]
        assert sql == UPSERT_AUTOROUTER_SESSION_SQL
        assert params == (
            "k1", "s1", "live-auto", "complexity", "bedrock/haiku",
            "2026-08-01T12:00:00", 100, 0.01, 0.02, 1, 0, None,
        )

    def test_a_connect_error_retries_the_same_statement(self):
        client = _FakeClient(failures=[httpx.ConnectError("boom")])
        asyncio.run(flush_autorouter_turn_transactions(client, [_transaction()]))
        assert len(client.db.calls) == 1

    def test_an_ambiguous_failure_drops_the_rest_of_the_batch(self):
        client = _FakeClient(failures=[RuntimeError("read timeout")])
        transactions = [_transaction(session_id="s1"), _transaction(session_id="s2")]
        asyncio.run(flush_autorouter_turn_transactions(client, transactions))
        assert client.db.calls == []

    def test_an_empty_batch_writes_nothing(self):
        client = _FakeClient()
        asyncio.run(flush_autorouter_turn_transactions(client, []))
        assert client.db.calls == []


class TestEnqueueSeam:
    @pytest.mark.asyncio
    async def test_update_database_seam_enqueues_only_auto_routed_success(self, monkeypatch: pytest.MonkeyPatch):
        import litellm
        from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
        from litellm.proxy.utils import PrismaClient

        monkeypatch.setattr(litellm, "autorouter_savings_baseline_model", None)
        monkeypatch.setattr(PrismaClient, "autorouter_turn_transactions", [])
        writer = DBSpendUpdateWriter()
        fake_prisma = type("P", (), {})()
        fake_prisma._autorouter_turn_transactions_lock = asyncio.Lock()
        fake_prisma.autorouter_turn_transactions = []

        routed = _payload()
        routed["metadata"] = json.dumps(_metadata())
        await writer._enqueue_autorouter_turn_transaction(payload=routed, prisma_client=fake_prisma)

        plain = _payload()
        plain["metadata"] = json.dumps({"usage_object": {"prompt_tokens": 9}})
        await writer._enqueue_autorouter_turn_transaction(payload=plain, prisma_client=fake_prisma)

        assert [t.router_name for t in fake_prisma.autorouter_turn_transactions] == ["live-auto"]
        assert fake_prisma.autorouter_turn_transactions[0].saved_spend == 0.0
