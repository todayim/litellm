"""
Per-session auto-router benchmarks rollup.

At request time the spend writer builds one AutoRouterTurnTransaction per successful
auto-routed request (a request whose metadata carries a routing_decision) and queues it
on the prisma client. The spend-log flush job drains the queue into
LiteLLM_AutoRouterSession with one conditional upsert per turn: the statement classifies
the turn (same model, first visit, return to a model the session already used, out of
order) against the row's own columns, so nothing is read before the write and concurrent
pods compose. The benchmarks endpoint aggregates these rows and never touches
LiteLLM_SpendLogs.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import DB_RETRY_SAFE_ERROR_TYPES

if TYPE_CHECKING:
    from litellm.proxy._types import SpendLogsPayload
    from litellm.proxy.utils import PrismaClient

CACHE_TTL_5M_SECONDS: Final = 300
CACHE_TTL_1H_SECONDS: Final = 3600


@dataclass(frozen=True, slots=True)
class AutoRouterTurnTransaction:
    api_key: str
    session_id: str
    router_name: str
    router_type: str
    model: str
    turn_at: datetime
    total_tokens: int
    spend: float
    saved_spend: float
    covered: bool
    cache_hit: bool
    cache_ttl_seconds: int | None


def _turn_time_utc(start_time_iso: str) -> datetime | None:
    try:
        parsed: Final = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _cache_ttl_seconds(usage_object: Mapping[str, object] | None) -> int | None:
    """The TTL this turn's cache write used, or None when nothing was written.

    Providers that report a TTL split do so under prompt_tokens_details; a write with no
    split is the provider's default five-minute cache.
    """
    from litellm.proxy.spend_tracking.savings import extract_cache_creation_tokens

    if not usage_object:
        return None
    details: Final = usage_object.get("prompt_tokens_details")
    creation: Final = details.get("cache_creation_token_details") if isinstance(details, Mapping) else None
    if isinstance(creation, Mapping):
        if creation.get("ephemeral_1h_input_tokens"):
            return CACHE_TTL_1H_SECONDS
        if creation.get("ephemeral_5m_input_tokens"):
            return CACHE_TTL_5M_SECONDS
    if extract_cache_creation_tokens(usage_object) > 0:
        return CACHE_TTL_5M_SECONDS
    return None


def build_autorouter_turn_transaction(
    payload: SpendLogsPayload,
    metadata: Mapping[str, object],
    saved_spend: float,
) -> AutoRouterTurnTransaction | None:
    """One rollup transaction for a successful auto-routed turn, else None.

    The routing_decision record is what says a request was auto-routed at all, so a
    request without one (including the auto-router's own classifier sub-calls) never
    reaches the rollup. Failed requests served nothing and are excluded. Cache facts
    are derived from the payload's own usage record through the savings owner, never
    handed in beside it.
    """
    if payload.get("status") != "success":
        return None
    routing_decision: Final = metadata.get("routing_decision")
    if not isinstance(routing_decision, Mapping) or not routing_decision:
        return None
    router_name: Final = routing_decision.get("router_model_name") or payload.get("model_group")
    api_key: Final = payload.get("api_key")
    session_id: Final = payload.get("session_id")
    model: Final = payload.get("model")
    if not (
        isinstance(router_name, str)
        and router_name
        and isinstance(api_key, str)
        and api_key
        and isinstance(session_id, str)
        and session_id
        and isinstance(model, str)
        and model
    ):
        return None
    turn_at: Final = _turn_time_utc(str(payload.get("startTime") or ""))
    if turn_at is None:
        return None
    from litellm.proxy.spend_tracking.savings import extract_cache_read_tokens

    usage_object_raw: Final = metadata.get("usage_object")
    usage_object: Final = usage_object_raw if isinstance(usage_object_raw, Mapping) else None
    covered: Final = bool(usage_object)
    return AutoRouterTurnTransaction(
        api_key=api_key,
        session_id=session_id,
        router_name=router_name,
        router_type=str(routing_decision.get("router_type") or "unknown"),
        model=model,
        turn_at=turn_at,
        total_tokens=int(payload.get("prompt_tokens") or 0) + int(payload.get("completion_tokens") or 0),
        spend=float(payload.get("spend") or 0.0),
        saved_spend=saved_spend,
        covered=covered,
        cache_hit=extract_cache_read_tokens(usage_object) > 0,
        cache_ttl_seconds=_cache_ttl_seconds(usage_object),
    )


_IN_ORDER: Final = "$6::timestamp >= t.last_turn_at"
_SAME: Final = f"{_IN_ORDER} AND t.last_model = $5"
_FIRST: Final = f"{_IN_ORDER} AND NOT t.models ? $5"
_RETURN: Final = f"{_IN_ORDER} AND t.models ? $5 AND t.last_model <> $5"
_RETURN_MISS: Final = f"{_RETURN} AND $10::int = 1 AND $11::int = 0 AND (t.models -> $5 ->> 'ttl') IS NOT NULL"
_IDLE_SECONDS: Final = "EXTRACT(EPOCH FROM $6::timestamp) - (t.models -> $5 ->> 'at')::float8"

UPSERT_AUTOROUTER_SESSION_SQL: Final = f"""
INSERT INTO "LiteLLM_AutoRouterSession" AS t (
    api_key, session_id, router_name, router_type, first_turn_at, last_turn_at,
    last_model, models, turns, unordered_turns, covered_turns,
    same_model_turns, same_model_hits, first_visit_turns, first_visit_hits,
    return_turns, return_hits, return_expired_misses, return_prefix_misses,
    ttl_5m_turns, ttl_1h_turns, total_tokens, spend, saved_spend
)
VALUES (
    $1, $2, $3, $4, $6::timestamp, $6::timestamp,
    $5, jsonb_build_object($5, jsonb_build_object('at', EXTRACT(EPOCH FROM $6::timestamp), 'ttl', $12::int)),
    1, 0, $10::int,
    0, 0, 1, $11::int,
    0, 0, 0, 0,
    (CASE WHEN $12::int = {CACHE_TTL_5M_SECONDS} THEN 1 ELSE 0 END),
    (CASE WHEN $12::int = {CACHE_TTL_1H_SECONDS} THEN 1 ELSE 0 END),
    $7::bigint, $8::float8, $9::float8
)
ON CONFLICT (api_key, session_id, router_name) DO UPDATE SET
    turns = t.turns + 1,
    total_tokens = t.total_tokens + EXCLUDED.total_tokens,
    spend = t.spend + EXCLUDED.spend,
    saved_spend = t.saved_spend + EXCLUDED.saved_spend,
    covered_turns = t.covered_turns + EXCLUDED.covered_turns,
    ttl_5m_turns = t.ttl_5m_turns + EXCLUDED.ttl_5m_turns,
    ttl_1h_turns = t.ttl_1h_turns + EXCLUDED.ttl_1h_turns,
    unordered_turns = t.unordered_turns + (CASE WHEN NOT ({_IN_ORDER}) THEN 1 ELSE 0 END),
    same_model_turns = t.same_model_turns + (CASE WHEN {_SAME} THEN 1 ELSE 0 END),
    same_model_hits = t.same_model_hits + (CASE WHEN {_SAME} AND $11::int = 1 THEN 1 ELSE 0 END),
    first_visit_turns = t.first_visit_turns + (CASE WHEN {_FIRST} THEN 1 ELSE 0 END),
    first_visit_hits = t.first_visit_hits + (CASE WHEN {_FIRST} AND $11::int = 1 THEN 1 ELSE 0 END),
    return_turns = t.return_turns + (CASE WHEN {_RETURN} THEN 1 ELSE 0 END),
    return_hits = t.return_hits + (CASE WHEN {_RETURN} AND $11::int = 1 THEN 1 ELSE 0 END),
    return_expired_misses = t.return_expired_misses
        + (CASE WHEN {_RETURN_MISS} AND {_IDLE_SECONDS} > (t.models -> $5 ->> 'ttl')::float8 THEN 1 ELSE 0 END),
    return_prefix_misses = t.return_prefix_misses
        + (CASE WHEN {_RETURN_MISS} AND {_IDLE_SECONDS} <= (t.models -> $5 ->> 'ttl')::float8 THEN 1 ELSE 0 END),
    models = t.models || jsonb_build_object($5, jsonb_build_object(
        'at', GREATEST(COALESCE((t.models -> $5 ->> 'at')::float8, 0), EXTRACT(EPOCH FROM $6::timestamp)),
        'ttl', (CASE WHEN {_IN_ORDER}
                THEN COALESCE($12::int, (t.models -> $5 ->> 'ttl')::int)
                ELSE COALESCE((t.models -> $5 ->> 'ttl')::int, $12::int) END)
    )),
    last_model = (CASE WHEN {_IN_ORDER} THEN $5 ELSE t.last_model END),
    first_turn_at = LEAST(t.first_turn_at, EXCLUDED.first_turn_at),
    last_turn_at = GREATEST(t.last_turn_at, EXCLUDED.last_turn_at)
"""


async def _upsert_turn_with_retry(
    prisma_client: PrismaClient,
    transaction: AutoRouterTurnTransaction,
    n_retry_times: int,
) -> None:
    for attempt in range(n_retry_times + 1):
        try:
            await prisma_client.db.execute_raw(
                UPSERT_AUTOROUTER_SESSION_SQL,
                transaction.api_key,
                transaction.session_id,
                transaction.router_name,
                transaction.router_type,
                transaction.model,
                transaction.turn_at.isoformat(),
                transaction.total_tokens,
                transaction.spend,
                transaction.saved_spend,
                int(transaction.covered),
                int(transaction.cache_hit),
                transaction.cache_ttl_seconds,
            )
            return
        except DB_RETRY_SAFE_ERROR_TYPES:
            if attempt >= n_retry_times:
                raise
            await asyncio.sleep(2**attempt + random.uniform(0, 1))


async def flush_autorouter_turn_transactions(
    prisma_client: PrismaClient,
    transactions: Sequence[AutoRouterTurnTransaction],
    n_retry_times: int = 3,
) -> None:
    """Drain a queue batch into the rollup, one upsert per turn.

    Statements run sequentially in per-session event order: a turn's classification
    depends on the turns before it, and Postgres rejects one multi-row INSERT touching
    the same key twice. Only ConnectError is retried, per statement, because it proves
    that statement never reached the database; any other failure drops the rest of the
    batch with an error log, since a repeated increment is worse than an undercount.
    Callers must not add their own retry around this function.
    """
    if not transactions:
        return
    ordered: Final = sorted(
        transactions,
        key=lambda transaction: (
            transaction.api_key,
            transaction.session_id,
            transaction.router_name,
            transaction.turn_at,
        ),
    )
    for index, transaction in enumerate(ordered):
        try:
            await _upsert_turn_with_retry(prisma_client, transaction, n_retry_times)
        except Exception as flush_err:  # noqa: BLE001  # any statement failure drops the batch remainder by design
            verbose_proxy_logger.error(
                "Spend tracking - auto-router session rollup flush failed; %s of %s turn transactions dropped: %s",
                len(ordered) - index,
                len(ordered),
                flush_err,
            )
            return
