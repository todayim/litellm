"""Tests for litellm.llms.custom_httpx.async_client_cleanup."""

import pytest

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


@pytest.mark.asyncio
async def test_cleanup_does_not_resurrect_closed_owned_client(monkeypatch):
    """Regression: a second cleanup pass must not rebuild a fresh client on an
    already-closed owned handler. Reading ``handler.client`` on an owned handler
    whose ``_client`` has been closed goes through the healing property and
    constructs a new client; cleanup must therefore avoid that property, or a
    pytest session-finish + atexit sequence would resurrect and leak clients."""
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())

    handler = AsyncHTTPHandler()
    assert handler._owns_client is True
    original_client = handler._client

    litellm.in_memory_llm_clients_cache.cache_dict["test-handler"] = handler

    await close_litellm_async_clients()
    assert original_client.is_closed

    await close_litellm_async_clients()

    assert handler._client is original_client, (
        "second cleanup pass must not resurrect a fresh client on a closed owned handler"
    )
    assert handler._client.is_closed
