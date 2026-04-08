from __future__ import annotations

import asyncio

import pytest

from src.libs.vector_store.opensearch_store import OpenSearchStore


def _make_store() -> OpenSearchStore:
    store = object.__new__(OpenSearchStore)
    store.max_attempts = 3
    store.retry_backoff_seconds = 0.1
    store.retry_backoff_max_seconds = 0.5
    store._client = None
    store._bg_loop = None
    store._loop_thread = None
    return store


@pytest.mark.asyncio
async def test_async_with_retry_eventually_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store()
    attempts = {"n": 0}
    sleep_calls: list[float] = []

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("temporary")
        return "ok"

    monkeypatch.setattr(store, "_backoff_delay", lambda attempt: 0.01 * (attempt + 1))

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("src.libs.vector_store.opensearch_store.asyncio.sleep", _fake_sleep)

    result = await store._async_with_retry("search", flaky)

    assert result == "ok"
    assert attempts["n"] == 3
    assert sleep_calls == [0.01, 0.02]


@pytest.mark.asyncio
async def test_async_with_retry_exhausted_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store()

    async def always_fail() -> str:
        raise ValueError("boom")

    monkeypatch.setattr(store, "_backoff_delay", lambda attempt: 0.0)

    async def _fake_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr("src.libs.vector_store.opensearch_store.asyncio.sleep", _fake_sleep)

    with pytest.raises(RuntimeError, match="OpenSearch count failed after 3 attempts"):
        await store._async_with_retry("count", always_fail)


def test_backoff_delay_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store()
    monkeypatch.setattr("src.libs.vector_store.opensearch_store.random.uniform", lambda a, b: b)

    delay = store._backoff_delay(attempt=10)

    # bounded base=0.5, jitter max=0.1 => total <= 0.6
    assert 0.5 <= delay <= 0.6
