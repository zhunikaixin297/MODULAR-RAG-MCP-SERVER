from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

from src.core.types import RetrievalResult
from src.mcp_server.tools.query_knowledge_hub import (
    QueryKnowledgeHubConfig,
    QueryKnowledgeHubTool,
)


class _HybridAlwaysRaises:
    def search(self, **kwargs: Any) -> Any:
        raise RuntimeError("temporary failure")


class _HybridFlaky:
    def __init__(self, fail_times: int, payload: Any):
        self._fail_times = fail_times
        self._calls = 0
        self._payload = payload

    @property
    def calls(self) -> int:
        return self._calls

    def search(self, **kwargs: Any) -> Any:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError("transient error")
        return self._payload


def _minimal_settings() -> Any:
    return SimpleNamespace(
        vector_store=SimpleNamespace(provider="opensearch", collection_name="base"),
        retrieval=SimpleNamespace(sparse_enabled=False),
    )


def test_perform_search_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = QueryKnowledgeHubTool(
        settings=_minimal_settings(),
        config=QueryKnowledgeHubConfig(
            search_max_attempts=3,
            search_retry_backoff_seconds=0.0,
            enable_rerank=False,
        ),
    )
    result = RetrievalResult(chunk_id="c1", score=0.9, text="hello", metadata={})
    hybrid = _HybridFlaky(fail_times=2, payload=[result])
    sleep_calls: list[float] = []
    monkeypatch.setattr("src.mcp_server.tools.query_knowledge_hub.time.sleep", sleep_calls.append)

    out = tool._perform_search(hybrid, "q", top_k=5, collection="base")

    assert len(out) == 1
    assert out[0].chunk_id == "c1"
    assert hybrid.calls == 3
    assert len(sleep_calls) == 2


def test_perform_search_supports_object_result_container(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = QueryKnowledgeHubTool(
        settings=_minimal_settings(),
        config=QueryKnowledgeHubConfig(search_max_attempts=1, enable_rerank=False),
    )
    result = RetrievalResult(chunk_id="c2", score=0.8, text="world", metadata={})
    container = SimpleNamespace(results=[result])
    hybrid = _HybridFlaky(fail_times=0, payload=container)
    monkeypatch.setattr("src.mcp_server.tools.query_knowledge_hub.time.sleep", lambda _: None)

    out = tool._perform_search(hybrid, "q", top_k=5, collection="base")

    assert len(out) == 1
    assert out[0].chunk_id == "c2"


def test_perform_search_exhausted_retries_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = QueryKnowledgeHubTool(
        settings=_minimal_settings(),
        config=QueryKnowledgeHubConfig(
            search_max_attempts=2,
            search_retry_backoff_seconds=0.0,
            enable_rerank=False,
        ),
    )
    hybrid = _HybridAlwaysRaises()
    monkeypatch.setattr("src.mcp_server.tools.query_knowledge_hub.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        tool._perform_search(hybrid, "q", top_k=5, collection="base")


def test_ensure_initialized_lru_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    created_hybrids: list[Any] = []

    monkeypatch.setattr(
        "src.libs.embedding.embedding_factory.EmbeddingFactory.create",
        lambda settings: object(),
    )
    monkeypatch.setattr(
        "src.core.query_engine.reranker.create_core_reranker",
        lambda settings: SimpleNamespace(is_enabled=False),
    )
    monkeypatch.setattr(
        "src.libs.vector_store.vector_store_factory.VectorStoreFactory.create",
        lambda settings: object(),
    )
    monkeypatch.setattr(
        "src.core.query_engine.dense_retriever.create_dense_retriever",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "src.core.query_engine.sparse_retriever.create_sparse_retriever",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "src.core.query_engine.hybrid_search.create_hybrid_search",
        lambda **kwargs: created_hybrids.append(object()) or created_hybrids[-1],
    )

    tool = QueryKnowledgeHubTool(
        settings=_minimal_settings(),
        config=QueryKnowledgeHubConfig(max_cached_collections=2, enable_rerank=False),
    )

    h1 = tool._ensure_initialized("c1")
    h2 = tool._ensure_initialized("c2")
    h3 = tool._ensure_initialized("c3")

    assert h1 is not h2 and h2 is not h3
    assert list(tool._hybrid_search_by_collection.keys()) == ["c2", "c3"]
    assert len(created_hybrids) == 3


def test_ensure_initialized_concurrent_access_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = {"n": 0}

    monkeypatch.setattr(
        "src.libs.embedding.embedding_factory.EmbeddingFactory.create",
        lambda settings: object(),
    )
    monkeypatch.setattr(
        "src.core.query_engine.reranker.create_core_reranker",
        lambda settings: SimpleNamespace(is_enabled=False),
    )
    monkeypatch.setattr(
        "src.libs.vector_store.vector_store_factory.VectorStoreFactory.create",
        lambda settings: object(),
    )
    monkeypatch.setattr(
        "src.core.query_engine.dense_retriever.create_dense_retriever",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "src.core.query_engine.sparse_retriever.create_sparse_retriever",
        lambda **kwargs: object(),
    )

    def _mk_hybrid(**kwargs: Any) -> Any:
        counter["n"] += 1
        return SimpleNamespace(name=f"h{counter['n']}")

    monkeypatch.setattr(
        "src.core.query_engine.hybrid_search.create_hybrid_search",
        _mk_hybrid,
    )

    tool = QueryKnowledgeHubTool(
        settings=_minimal_settings(),
        config=QueryKnowledgeHubConfig(max_cached_collections=3, enable_rerank=False),
    )
    collections = ["a", "b", "c", "d"] * 5

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(tool._ensure_initialized, collections))

    assert all(r is not None for r in results)
    assert len(tool._hybrid_search_by_collection) <= 3

