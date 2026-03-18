from types import SimpleNamespace
from unittest.mock import MagicMock

from src.libs.reranker.tei_reranker import TEIReranker


class _Settings:
    def __init__(self):
        self.rerank = SimpleNamespace(
            base_url="http://localhost:8088",
            model="test-reranker",
            timeout_seconds=3,
            max_concurrency=2,
            api_key=None,
        )


def _build_candidates() -> list[dict]:
    return [
        {"id": "a", "text": "alpha"},
        {"id": "b", "text": "beta"},
        {"id": "c", "text": "gamma"},
    ]


def test_tei_rerank_parses_scores_as_object_list():
    reranker = TEIReranker(settings=_Settings())
    response = MagicMock()
    response.json.return_value = {
        "scores": [
            {"index": 0, "score": 0.2},
            {"index": 1, "score": 0.9},
            {"index": 2, "score": 0.5},
        ]
    }
    response.raise_for_status.return_value = None
    reranker._session.post = MagicMock(return_value=response)

    results = reranker.rerank("q", _build_candidates())

    assert [item["id"] for item in results] == ["b", "c", "a"]
    assert results[0]["rerank_score"] == 0.9


def test_tei_rerank_parses_results_indexed_format():
    reranker = TEIReranker(settings=_Settings())
    response = MagicMock()
    response.json.return_value = {
        "results": [
            {"index": 2, "score": 0.95},
            {"index": 0, "score": 0.7},
            {"index": 1, "score": 0.4},
        ]
    }
    response.raise_for_status.return_value = None
    reranker._session.post = MagicMock(return_value=response)

    results = reranker.rerank("q", _build_candidates())

    assert [item["id"] for item in results] == ["c", "a", "b"]
    assert results[0]["rerank_score"] == 0.95
