from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class TEIRerankError(RuntimeError):
    pass


class TEIReranker(BaseReranker):
    def __init__(
        self,
        settings: Any,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_concurrency: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        rerank_settings = getattr(settings, "rerank", None)
        self.base_url = base_url or getattr(rerank_settings, "base_url", None)
        self.api_key = api_key or getattr(rerank_settings, "api_key", None)
        self.timeout = float(timeout or getattr(rerank_settings, "timeout", 30.0))
        self.max_concurrency = int(max_concurrency or getattr(rerank_settings, "max_concurrency", 50))
        if not self.base_url:
            raise TEIRerankError("Missing rerank.base_url for TEI reranker")
        self.url = f"{self.base_url.rstrip('/')}/rerank"
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        self._session = requests.Session()

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)
        top_k = kwargs.get("top_k", len(candidates))
        texts = []
        for candidate in candidates:
            text = candidate.get("text") or candidate.get("content", "")
            texts.append(str(text))
        payload = {"query": query, "texts": texts, "truncate": True}
        try:
            response = self._session.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            scores = self._extract_scores(response.json(), len(candidates))
        except Exception as e:
            raise TEIRerankError(f"TEI reranker request failed: {e}") from e
        reranked = []
        for idx, candidate in enumerate(candidates):
            score = scores[idx] if idx < len(scores) else 0.0
            updated = dict(candidate)
            updated["rerank_score"] = score
            reranked.append(updated)
        reranked.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
        if isinstance(top_k, int) and top_k > 0:
            return reranked[:top_k]
        return reranked

    def _extract_scores(self, data: Any, total: int) -> List[float]:
        if isinstance(data, dict):
            if "scores" in data and isinstance(data["scores"], list):
                return [self._to_score(score) for score in data["scores"]]
            if "results" in data and isinstance(data["results"], list):
                scores = [0.0 for _ in range(total)]
                for item in data["results"]:
                    if not isinstance(item, dict):
                        continue
                    index = item.get("index")
                    score = item.get("score")
                    if isinstance(index, int) and index < total:
                        scores[index] = self._to_score(score)
                return scores
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                scores = [0.0 for _ in range(total)]
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    index = item.get("index")
                    if isinstance(index, int) and index < total:
                        scores[index] = self._to_score(item.get("score"))
                return scores
            return [self._to_score(score) for score in data]
        return [0.0 for _ in range(total)]

    def _to_score(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, dict):
            if "score" in value:
                return self._to_score(value.get("score"))
            if "value" in value:
                return self._to_score(value.get("value"))
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def close(self) -> None:
        self._session.close()
