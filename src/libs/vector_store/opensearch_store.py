from __future__ import annotations

import asyncio
import logging
from threading import Thread
from typing import Any, Dict, List, Optional
import jieba

try:
    from opensearchpy import AsyncOpenSearch
    from opensearchpy.helpers import async_bulk
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False

from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class OpenSearchStore(BaseVectorStore):
    def __init__(self, settings: Any, **kwargs: Any) -> None:
        if not OPENSEARCH_AVAILABLE:
            raise ImportError(
                "opensearch-py is required for OpenSearchStore. "
                "Install it with: pip install opensearch-py"
            )

        try:
            vector_store_settings = settings.vector_store
            opensearch_config = vector_store_settings.opensearch
            if opensearch_config is None:
                raise AttributeError("settings.vector_store.opensearch is None")
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vector_store.opensearch. "
                "Please ensure 'vector_store.opensearch' is specified in settings.yaml"
            ) from e

        base_index_name = getattr(opensearch_config, "index_name", "default")
        collection_name = kwargs.get("collection_name") or getattr(
            getattr(settings, "vector_store", None), "collection_name", None
        )
        self.index_name = collection_name or base_index_name
        self.timeout_seconds = getattr(opensearch_config, "timeout_seconds", 60)
        self.max_retries = getattr(opensearch_config, "max_retries", 3)
        self.max_concurrency = getattr(opensearch_config, "max_concurrency", 10)
        self.batch_size = getattr(opensearch_config, "batch_size", 200)
        self.refresh = getattr(opensearch_config, "refresh", False)

        hosts = getattr(opensearch_config, "hosts", None)
        if not hosts:
            host = getattr(opensearch_config, "host", "localhost")
            port = getattr(opensearch_config, "port", 9200)
            scheme = getattr(opensearch_config, "scheme", "http")
            hosts = [f"{scheme}://{host}:{port}"]

        self.username = getattr(opensearch_config, "username", None)
        self.password = getattr(opensearch_config, "password", None)
        self.use_ssl = getattr(opensearch_config, "use_ssl", False)
        self.verify_certs = getattr(opensearch_config, "verify_certs", False)
        self.hosts = hosts
        self._client: Optional[AsyncOpenSearch] = None
        self._bg_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread = Thread(target=self._run_loop, name="OpenSearchStoreLoop", daemon=True)
        self._loop_thread.start()

        self.dimension = getattr(settings.embedding, "dimensions", None)
        if not self.dimension:
            raise ValueError("Missing embedding dimensions in settings.embedding.dimensions")

        self._run_async(self._ensure_index())

    @property
    def client(self) -> AsyncOpenSearch:
        """Lazy initialization of the OpenSearch client on the dedicated background loop."""
        if self._client is None:
            self._client = AsyncOpenSearch(
                hosts=self.hosts,
                http_auth=(self.username, self.password) if self.username or self.password else None,
                use_ssl=self.use_ssl,
                verify_certs=self.verify_certs,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
                retry_on_timeout=True,
            )
        return self._client

    async def close(self) -> None:
        """Close resources for this store instance."""
        self._shutdown()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._bg_loop)
        self._bg_loop.run_forever()

    def _shutdown(self) -> None:
        if self._client is not None:
            close_future = asyncio.run_coroutine_threadsafe(self._client.close(), self._bg_loop)
            try:
                close_future.result(timeout=10)
            except Exception:
                pass
            self._client = None
        if self._bg_loop.is_running():
            self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        if self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)
        if not self._bg_loop.is_closed():
            self._bg_loop.close()

    def __del__(self) -> None:
        self._shutdown()

    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.validate_records(records)
        self._run_async(self._async_upsert(records))

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.validate_query_vector(vector, top_k)
        requested_field = str(kwargs.get("vector_field", "embedding_content"))
        field_mapping = {
            "content": "embedding_content",
            "summary": "embedding_summary",
            "hypothetical_questions": "embedding_hypothetical_questions",
        }
        vector_field = field_mapping.get(requested_field, requested_field)
        return self._run_async(self._async_query(vector, top_k, filters, vector_field))

    def keyword_search(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if not query_text or not query_text.strip():
            return []
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        tokenized = " ".join(jieba.cut_for_search(query_text))
        return self._run_async(self._async_keyword_search(tokenized, top_k, filters))

    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if not ids:
            raise ValueError("IDs list cannot be empty")
        self._run_async(self._async_delete(ids))

    def clear(
        self,
        collection_name: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        target_index = collection_name or self.index_name
        self._run_async(self._async_clear(target_index))

    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if not ids:
            raise ValueError("IDs list cannot be empty")
        return self._run_async(self._async_get_by_ids(ids))

    def delete_by_metadata(
        self,
        filters: Dict[str, Any],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> int:
        if not filters:
            raise ValueError("filters cannot be empty")
        return self._run_async(self._async_delete_by_metadata(filters))

    def get_ids_by_metadata(
        self,
        filters: Dict[str, Any],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        if not filters:
            raise ValueError("filters cannot be empty")
        return self._run_async(self._async_get_ids_by_metadata(filters))

    def count_by_metadata(
        self,
        filters: Dict[str, Any],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> int:
        if not filters:
            raise ValueError("filters cannot be empty")
        return self._run_async(self._async_count_by_metadata(filters))

    def get_by_metadata(
        self,
        filters: Dict[str, Any],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if not filters:
            raise ValueError("filters cannot be empty")
        return self._run_async(self._async_get_by_metadata(filters))

    async def _ensure_index(self) -> None:
        try:
            exists = await self.client.indices.exists(index=self.index_name)
        except Exception as e:
            logger.error(f"OpenSearch index existence check failed: {e}", exc_info=True)
            raise

        if exists:
            return

        body = {
            "settings": {
                "index": {
                    "knn": True,
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "summary": {"type": "text"},
                    "hypothetical_questions_merged": {"type": "text"},
                    "embedding_content": {
                        "type": "knn_vector",
                        "dimension": self.dimension,
                    },
                    "embedding_summary": {
                        "type": "knn_vector",
                        "dimension": self.dimension,
                    },
                    "embedding_hypothetical_questions": {
                        "type": "knn_vector",
                        "dimension": self.dimension,
                    },
                    "metadata": {"type": "object", "enabled": True},
                    "doc_hash": {"type": "keyword"},
                }
            },
        }

        try:
            await self.client.indices.create(index=self.index_name, body=body)
        except Exception as e:
            logger.error(f"OpenSearch index creation failed: {e}", exc_info=True)
            raise

    async def _async_upsert(self, records: List[Dict[str, Any]]) -> None:
        actions = []
        for record in records:
            metadata = record.get("metadata", {})
            vectors = record.get("vectors", {})
            summary_text = metadata.get("summary", "")
            questions = metadata.get("hypothetical_questions", [])
            questions_text = " ".join(questions) if isinstance(questions, list) else str(questions)
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_id": str(record["id"]),
                    "_source": {
                        "chunk_id": str(record["id"]),
                        "document_id": metadata.get("doc_hash") or metadata.get("source_ref"),
                        "content": metadata.get("text", ""),
                        "summary": summary_text,
                        "hypothetical_questions_merged": questions_text,
                        "embedding_content": record["vector"],
                        "embedding_summary": vectors.get("summary"),
                        "embedding_hypothetical_questions": vectors.get("hypothetical_questions"),
                        "metadata": metadata,
                        "doc_hash": metadata.get("doc_hash"),
                    },
                }
            )

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _bulk_batch(batch: List[Dict[str, Any]]) -> None:
            async with semaphore:
                try:
                    await async_bulk(
                        self.client,
                        batch,
                        chunk_size=self.batch_size,
                        raise_on_error=False,
                        request_timeout=self.timeout_seconds,
                        refresh="wait_for" if self.refresh else False,
                    )
                except Exception as e:
                    logger.error(f"OpenSearch bulk upsert failed: {e}", exc_info=True)
                    raise

        tasks = []
        for i in range(0, len(actions), self.batch_size):
            tasks.append(_bulk_batch(actions[i : i + self.batch_size]))

        await asyncio.gather(*tasks)

    async def _async_query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
        vector_field: str,
    ) -> List[Dict[str, Any]]:
        filter_clause = None
        if filters:
            terms = []
            for key, value in filters.items():
                if key == "collection":
                    continue
                field = f"metadata.{key}"
                terms.append({"term": {field: value}})
            if terms:
                filter_clause = {"bool": {"must": terms}}

        knn_body: Dict[str, Any] = {
            "vector": vector,
            "k": top_k,
        }
        if filter_clause is not None:
            knn_body["filter"] = filter_clause

        query = {
            "size": top_k,
            "query": {
                "knn": {
                    vector_field: knn_body,
                }
            },
        }

        try:
            response = await self.client.search(index=self.index_name, body=query)
        except Exception as e:
            logger.error(f"OpenSearch query failed: {e}", exc_info=True)
            raise RuntimeError(f"OpenSearch query failed: {e}") from e

        hits = response.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            source = hit.get("_source", {})
            results.append(
                {
                    "id": hit.get("_id"),
                    "score": hit.get("_score", 0.0),
                    "text": source.get("content", ""),
                    "metadata": source.get("metadata", {}),
                }
            )
        return results

    async def _async_keyword_search(
        self,
        tokenized_query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        must_clauses: List[Dict[str, Any]] = [
            {
                "multi_match": {
                    "query": tokenized_query,
                    "type": "best_fields",
                    "fields": [
                        "content^2.5",
                        "hypothetical_questions_merged^2",
                        "summary^1.5",
                    ],
                }
            }
        ]
        if filters:
            for key, value in filters.items():
                if key == "collection":
                    continue
                must_clauses.append({"term": {f"metadata.{key}": value}})
        query = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": must_clauses,
                }
            },
        }
        try:
            response = await self.client.search(index=self.index_name, body=query)
        except Exception as e:
            logger.error(f"OpenSearch keyword search failed: {e}", exc_info=True)
            raise RuntimeError(f"OpenSearch keyword search failed: {e}") from e
        hits = response.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            source = hit.get("_source", {})
            results.append(
                {
                    "id": hit.get("_id"),
                    "chunk_id": hit.get("_id"),
                    "score": hit.get("_score", 0.0),
                    "text": source.get("content", ""),
                    "metadata": source.get("metadata", {}),
                }
            )
        return results

    async def _async_delete(self, ids: List[str]) -> None:
        actions = [
            {"_op_type": "delete", "_index": self.index_name, "_id": str(id_)}
            for id_ in ids
        ]
        try:
            await async_bulk(
                self.client,
                actions,
                chunk_size=self.batch_size,
                raise_on_error=False,
                request_timeout=self.timeout_seconds,
                refresh="wait_for" if self.refresh else False,
            )
        except Exception as e:
            logger.error(f"OpenSearch delete failed: {e}", exc_info=True)
            raise RuntimeError(f"OpenSearch delete failed: {e}") from e

    async def _async_clear(self, index_name: str) -> None:
        try:
            exists = await self.client.indices.exists(index=index_name)
            if exists:
                await self.client.indices.delete(index=index_name)
            await self.client.indices.create(index=index_name, body={
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "document_id": {"type": "keyword"},
                        "content": {"type": "text"},
                        "summary": {"type": "text"},
                        "hypothetical_questions_merged": {"type": "text"},
                        "embedding_content": {"type": "knn_vector", "dimension": self.dimension},
                        "embedding_summary": {"type": "knn_vector", "dimension": self.dimension},
                        "embedding_hypothetical_questions": {"type": "knn_vector", "dimension": self.dimension},
                        "metadata": {"type": "object", "enabled": True},
                        "doc_hash": {"type": "keyword"},
                    }
                },
            })
        except Exception as e:
            logger.error(f"OpenSearch clear failed: {e}", exc_info=True)
            raise RuntimeError(f"OpenSearch clear failed: {e}") from e

    async def _async_get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        try:
            response = await self.client.mget(
                index=self.index_name,
                body={"ids": [str(id_) for id_ in ids]},
            )
        except Exception as e:
            logger.error(f"OpenSearch mget failed: {e}", exc_info=True)
            raise RuntimeError(f"OpenSearch get_by_ids failed: {e}") from e

        docs = response.get("docs", [])
        lookup = {
            doc.get("_id"): doc.get("_source", {})
            for doc in docs
            if doc.get("found")
        }
        output = []
        for id_ in ids:
            source = lookup.get(str(id_))
            if not source:
                output.append({})
            else:
                output.append(
                    {
                        "id": str(id_),
                        "text": source.get("content", ""),
                        "metadata": source.get("metadata", {}),
                    }
                )
        return output

    async def _async_delete_by_metadata(self, filter_dict: Dict[str, Any]) -> int:
        must_terms = []
        for key, value in filter_dict.items():
            field = f"metadata.{key}"
            must_terms.append({"term": {field: value}})
        query = {"query": {"bool": {"must": must_terms}}}
        try:
            response = await self.client.delete_by_query(
                index=self.index_name,
                body=query,
                refresh="wait_for" if self.refresh else False,
            )
            return int(response.get("deleted", 0))
        except Exception as e:
            logger.error(f"OpenSearch delete_by_metadata failed: {e}", exc_info=True)
            raise RuntimeError(f"OpenSearch delete_by_metadata failed: {e}") from e

    async def _async_get_ids_by_metadata(self, filter_dict: Dict[str, Any]) -> List[str]:
        must_terms = []
        for key, value in filter_dict.items():
            field = f"metadata.{key}"
            must_terms.append({"term": {field: value}})
        query = {
            "query": {"bool": {"must": must_terms}},
            "_source": False,
            "size": 10000,  # Max result size for this operation
        }
        try:
            response = await self.client.search(index=self.index_name, body=query)
            hits = response.get("hits", {}).get("hits", [])
            return [hit["_id"] for hit in hits]
        except Exception as e:
            logger.error(f"OpenSearch get_ids_by_metadata failed: {e}", exc_info=True)
            raise RuntimeError(f"OpenSearch get_ids_by_metadata failed: {e}") from e

    async def _async_count_by_metadata(self, filter_dict: Dict[str, Any]) -> int:
        must_terms = []
        for key, value in filter_dict.items():
            field = f"metadata.{key}"
            must_terms.append({"term": {field: value}})
        query = {"query": {"bool": {"must": must_terms}}}
        try:
            response = await self.client.count(index=self.index_name, body=query)
            return int(response.get("count", 0))
        except Exception as e:
            logger.error(f"OpenSearch count_by_metadata failed: {e}", exc_info=True)
            raise RuntimeError(f"OpenSearch count_by_metadata failed: {e}") from e

    async def _async_get_by_metadata(self, filter_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        must_terms = []
        for key, value in filter_dict.items():
            field = f"metadata.{key}"
            must_terms.append({"term": {field: value}})
        query = {
            "size": 10000,
            "query": {"bool": {"must": must_terms}},
            "sort": [{"metadata.chunk_index": {"order": "asc"}}],
        }
        try:
            response = await self.client.search(index=self.index_name, body=query)
            hits = response.get("hits", {}).get("hits", [])
            results = []
            for hit in hits:
                source = hit.get("_source", {})
                results.append({
                    "id": hit.get("_id"),
                    "text": source.get("content", ""),
                    "metadata": source.get("metadata", {}),
                })
            return results
        except Exception as e:
            logger.error(f"OpenSearch get_by_metadata failed: {e}", exc_info=True)
            raise RuntimeError(f"OpenSearch get_by_metadata failed: {e}") from e

    def _run_async(self, coro: Any) -> Any:
        if self._bg_loop.is_closed():
            raise RuntimeError("OpenSearch background event loop is closed")
        future = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
        return future.result()
