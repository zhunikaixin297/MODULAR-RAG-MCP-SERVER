"""Factory for creating VectorStore provider instances.

This module implements the Factory Pattern to instantiate the appropriate
VectorStore provider based on configuration, enabling configuration-driven selection
of different backends without code changes.
"""

from __future__ import annotations

import asyncio
import atexit
import inspect
from threading import RLock
from typing import TYPE_CHECKING, Any

from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings


class VectorStoreFactory:
    """Factory for creating VectorStore provider instances.
    
    This factory reads the provider configuration from settings and instantiates
    the corresponding VectorStore implementation. Supported providers will be added
    in subsequent tasks (B7.6 and beyond).
    
    Design Principles Applied:
    - Factory Pattern: Centralizes object creation logic.
    - Config-Driven: Provider selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown providers.
    """
    
    # Registry of supported providers (to be populated in B7.x tasks)
    _PROVIDERS: dict[str, type[BaseVectorStore]] = {}
    _INSTANCES: dict[tuple[Any, ...], BaseVectorStore] = {}
    _LOCK = RLock()
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseVectorStore]) -> None:
        """Register a new VectorStore provider implementation.
        
        This method allows provider implementations to register themselves
        with the factory, supporting extensibility.
        
        Args:
            name: The provider identifier (e.g., 'chroma', 'qdrant', 'milvus').
            provider_class: The BaseVectorStore subclass implementing the provider.
        
        Raises:
            ValueError: If provider_class doesn't inherit from BaseVectorStore.
        """
        if not issubclass(provider_class, BaseVectorStore):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseVectorStore"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseVectorStore:
        """Create a VectorStore instance based on configuration.
        
        Args:
            settings: The application settings containing VectorStore configuration.
            **override_kwargs: Optional parameters to override config values.
        
        Returns:
            An instance of the configured VectorStore provider.
        
        Raises:
            ValueError: If the configured provider is not supported.
            AttributeError: If required configuration fields are missing.
        
        Example:
            >>> settings = Settings.load('config/settings.yaml')
            >>> vector_store = VectorStoreFactory.create(settings)
            >>> vector_store.upsert([{'id': 'doc1', 'vector': [0.1, 0.2]}])
        """
        # Extract provider name from settings
        try:
            provider_name = settings.vector_store.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vector_store.provider. "
                "Please ensure 'vector_store.provider' is specified in settings.yaml"
            ) from e
        
        # Look up provider class in registry
        provider_class = cls._PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported VectorStore provider: '{provider_name}'. "
                f"Available providers: {available}. "
                f"Provider implementations will be added in task B7.6 and beyond."
            )
        
        cache_key = (
            provider_name,
            provider_class,
            cls._freeze_kwargs(override_kwargs),
            cls._settings_signature(settings),
        )
        with cls._LOCK:
            cached = cls._INSTANCES.get(cache_key)
            if cached is not None:
                return cached

        try:
            instance = provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate VectorStore provider '{provider_name}': {e}"
            ) from e
        with cls._LOCK:
            cls._INSTANCES[cache_key] = instance
        return instance
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.
        
        Returns:
            Sorted list of provider names.
        
        Example:
            >>> VectorStoreFactory.list_providers()
            ['chroma', 'milvus', 'qdrant']
        """
        return sorted(cls._PROVIDERS.keys())

    @classmethod
    def close_all(cls) -> None:
        with cls._LOCK:
            instances = list(cls._INSTANCES.values())
            cls._INSTANCES.clear()
        for instance in instances:
            close_fn = getattr(instance, "close", None)
            if not callable(close_fn):
                continue
            result = close_fn()
            if inspect.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(result)
                except RuntimeError:
                    asyncio.run(result)

    @classmethod
    def _freeze_kwargs(cls, kwargs: dict[str, Any]) -> tuple[Any, ...]:
        if not kwargs:
            return ()
        return tuple((key, cls._freeze_value(value)) for key, value in sorted(kwargs.items()))

    @classmethod
    def _freeze_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return tuple((k, cls._freeze_value(v)) for k, v in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze_value(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(cls._freeze_value(v) for v in value))
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return repr(value)

    @classmethod
    def _settings_signature(cls, settings: Settings) -> tuple[Any, ...]:
        vector_store = getattr(settings, "vector_store", None)
        if vector_store is None:
            return ()
        provider = getattr(vector_store, "provider", None)
        base = (
            provider,
            getattr(vector_store, "collection_name", None),
            getattr(vector_store, "persist_directory", None),
        )
        if str(provider).lower() != "opensearch":
            return base
        opensearch = getattr(vector_store, "opensearch", None)
        if opensearch is None:
            return base
        return base + (
            cls._freeze_value(getattr(opensearch, "hosts", None)),
            getattr(opensearch, "host", None),
            getattr(opensearch, "port", None),
            getattr(opensearch, "scheme", None),
            getattr(opensearch, "index_name", None),
            getattr(opensearch, "username", None),
            getattr(opensearch, "use_ssl", None),
            getattr(opensearch, "verify_certs", None),
        )


atexit.register(VectorStoreFactory.close_all)
