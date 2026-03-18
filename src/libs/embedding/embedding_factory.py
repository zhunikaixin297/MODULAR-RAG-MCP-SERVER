"""Factory for creating Embedding provider instances.

This module implements the Factory Pattern to instantiate the appropriate
Embedding provider based on configuration, enabling configuration-driven selection
of different backends without code changes.
"""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING, Any

from src.libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from src.core.settings import Settings


class EmbeddingFactory:
    """Factory for creating Embedding provider instances.
    
    This factory reads the provider configuration from settings and instantiates
    the corresponding Embedding implementation. Supported providers: OpenAI, Azure,
    and more to be added in subsequent tasks.
    
    Design Principles Applied:
    - Factory Pattern: Centralizes object creation logic.
    - Config-Driven: Provider selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown providers.
    """
    
    # Registry of supported providers
    _PROVIDERS: dict[str, type[BaseEmbedding]] = {}
    _INSTANCES: dict[tuple[Any, ...], BaseEmbedding] = {}
    _LOCK = RLock()
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseEmbedding]) -> None:
        """Register a new Embedding provider implementation.
        
        This method allows provider implementations to register themselves
        with the factory, supporting extensibility.
        
        Args:
            name: The provider identifier (e.g., 'openai', 'azure', 'local').
            provider_class: The BaseEmbedding subclass implementing the provider.
        
        Raises:
            ValueError: If provider_class doesn't inherit from BaseEmbedding.
        """
        if not issubclass(provider_class, BaseEmbedding):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseEmbedding"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEmbedding:
        """Create an Embedding instance based on configuration.
        
        Args:
            settings: The application settings containing Embedding configuration.
            **override_kwargs: Optional parameters to override config values.
        
        Returns:
            An instance of the configured Embedding provider.
        
        Raises:
            ValueError: If the configured provider is not supported.
            AttributeError: If required configuration fields are missing.
        
        Example:
            >>> settings = Settings.load('config/settings.yaml')
            >>> embedding = EmbeddingFactory.create(settings)
            >>> vectors = embedding.embed(["hello world", "test"])
        """
        # Extract provider name from settings
        try:
            provider_name = settings.embedding.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.embedding.provider. "
                "Please ensure 'embedding.provider' is specified in settings.yaml"
            ) from e
        
        # Look up provider class in registry
        provider_class = cls._PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Embedding provider: '{provider_name}'. "
                f"Available providers: {available}"
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
                f"Failed to instantiate Embedding provider '{provider_name}': {e}"
            ) from e
        with cls._LOCK:
            cls._INSTANCES[cache_key] = instance
        return instance
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.
        
        Returns:
            Sorted list of available provider identifiers.
        """
        return sorted(cls._PROVIDERS.keys())

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
        embedding = getattr(settings, "embedding", None)
        if embedding is None:
            return ()
        return (
            getattr(embedding, "provider", None),
            getattr(embedding, "model", None),
            getattr(embedding, "base_url", None),
            getattr(embedding, "azure_endpoint", None),
            getattr(embedding, "deployment_name", None),
            getattr(embedding, "api_version", None),
            getattr(embedding, "dimensions", None),
        )


# Auto-register providers on module import
def _register_builtin_providers() -> None:
    """Register built-in Embedding providers with the factory."""
    try:
        from src.libs.embedding.openai_embedding import OpenAIEmbedding
        EmbeddingFactory.register_provider("openai", OpenAIEmbedding)
    except ImportError:
        pass  # OpenAI provider not available
    
    try:
        from src.libs.embedding.azure_embedding import AzureEmbedding
        EmbeddingFactory.register_provider("azure", AzureEmbedding)
    except ImportError:
        pass  # Azure provider not available
    
    try:
        from src.libs.embedding.ollama_embedding import OllamaEmbedding
        EmbeddingFactory.register_provider("ollama", OllamaEmbedding)
    except ImportError:
        pass  # Ollama provider not available


# Register providers when module is imported
_register_builtin_providers()
