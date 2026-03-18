from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING, Any

from src.libs.loader.base_loader import BaseLoader

if TYPE_CHECKING:
    from src.core.settings import Settings


class LoaderFactory:
    _PROVIDERS: dict[str, type[BaseLoader]] = {}
    _INSTANCES: dict[tuple[Any, ...], BaseLoader] = {}
    _LOCK = RLock()

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLoader]) -> None:
        if not issubclass(provider_class, BaseLoader):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseLoader"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseLoader:
        try:
            loader_settings = settings.loader
            provider_name = loader_settings.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.loader.provider. "
                "Please ensure 'loader.provider' is specified in settings.yaml"
            ) from e

        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Loader provider: '{provider_name}'. "
                f"Available providers: {available}."
            )

        if provider_name == "docling":
            cache_key = (
                provider_name,
                provider_class,
                cls._freeze_kwargs(override_kwargs),
                cls._docling_signature(settings),
            )
            with cls._LOCK:
                instance = cls._INSTANCES.get(cache_key)
                if instance is not None:
                    return instance
                try:
                    created = provider_class(settings=settings, **override_kwargs)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to instantiate Loader provider '{provider_name}': {e}"
                    ) from e
                cls._INSTANCES[cache_key] = created
                return created

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Loader provider '{provider_name}': {e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
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
    def _docling_signature(cls, settings: Settings) -> tuple[Any, ...]:
        loader_settings = getattr(settings, "loader", None)
        docling = getattr(loader_settings, "docling", None) if loader_settings else None
        general = getattr(docling, "general", None) if docling else None
        if general is None:
            return ()
        return (
            getattr(general, "images_scale", None),
            getattr(general, "generate_picture_images", None),
            getattr(general, "do_ocr", None),
            getattr(general, "do_formula_recognition", None),
            getattr(general, "do_table_enrichment", None),
            getattr(general, "do_pic_enrichment", None),
            getattr(general, "accelerator_device", None),
            getattr(general, "accelerator_num_threads", None),
            getattr(general, "timeout_seconds", None),
            getattr(general, "image_output_dir", None),
        )


def _register_builtin_providers() -> None:
    try:
        from src.libs.loader.pdf_loader import PdfLoader
        LoaderFactory.register_provider("pdf", PdfLoader)
    except Exception:
        pass
    try:
        from src.libs.loader.docling_loader import DoclingLoader
        LoaderFactory.register_provider("docling", DoclingLoader)
    except Exception:
        pass


_register_builtin_providers()
