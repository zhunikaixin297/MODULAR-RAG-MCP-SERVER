"""
Loader Module.

This package contains document loader components:
- Base loader class
- PDF loader
- File integrity checker
"""

from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker

try:
    from src.libs.loader.docling_loader import DoclingLoader
    LoaderFactory.register_provider("docling", DoclingLoader)
except ImportError:
    DoclingLoader = None

try:
    LoaderFactory.register_provider("pdf", PdfLoader)
except Exception:
    pass

__all__ = [
    "BaseLoader",
    "PdfLoader",
    "DoclingLoader",
    "LoaderFactory",
    "FileIntegrityChecker",
    "SQLiteIntegrityChecker",
]
