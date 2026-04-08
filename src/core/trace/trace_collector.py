"""Trace collector – receives finished TraceContext and persists them.

The collector is the bridge between in-memory TraceContext objects and
the on-disk JSON Lines log used by the Dashboard.  It is intentionally
decoupled from the logging module so that trace persistence remains
predictable and testable.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.core.settings import load_settings, resolve_path
from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# Default absolute path for traces file (CWD-independent)
_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceCollector:
    """Collects finished traces and appends them to a JSON Lines file.

    Args:
        traces_path: File path for the ``traces.jsonl`` output.
            Parent directories are created automatically.
    """

    def __init__(self, traces_path: str | Path = _DEFAULT_TRACES_PATH) -> None:
        self._enabled = True
        resolved_path: Path = Path(traces_path)

        # Honor settings.observability.trace_enabled / trace_file when available.
        try:
            settings = load_settings()
            observability = getattr(settings, "observability", None)
            if observability is not None:
                self._enabled = bool(getattr(observability, "trace_enabled", True))
                if traces_path == _DEFAULT_TRACES_PATH:
                    configured_path = getattr(observability, "trace_file", None)
                    if configured_path:
                        resolved_path = resolve_path(configured_path)
        except Exception:
            # Keep collector non-fatal: use defaults if settings cannot be loaded.
            self._enabled = True

        self._path = Path(resolved_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def collect(self, trace: TraceContext) -> None:
        """Persist a single trace as one JSON line.

        If the trace has not been finished yet, ``finish()`` is called
        automatically so the output always contains timing data.

        Args:
            trace: A populated :class:`TraceContext`.
        """
        if not self._enabled:
            return

        if trace.finished_at is None:
            trace.finish()

        line = json.dumps(trace.to_dict(), ensure_ascii=False)
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            logger.exception("Failed to write trace %s", trace.trace_id)

    @property
    def path(self) -> Path:
        """Return the resolved path of the traces file."""
        return self._path

    @property
    def enabled(self) -> bool:
        """Whether trace collection is enabled by runtime settings."""
        return self._enabled
