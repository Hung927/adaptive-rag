"""Langfuse tracing helpers for the RAG pipeline (Langfuse v4 / OTel API)."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from rag.core.config import LangfuseSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noop stubs (used when Langfuse is disabled)
# ---------------------------------------------------------------------------


class _NoopSpan:
    def end(self, **_: Any) -> None:
        pass

    def update(self, **_: Any) -> None:
        pass


class _NoopTrace:
    def update(self, **_: Any) -> None:
        pass

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# PipelineTracer
# ---------------------------------------------------------------------------


class PipelineTracer:
    """Thin wrapper around Langfuse v4 for pipeline node tracing.

    Langfuse v4 uses an OpenTelemetry-based context-manager API::

        with lf.start_as_current_observation(name="root") as root:
            root.update(input=..., output=...)
            with root.start_as_current_observation(name="child") as child:
                child.update(input=..., output=...)
    """

    def __init__(self, cfg: LangfuseSettings) -> None:
        self._enabled = cfg.enabled
        self._client: Any = None
        if self._enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    secret_key=cfg.secret_key,
                    public_key=cfg.public_key,
                    host=cfg.host,
                )
                logger.info("[tracing] Langfuse 已啟用，host=%s", cfg.host)
            except Exception as exc:
                logger.warning("[tracing] Langfuse 初始化失敗，停用追蹤: %s", exc)
                self._enabled = False

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        query: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """Open a top-level trace for one pipeline run."""
        if not self._enabled or self._client is None:
            yield _NoopTrace()
            return

        with self._client.start_as_current_observation(
            name=name,
            as_type="span",
            input={"query": query},
            metadata=metadata or {},
        ) as root_span:
            try:
                yield root_span
            finally:
                self._client.flush()

    @contextmanager
    def span(
        self,
        trace: Any,
        node_name: str,
        *,
        input: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """Open a child span for a single pipeline node.

        ``trace`` is the parent observation returned by :meth:`trace`.
        When Langfuse is disabled, ``trace`` is a ``_NoopTrace`` and
        this yields a ``_NoopSpan``.
        """
        if isinstance(trace, _NoopTrace):
            yield _NoopSpan()
            return

        with trace.start_as_current_observation(
            name=node_name,
            as_type="span",
            input=input or {},
        ) as span:
            try:
                yield span
            except Exception as exc:
                span.update(metadata={"error": str(exc)})
                raise
            # caller is expected to call span.update(output=...) before the
            # context exits; nothing extra needed here
