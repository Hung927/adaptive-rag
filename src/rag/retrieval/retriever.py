"""Retriever — thin wrapper for query convenience."""

from __future__ import annotations

from rag.core.types import QueryResult
from rag.retrieval.store import QdrantStore


def retrieve(
    store: QdrantStore,
    query: str,
    top_k: int = 5,
    source_file: str | None = None,
) -> list[QueryResult]:
    """Retrieve relevant chunks for a query."""
    return store.query(query_text=query, top_k=top_k, source_file=source_file)
