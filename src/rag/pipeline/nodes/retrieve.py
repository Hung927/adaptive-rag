"""Retrieve node — vector search from Qdrant."""

from __future__ import annotations

import logging

from rag.core.types import QueryResult
from rag.pipeline.state import PipelineState
from rag.retrieval.store import QdrantStore

logger = logging.getLogger(__name__)


def retrieve_node(
    state: PipelineState,
    store: QdrantStore,
    top_k: int = 5,
) -> dict:
    """Retrieve relevant chunks using the rewritten query (or original query).

    On retry after retrieval failure:
    - Excludes confirmed_chunks from the new search
    - Merges confirmed_chunks back after retrieval so generate always has them
    """
    query = state.get("rewritten_query") or state["query"]
    confirmed_chunks: list[QueryResult] = state.get("confirmed_chunks") or []
    exclude_ids = {c["chunk_id"] for c in confirmed_chunks}

    logger.info("=" * 60)
    logger.info("[retrieve] 輸入")
    logger.info("[retrieve]   查詢: %s", query)
    if exclude_ids:
        logger.info("[retrieve]   排除 chunk_ids: %s", sorted(exclude_ids))

    results = store.query(
        query_text=query,
        top_k=top_k,
        exclude_ids=exclude_ids if exclude_ids else None,
    )

    # Merge confirmed_chunks (carry-over) + new results, confirmed first
    merged = confirmed_chunks + [r for r in results if r["chunk_id"] not in exclude_ids]

    logger.info("[retrieve] 輸出")
    for i, r in enumerate(merged):
        tag = "✓confirmed" if r["chunk_id"] in exclude_ids else "new"
        logger.info(
            "[retrieve]   Chunk %d [%s] | %s p%s | 相似度: %.3f | %s",
            i, tag, r["source_file"], r.get("page_number", "-"),
            r["similarity"], r["text"][:80].replace("\n", " "),
        )
    logger.info("=" * 60)

    return {"retrieved_chunks": merged}
