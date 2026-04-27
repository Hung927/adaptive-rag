"""Layer 4: Vector store — Qdrant-backed storage with CRUD and dedup."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)

from rag.core.config import Settings
from rag.core.embeddings import (
    AzureOpenAIEmbedding,
    CachedEmbedding,
    EmbeddingFunction,
)
from rag.core.types import DocumentInfo, QueryResult
from rag.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStoreProtocol(Protocol):
    """Vector store interface."""

    def add_chunks(self, chunks: list[Chunk]) -> int: ...

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        source_file: str | None = None,
    ) -> list[QueryResult]: ...

    def list_documents(self) -> list[DocumentInfo]: ...

    def delete_document(self, source_file: str) -> int: ...


class QdrantStore:
    """Qdrant-backed vector store."""

    def __init__(
        self,
        settings: Settings,
        embedding_fn: EmbeddingFunction | None = None,
    ):
        self.settings = settings
        self.collection_name = settings.qdrant.collection

        self.client = QdrantClient(url=settings.qdrant.url)

        if embedding_fn is not None:
            self.embedding_fn = embedding_fn
        else:
            base = AzureOpenAIEmbedding(settings)
            self.embedding_fn = CachedEmbedding(base)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            # Get vector dimension from a test embedding
            test_vec = self.embedding_fn.embed(["test"])[0]
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=len(test_vec),
                    distance=Distance.COSINE,
                ),
            )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to Qdrant. Returns number of chunks added."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedding_fn.embed(texts)

        points = []
        for chunk, vector in zip(chunks, embeddings):
            payload = {
                "text": chunk.text,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "chunk_strategy": chunk.chunk_strategy,
                "created_at": chunk.created_at.isoformat(),
            }
            if chunk.page_number is not None:
                payload["page_number"] = chunk.page_number

            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(points)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        source_file: str | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[QueryResult]:
        """Semantic search with optional filtering and dedup.

        exclude_ids: chunk_ids to exclude from results (used when carrying over
        confirmed_chunks during retrieval retry).
        """
        query_vector = self.embedding_fn.embed([query_text])[0]

        must_conditions = []
        must_not_conditions = []

        if source_file:
            must_conditions.append(
                FieldCondition(key="source_file", match=MatchValue(value=source_file))
            )
        if exclude_ids:
            must_not_conditions.append(
                HasIdCondition(has_id=list(exclude_ids))
            )

        query_filter = None
        if must_conditions or must_not_conditions:
            query_filter = Filter(
                must=must_conditions or None,
                must_not=must_not_conditions or None,
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k * 2,  # Over-fetch for dedup
            with_payload=True,
        )

        query_results: list[QueryResult] = []
        for point in results.points:
            payload = point.payload or {}
            qr: QueryResult = {
                "chunk_id": str(point.id),
                "text": payload.get("text", ""),
                "source_file": payload.get("source_file", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "similarity": point.score,
            }
            if "page_number" in payload:
                qr["page_number"] = payload["page_number"]
            query_results.append(qr)

        deduped = self._deduplicate(query_results)
        return deduped[:top_k]

    def list_documents(self) -> list[DocumentInfo]:
        """List all documents in the collection."""
        # Scroll all points to group by source_file
        all_points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=["source_file"],
        )

        docs: dict[str, int] = defaultdict(int)
        for point in all_points:
            sf = (point.payload or {}).get("source_file", "unknown")
            docs[sf] += 1

        return [
            DocumentInfo(
                document_id=sf,
                source_file=sf,
                total_chunks=count,
            )
            for sf, count in docs.items()
        ]

    def delete_document(self, source_file: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        # Find matching points
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_file", match=MatchValue(value=source_file)
                    )
                ]
            ),
            limit=10000,
        )

        if not points:
            return 0

        ids = [p.id for p in points]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )
        return len(ids)

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)

    def _deduplicate(self, results: list[QueryResult]) -> list[QueryResult]:
        """Remove duplicate chunks by word overlap > 70%."""
        grouped: dict[str, list[QueryResult]] = defaultdict(list)
        for r in results:
            grouped[r["source_file"]].append(r)

        deduped: list[QueryResult] = []
        for group in grouped.values():
            group.sort(key=lambda x: x["similarity"], reverse=True)
            kept: list[QueryResult] = []
            for candidate in group:
                if all(
                    self._overlap(candidate["text"], k["text"]) <= 0.7
                    for k in kept
                ):
                    kept.append(candidate)
            deduped.extend(kept)

        deduped.sort(key=lambda x: x["similarity"], reverse=True)
        return deduped

    @staticmethod
    def _overlap(text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)
