"""Indexer — orchestrates load → chunk → store."""

from __future__ import annotations

import re
from pathlib import Path

from rag.core.config import Settings
from rag.core.types import IngestResult
from rag.ingestion.chunker import Chunk, STRATEGIES
from rag.ingestion.loader import get_loader
from rag.ingestion.text_cleaner import clean_chunk_text

# Table protection (preserve markdown tables from being split)
_TABLE_PATTERN = re.compile(
    r"(\|[^\n]+\|\n\|[-:|\s]+\|\n(?:\|[^\n]+\|\n)*)"
)


def _protect_tables(text: str) -> tuple[str, dict[str, str]]:
    tables: dict[str, str] = {}
    counter = 0

    def replacer(match: re.Match) -> str:
        nonlocal counter
        placeholder = f"__TABLE_{counter}__"
        tables[placeholder] = match.group(0)
        counter += 1
        return placeholder

    return _TABLE_PATTERN.sub(replacer, text), tables


def _restore_tables(chunks: list[str], table_map: dict[str, str]) -> list[str]:
    restored = []
    for chunk in chunks:
        for placeholder, table in table_map.items():
            chunk = chunk.replace(placeholder, table)
        restored.append(chunk)
    return restored


def process_document(
    file_path: str | Path,
    settings: Settings,
    strategy_name: str = "recursive",
) -> list[Chunk]:
    """Process a document into chunks.

    Args:
        file_path: Path to the document.
        settings: Application settings.
        strategy_name: Chunking strategy name.

    Returns:
        List of Chunk objects ready for vector store insertion.
    """
    file_path = Path(file_path)
    loader = get_loader(file_path)
    pages = loader.load(file_path)

    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    strategy = STRATEGIES[strategy_name]

    all_chunks: list[Chunk] = []

    for page in pages:
        text = page["text"]
        page_num = page["page_number"]

        protected_text, table_map = _protect_tables(text)
        chunk_texts = strategy.chunk(
            protected_text,
            settings.chunking.chunk_size,
            settings.chunking.chunk_overlap,
        )
        chunk_texts = _restore_tables(chunk_texts, table_map)

        for idx, chunk_text in enumerate(chunk_texts):
            chunk = Chunk(
                text=clean_chunk_text(chunk_text),
                source_file=file_path.name,
                page_number=page_num,
                chunk_index=idx,
                chunk_id=Chunk.make_id(file_path.name, chunk_text),
                chunk_strategy=strategy_name,
            )
            all_chunks.append(chunk)

    return all_chunks


def ingest_document(
    file_path: str | Path,
    settings: Settings,
    store: "QdrantStore",  # noqa: F821 — forward ref
    strategy_name: str = "recursive",
    original_filename: str | None = None,
) -> IngestResult:
    """Full ingest pipeline: load → chunk → store."""
    from rag.retrieval.store import QdrantStore

    file_path = Path(file_path)
    display_name = original_filename or file_path.name
    try:
        chunks = process_document(file_path, settings, strategy_name)
        # Override source_file in chunks to use the original filename
        for chunk in chunks:
            chunk.source_file = display_name
        count = store.add_chunks(chunks)
        return IngestResult(
            status="ok",
            source_file=display_name,
            total_chunks=count,
        )
    except Exception as e:
        return IngestResult(
            status="error",
            source_file=display_name,
            total_chunks=0,
            error=str(e),
        )
