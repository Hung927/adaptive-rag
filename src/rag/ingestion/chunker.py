"""Chunking strategies — text → chunk list."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol


@dataclass
class Chunk:
    """A single document fragment with metadata."""

    text: str
    source_file: str
    page_number: int | None
    chunk_index: int
    chunk_id: str
    chunk_strategy: str
    extra_metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def make_id(source_file: str, text: str) -> str:
        import uuid
        content = f"{source_file}:{text}"
        digest = hashlib.sha256(content.encode()).digest()
        return str(uuid.UUID(bytes=digest[:16]))


class ChunkingStrategy(Protocol):
    """Chunking strategy interface."""

    def chunk(self, text: str, chunk_size: int, overlap: int) -> list[str]: ...


class RecursiveStrategy:
    """Recursive separator strategy (default)."""

    def chunk(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        return self._split(text, chunk_size, overlap, ["\n\n", "\n", " ", ""])

    def _split(
        self, text: str, size: int, overlap: int, seps: list[str]
    ) -> list[str]:
        if not seps:
            return self._by_char(text, size, overlap)

        sep = seps[0]
        parts = text.split(sep) if sep else list(text)
        chunks: list[str] = []
        current = ""

        for part in parts:
            joiner = sep if current else ""
            if len(current) + len(joiner) + len(part) <= size:
                current += joiner + part
            else:
                if current:
                    chunks.append(current)
                if len(part) > size:
                    chunks.extend(self._split(part, size, overlap, seps[1:]))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _by_char(text: str, size: int, overlap: int) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + size])
            start += size - overlap if overlap else size
        return chunks


class MarkdownStrategy:
    """Markdown header-based strategy."""

    def __init__(self, min_chunk_size: int = 100):
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        sections = re.split(r"(?=^#{1,6}\s)", text, flags=re.MULTILINE)
        chunks: list[str] = []
        current = ""

        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(current) + len(section) + 2 <= chunk_size:
                current += ("\n\n" + section) if current else section
            else:
                if current:
                    chunks.append(current)
                if len(section) > chunk_size:
                    fallback = RecursiveStrategy()
                    chunks.extend(fallback.chunk(section, chunk_size, overlap))
                    current = ""
                else:
                    current = section

        if current:
            chunks.append(current)

        if self.min_chunk_size > 0:
            merged: list[str] = []
            for c in chunks:
                if merged and len(merged[-1]) < self.min_chunk_size:
                    if len(merged[-1]) + len(c) + 2 <= chunk_size:
                        merged[-1] += "\n\n" + c
                        continue
                merged.append(c)
            chunks = merged

        return chunks


class ParagraphStrategy:
    """Paragraph-boundary strategy."""

    def chunk(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) + 2 <= chunk_size:
                current += ("\n\n" + para) if current else para
            else:
                if current:
                    chunks.append(current)
                if len(para) > chunk_size:
                    fallback = RecursiveStrategy()
                    chunks.extend(fallback.chunk(para, chunk_size, overlap))
                    current = ""
                else:
                    current = para

        if current:
            chunks.append(current)
        return chunks


# Registry
STRATEGIES: dict[str, ChunkingStrategy] = {
    "recursive": RecursiveStrategy(),
    "markdown": MarkdownStrategy(),
    "paragraph": ParagraphStrategy(),
}
