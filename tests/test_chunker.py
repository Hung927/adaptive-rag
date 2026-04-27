"""Tests for chunking strategies."""

from rag.ingestion.chunker import (
    Chunk,
    MarkdownStrategy,
    ParagraphStrategy,
    RecursiveStrategy,
)


class TestRecursiveStrategy:
    def test_basic_split(self):
        strategy = RecursiveStrategy()
        text = "Hello world. " * 100
        chunks = strategy.chunk(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_short_text_no_split(self):
        strategy = RecursiveStrategy()
        text = "Short text."
        chunks = strategy.chunk(text, chunk_size=100, overlap=0)
        assert chunks == ["Short text."]

    def test_respects_paragraph_boundaries(self):
        strategy = RecursiveStrategy()
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = strategy.chunk(text, chunk_size=30, overlap=0)
        assert len(chunks) >= 2


class TestMarkdownStrategy:
    def test_splits_by_headers(self):
        strategy = MarkdownStrategy(min_chunk_size=0)
        text = "# Title\nContent one.\n\n## Section\nContent two."
        chunks = strategy.chunk(text, chunk_size=500, overlap=0)
        assert len(chunks) >= 1

    def test_merges_short_chunks(self):
        strategy = MarkdownStrategy(min_chunk_size=50)
        text = "# A\nHi\n\n# B\nBye"
        chunks = strategy.chunk(text, chunk_size=500, overlap=0)
        assert len(chunks) == 1  # Merged because both are short


class TestParagraphStrategy:
    def test_splits_by_paragraph(self):
        strategy = ParagraphStrategy()
        text = "Paragraph one here.\n\nParagraph two here.\n\nParagraph three here."
        chunks = strategy.chunk(text, chunk_size=30, overlap=0)
        assert len(chunks) == 3


class TestChunk:
    def test_make_id_deterministic(self):
        id1 = Chunk.make_id("file.pdf", "hello world")
        id2 = Chunk.make_id("file.pdf", "hello world")
        assert id1 == id2
        assert len(id1) == 16

    def test_make_id_different_for_different_input(self):
        id1 = Chunk.make_id("file.pdf", "hello")
        id2 = Chunk.make_id("file.pdf", "world")
        assert id1 != id2
