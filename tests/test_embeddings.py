"""Tests for embedding functions."""

from rag.core.embeddings import CachedEmbedding, LocalDeterministicEmbedding


class TestLocalDeterministicEmbedding:
    def test_deterministic(self):
        emb = LocalDeterministicEmbedding(dim=32)
        v1 = emb.embed(["hello"])[0]
        v2 = emb.embed(["hello"])[0]
        assert v1 == v2

    def test_different_texts_different_vectors(self):
        emb = LocalDeterministicEmbedding(dim=32)
        v1 = emb.embed(["hello"])[0]
        v2 = emb.embed(["world"])[0]
        assert v1 != v2

    def test_normalized(self):
        emb = LocalDeterministicEmbedding(dim=32)
        v = emb.embed(["test"])[0]
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-6


class TestCachedEmbedding:
    def test_cache_hit(self):
        base = LocalDeterministicEmbedding(dim=32)
        cached = CachedEmbedding(base, cache_size=100)

        v1 = cached.embed(["hello"])[0]
        v2 = cached.embed(["hello"])[0]
        assert v1 == v2

    def test_batch(self):
        base = LocalDeterministicEmbedding(dim=32)
        cached = CachedEmbedding(base, batch_size=2)

        results = cached.embed(["a", "b", "c", "d", "e"])
        assert len(results) == 5
        assert all(len(v) == 32 for v in results)
