"""Layer 1: Embedding pipeline — Azure OpenAI embeddings with LRU cache."""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from typing import Protocol

from rag.core.config import Settings


class EmbeddingFunction(Protocol):
    """Embedding function interface."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class AzureOpenAIEmbedding:
    """Embedding function using Azure OpenAI API."""

    def __init__(self, settings: Settings):
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            api_key=settings.azure_openai.api_key,
            azure_endpoint=settings.azure_openai.endpoint,
            api_version=settings.azure_openai.api_version,
        )
        self._deployment = settings.azure_openai.embedding_deployment

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model=self._deployment,
        )
        return [item.embedding for item in response.data]


class LocalDeterministicEmbedding:
    """Deterministic local embeddings for testing (no API calls)."""

    def __init__(self, dim: int = 128):
        self._dim = max(16, dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec: list[float] = []
        for i in range(self._dim):
            payload = f"{i}:{text}".encode("utf-8", errors="ignore")
            digest = hashlib.sha256(payload).digest()
            u32 = int.from_bytes(digest[:4], byteorder="big", signed=False)
            vec.append((u32 / 2147483648.0) - 1.0)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]


class CachedEmbedding:
    """LRU cache wrapper with batch splitting."""

    def __init__(
        self,
        base: EmbeddingFunction,
        cache_size: int = 10000,
        batch_size: int = 100,
    ):
        self._base = base
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self._cache_size = cache_size
        self._batch_size = batch_size

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: dict[int, list[float]] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        with self._lock:
            for i, text in enumerate(texts):
                h = self._hash(text)
                if h in self._cache:
                    self._cache.move_to_end(h)
                    results[i] = self._cache[h]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

        if uncached_texts:
            new_embeddings: list[list[float]] = []
            for start in range(0, len(uncached_texts), self._batch_size):
                batch = uncached_texts[start : start + self._batch_size]
                new_embeddings.extend(self._base.embed(batch))

            with self._lock:
                for idx, text, emb in zip(
                    uncached_indices, uncached_texts, new_embeddings
                ):
                    h = self._hash(text)
                    while len(self._cache) >= self._cache_size:
                        self._cache.popitem(last=False)
                    self._cache[h] = emb
                    results[idx] = emb

        return [results[i] for i in range(len(texts))]
