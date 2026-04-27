"""Test fixtures — isolate from .env."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove RAG env vars to prevent .env leakage into tests."""
    for key in list(os.environ):
        if key.startswith(("AZURE_OPENAI_", "QDRANT_", "CHUNK_", "EVAL_")):
            monkeypatch.delenv(key, raising=False)
