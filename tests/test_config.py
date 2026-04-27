"""Tests for configuration loading."""

import os

import pytest

from rag.core.config import load_settings


def test_load_settings_missing_key():
    """Should raise ValueError when required env vars are missing."""
    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        load_settings()


def test_load_settings_ok(monkeypatch):
    """Should load settings when env vars are set."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")

    settings = load_settings()
    assert settings.azure_openai.api_key == "test-key"
    assert settings.qdrant.url == "http://localhost:6333"
    assert settings.chunking.chunk_size == 500
