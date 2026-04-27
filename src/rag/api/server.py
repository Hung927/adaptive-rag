"""FastAPI application — REST API for the RAG system."""

from __future__ import annotations

import logging
import shutil
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from rag.core.config import Settings, load_settings
from rag.core.types import DocumentInfo, GenerationResult, IngestResult
from rag.ingestion.indexer import ingest_document
from rag.pipeline.builder import run_pipeline
from rag.retrieval.store import QdrantStore

_settings: Settings | None = None
_store: QdrantStore | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def _get_store() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore(_get_settings())
    return _store


def create_app() -> FastAPI:
    app = FastAPI(title="RAG System API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ingest", response_model=None)
    async def ingest(file: UploadFile = File(...)) -> IngestResult:
        """Upload and ingest a document (PDF, DOCX, TXT, MD)."""
        settings = _get_settings()
        store = _get_store()

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename or "doc").suffix
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            result = ingest_document(tmp_path, settings, store, original_filename=file.filename)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return result

    @app.post("/chat", response_model=None)
    def chat(query: str) -> GenerationResult:
        """Ask a question against ingested documents."""
        settings = _get_settings()
        store = _get_store()
        return run_pipeline(settings, store, query)

    @app.get("/documents", response_model=None)
    def list_documents() -> list[DocumentInfo]:
        """List all ingested documents."""
        store = _get_store()
        return store.list_documents()

    @app.delete("/documents/{source_file}")
    def delete_document(source_file: str):
        """Delete a document and its chunks."""
        store = _get_store()
        count = store.delete_document(source_file)
        return {"status": "ok", "chunks_deleted": count}

    return app


app = create_app()
