#!/bin/bash
# Start the RAG API server
set -e

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting RAG API on ${HOST}:${PORT}..."
uv run uvicorn rag.api.server:app --host "$HOST" --port "$PORT" --reload
