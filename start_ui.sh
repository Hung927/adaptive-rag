#!/bin/bash
# Start API + Streamlit UI
set -e

HOST="${HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

echo "Starting API server..."
uv run uvicorn rag.api.server:app --host "$HOST" --port "$API_PORT" &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
        echo "API ready!"
        break
    fi
    sleep 1
done

echo "Starting Streamlit UI on port ${UI_PORT}..."
uv run streamlit run demo/streamlit_app.py --server.port "$UI_PORT" --server.address "$HOST"

# Cleanup
kill $API_PID 2>/dev/null
