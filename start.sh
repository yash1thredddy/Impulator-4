#!/bin/bash
# IMPULATOR Start Script
# Starts both FastAPI backend and Streamlit frontend
# Designed for HF Spaces, Docker, and local development

set -e

# Configuration with defaults
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"
export FRONTEND_PORT="${FRONTEND_PORT:-7860}"
export PYTHONPATH="${PYTHONPATH:-/app}"

echo "=================================================="
echo "Starting IMPULATOR"
echo "  Backend:  ${API_HOST}:${API_PORT}"
echo "  Frontend: port ${FRONTEND_PORT}"
echo "=================================================="

# Ensure data directories exist
mkdir -p /app/data /app/data/results /app/data/logs 2>/dev/null || true

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down gracefully..."
    if [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        wait $BACKEND_PID 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        wait $FRONTEND_PID 2>/dev/null || true
    fi
    echo "Shutdown complete."
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGINT SIGTERM SIGQUIT

# Start backend
echo "Starting backend..."
python -m uvicorn backend.main:app \
    --host $API_HOST \
    --port $API_PORT \
    --log-level info &
BACKEND_PID=$!

# Wait for backend to be ready (max 60 seconds)
echo "Waiting for backend to be ready..."
BACKEND_READY=false
for i in {1..60}; do
    if curl -sf "http://localhost:${API_PORT}/api/v1/health" > /dev/null 2>&1; then
        echo "Backend is ready! (took ${i}s)"
        BACKEND_READY=true
        break
    fi
    sleep 1
done

if [ "$BACKEND_READY" = false ]; then
    echo "WARNING: Backend health check failed after 60s, starting frontend anyway..."
fi

# Start frontend
echo "Starting frontend..."
python -m streamlit run frontend/app.py \
    --server.port $FRONTEND_PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false &
FRONTEND_PID=$!

echo "=================================================="
echo "IMPULATOR is running!"
echo "  Frontend: http://localhost:${FRONTEND_PORT}"
echo "  Backend:  http://localhost:${API_PORT}"
echo "  API Docs: http://localhost:${API_PORT}/docs"
echo "=================================================="

# Wait for both processes - if either dies, container will restart
wait $BACKEND_PID $FRONTEND_PID
