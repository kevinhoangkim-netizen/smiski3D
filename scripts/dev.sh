#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .local/matplotlib .local/cache
export MPLCONFIGDIR="$PWD/.local/matplotlib"
export XDG_CACHE_HOME="$PWD/.local/cache"
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 &
backend_pid=$!
cleanup() { kill "$backend_pid" "${frontend_pid:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
npm run dev --prefix frontend -- --strictPort &
frontend_pid=$!
wait -n "$backend_pid" "$frontend_pid"
