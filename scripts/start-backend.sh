#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
runtime_dir="$project_dir/.runtime"
mkdir -p "$runtime_dir"
backend_port="${BACKEND_PORT:-8010}"

.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port "$backend_port" --reload &
backend_pid=$!
echo "$backend_pid" > "$runtime_dir/backend.pid"

cleanup() {
  trap - EXIT INT TERM
  kill -TERM "$backend_pid" 2>/dev/null || true
  rm -f "$runtime_dir/backend.pid"
}

trap cleanup EXIT INT TERM
wait "$backend_pid"
