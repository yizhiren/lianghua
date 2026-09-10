#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
runtime_dir="$project_dir/.runtime"
mkdir -p "$runtime_dir"
frontend_port="${FRONTEND_PORT:-3001}"
backend_port="${BACKEND_PORT:-8010}"

node_is_compatible() {
  command -v node >/dev/null 2>&1 && node -e '
  const [major, minor] = process.versions.node.split(".").map(Number);
  process.exit(major > 22 || (major === 22 && minor >= 13) ? 0 : 1);
  ' >/dev/null 2>&1
}

if ! node_is_compatible; then
  for node_bin_dir in /opt/homebrew/opt/node@22/bin /usr/local/opt/node@22/bin; do
    if [[ -x "$node_bin_dir/node" ]]; then
      export PATH="$node_bin_dir:$PATH"
      break
    fi
  done
fi

if ! node_is_compatible; then
  echo "启动失败：需要 Node.js 22.13+，当前为 $(node -v 2>/dev/null || echo '未安装')。请先升级 Node.js，具体命令见 README。" >&2
  exit 1
fi

echo "使用 Node.js $(node -v) ($(command -v node))"

if [[ ! -x .venv/bin/uvicorn ]]; then
  echo "首次运行请先执行：python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port "$backend_port" &
backend_pid=$!
echo "$backend_pid" > "$runtime_dir/backend.pid"

npm run dev -- --port "$frontend_port" &
frontend_pid=$!
echo "$frontend_pid" > "$runtime_dir/frontend.pid"

cleanup() {
  trap - EXIT INT TERM
  kill -TERM "$frontend_pid" 2>/dev/null || true
  kill -TERM "$backend_pid" 2>/dev/null || true
  rm -f "$runtime_dir/frontend.pid" "$runtime_dir/backend.pid"
}

trap cleanup EXIT INT TERM
wait "$frontend_pid"
