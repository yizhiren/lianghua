#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
runtime_dir="$project_dir/.runtime"
dry_run=false

if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
elif [[ $# -gt 0 ]]; then
  echo "用法：$0 [--dry-run]" >&2
  exit 2
fi

process_cwd() {
  lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1
}

process_command() {
  local command
  command="$(ps -p "$1" -o command= 2>/dev/null || true)"
  echo "${command:-命令不可见}"
}

is_running() {
  lsof -a -p "$1" -d cwd -Fn 2>/dev/null | grep -q '^n'
}

belongs_to_project() {
  local pid="$1"
  [[ "$(process_cwd "$pid")" == "$project_dir" ]]
}

terminate_tree() {
  local pid="$1"
  local child

  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    terminate_tree "$child"
  done

  if ! is_running "$pid"; then
    return
  fi

  if [[ "$dry_run" == true ]]; then
    echo "将停止 PID ${pid}：$(process_command "$pid")"
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
}

stop_recorded_process() {
  local label="$1"
  local pid_file="$2"
  local pid

  [[ -f "$pid_file" ]] || return 1
  pid="$(tr -dc '0-9' < "$pid_file")"
  if [[ -z "$pid" ]] || ! is_running "$pid"; then
    rm -f "$pid_file"
    return 1
  fi
  if ! belongs_to_project "$pid"; then
    echo "跳过 ${label} PID ${pid}：进程工作目录不属于当前项目。" >&2
    return 1
  fi

  terminate_tree "$pid"
  [[ "$dry_run" == true ]] || rm -f "$pid_file"
  return 0
}

stop_port_listener() {
  local label="$1"
  local port="$2"
  local found=false
  local pid

  for pid in $(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u || true); do
    if belongs_to_project "$pid"; then
      terminate_tree "$pid"
      found=true
    else
      echo "跳过端口 ${port} 的 PID ${pid}：进程工作目录不属于当前项目。" >&2
    fi
  done

  [[ "$found" == true ]]
}

stopped=false

if stop_recorded_process "前端" "$runtime_dir/frontend.pid"; then
  stopped=true
fi
for frontend_port in 3001; do
  if stop_port_listener "前端" "$frontend_port"; then
    stopped=true
  fi
done

if stop_recorded_process "后端" "$runtime_dir/backend.pid"; then
  stopped=true
fi
for backend_port in 8010; do
  if stop_port_listener "后端" "$backend_port"; then
    stopped=true
  fi
done

if [[ "$dry_run" == true ]]; then
  [[ "$stopped" == true ]] && echo "以上进程属于当前项目；未执行停止操作。" || echo "没有发现当前项目正在运行的服务。"
elif [[ "$stopped" == true ]]; then
  echo "已向量策前端和后端发送停止信号。"
else
  echo "量策服务未运行。"
fi
