#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

lines=100
errors_only=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lines)
      [[ "${2:-}" =~ ^[0-9]+$ ]] || { echo "--lines 需要正整数" >&2; exit 2; }
      lines="$2"
      shift 2
      ;;
    --errors)
      errors_only=true
      shift
      ;;
    *)
      echo "用法：$0 [--lines 数量] [--errors]" >&2
      exit 2
      ;;
  esac
done

log_path=".runtime/backend.log"
configured_path="$(sed -n 's/^BACKEND_LOG_PATH=//p' .env 2>/dev/null | tail -n 1)"
[[ -n "$configured_path" ]] && log_path="$configured_path"

if [[ ! -f "$log_path" ]]; then
  echo "日志文件尚不存在：$log_path" >&2
  echo "启动服务后再执行此命令。" >&2
  exit 1
fi

echo "正在跟踪 ${log_path}（Ctrl+C 退出）"
if [[ "$errors_only" == true ]]; then
  tail -n "$lines" -F "$log_path" | rg --line-buffered '"level":"(WARNING|ERROR|CRITICAL)"|"event":"ai_request_failed"'
else
  tail -n "$lines" -F "$log_path"
fi
