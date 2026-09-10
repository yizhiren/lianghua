#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
remote="${DEPLOY_REMOTE:-qroot@192.168.3.176}"
remote_dir="${DEPLOY_DIR:-/home/qroot/lianghua}"
sudo_password="${DEPLOY_SUDO_PASSWORD:?请通过 DEPLOY_SUDO_PASSWORD 提供远端 sudo 密码，不要写入脚本或仓库}"

cd "$project_dir"
command -v rsync >/dev/null || { echo "部署失败：本机需要 rsync。" >&2; exit 1; }

echo "同步项目到 ${remote}:${remote_dir} ..."
ssh "$remote" "mkdir -p '$remote_dir'"
rsync -a --delete --exclude '.git' --exclude 'node_modules' --exclude '.venv' \
  --exclude '.wrangler' --exclude '.pytest_cache' --exclude 'dist' \
  "$project_dir/" "$remote:$remote_dir/"

echo "在远端构建并启动 Docker 服务 ..."
ssh "$remote" "cd '$remote_dir' && export NEXT_PUBLIC_API_URL='http://192.168.3.176:8010/api' && printf '%s\\n' '$sudo_password' | sudo -S -p '' docker compose -f docker-compose.remote.yml up -d --build"

echo "部署完成："
echo "  前端: http://192.168.3.176:3001"
echo "  API:   http://192.168.3.176:8010/docs"
