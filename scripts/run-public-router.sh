#!/usr/bin/env bash
# Download the public router checkpoints if needed, print Goose usage, then
# launch the OpenAI-compatible router proxy.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
PORT="${PORT:-4000}"
ROUTER_ARTIFACT_REPO="${ROUTER_ARTIFACT_REPO:-micdn/llm-router-goose-public}"
export PORT ROUTER_ARTIFACT_REPO

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

"$PYTHON" scripts/router_artifacts.py download --repo "$ROUTER_ARTIFACT_REPO"
"$PYTHON" scripts/router_artifacts.py goose-instructions \
  --repo "$ROUTER_ARTIFACT_REPO" \
  --port "$PORT"

export POOL="${POOL:-configs/combined-pool.yaml}"
export LITELLM="${LITELLM:-configs/litellm-combined.yaml}"
exec ./scripts/run.sh
