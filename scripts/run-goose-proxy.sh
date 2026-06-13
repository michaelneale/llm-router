#!/usr/bin/env bash
# Start the LLM Router proxy for goose.
# Pulls OPENAI/ANTHROPIC keys from the goose keychain entry at runtime.
set -euo pipefail
cd "$(dirname "$0")/.."

KEYS=$(security find-generic-password -s "goose" -w)
export OPENAI_API_KEY=$(echo "$KEYS" | python3 -c "import json,sys;print(json.load(sys.stdin)['OPENAI_API_KEY'])")
export ANTHROPIC_API_KEY=$(echo "$KEYS" | python3 -c "import json,sys;print(json.load(sys.stdin)['ANTHROPIC_API_KEY'])")

export ROUTER_DEVICE="${ROUTER_DEVICE:-mps}"   # M4 GPU: ~0.1-0.4s/route vs ~9s on CPU. Set ROUTER_DEVICE=cpu to override.
PORT="${PORT:-4000}"

exec .venv/bin/model-router proxy \
  --litellm-config configs/litellm-goose.yaml \
  --router-config configs/goose-mix.yaml \
  --port "$PORT"
