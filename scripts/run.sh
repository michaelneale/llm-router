#!/usr/bin/env bash
# Start the LLM Router proxy + savings dashboard for use with a harness (goose).
#
# Portable entrypoint:
#   - Uses OPENAI_API_KEY / ANTHROPIC_API_KEY from the environment if set.
#   - On macOS, falls back to pulling them from the "goose" keychain entry.
#   - Auto-detects the routing device (mps on Apple Silicon, else cpu).
#
# Usage:
#   ./scripts/run.sh                       # defaults: port 4000, configs/*-goose.yaml
#   PORT=4100 ./scripts/run.sh
#   ROUTER_DEVICE=cuda ./scripts/run.sh
#   POOL=configs/my-mix.yaml ./scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

POOL="${POOL:-configs/goose-mix.yaml}"
LITELLM="${LITELLM:-configs/litellm-goose.yaml}"
PORT="${PORT:-4000}"
ROUTER_ROUTE_LOG="${ROUTER_ROUTE_LOG:-/tmp/router-routes.jsonl}"
export ROUTER_ROUTE_LOG

# --- API keys: env first, then macOS keychain ---
if [[ -z "${OPENAI_API_KEY:-}" || -z "${ANTHROPIC_API_KEY:-}" ]]; then
  if command -v security >/dev/null 2>&1; then
    if KEYS=$(security find-generic-password -s "goose" -w 2>/dev/null); then
      : "${OPENAI_API_KEY:=$(echo "$KEYS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('OPENAI_API_KEY',''))")}"
      : "${ANTHROPIC_API_KEY:=$(echo "$KEYS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('ANTHROPIC_API_KEY',''))")}"
      export OPENAI_API_KEY ANTHROPIC_API_KEY
    fi
  fi
fi
if [[ -z "${OPENAI_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "WARN: no OPENAI_API_KEY / ANTHROPIC_API_KEY found (env or keychain)." >&2
  echo "      Set the keys your pool ($POOL) needs, or the upstream calls will fail." >&2
fi

# Agents often point OpenAI-compatible clients at this proxy by exporting
# OPENAI_BASE_URL=http://localhost:$PORT/v1. Do not let the proxy inherit that
# as its upstream OpenAI base, or routed OpenAI calls recurse back into itself.
if [[ "${ROUTER_PRESERVE_OPENAI_BASE_URL:-0}" != "1" ]]; then
  case "${OPENAI_BASE_URL:-}${OPENAI_API_BASE:-}" in
    *localhost:$PORT*|*127.0.0.1:$PORT*)
      unset OPENAI_BASE_URL OPENAI_API_BASE
      ;;
  esac
fi

# --- routing device: detect Apple Silicon -> mps, else cpu ---
if [[ -z "${ROUTER_DEVICE:-}" ]]; then
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    ROUTER_DEVICE="mps"   # ~0.1-0.4s/route vs ~9s on CPU
  else
    ROUTER_DEVICE="cpu"
  fi
fi
export ROUTER_DEVICE
export ROUTER_DISABLE_SWITCHING="${ROUTER_DISABLE_SWITCHING:-0}"

# --- ensure the litellm config exists (generate from the pool if missing) ---
BIN=".venv/bin/model-router"
[[ -x "$BIN" ]] || BIN="model-router"   # fall back to PATH
if [[ ! -f "$LITELLM" ]]; then
  echo "Generating $LITELLM from $POOL ..."
  "$BIN" proxy-config --config "$POOL" --output "$LITELLM"
fi

echo "Router device : $ROUTER_DEVICE"
echo "Pool          : $POOL"
if [[ -n "${ROUTER_TOLERANCE:-}" ]]; then
  echo "Tolerance     : $ROUTER_TOLERANCE (env override)"
fi
echo "Switching     : $([[ "$ROUTER_DISABLE_SWITCHING" == "1" ]] && echo disabled || echo config)"
echo "Route log     : $ROUTER_ROUTE_LOG"
echo "Proxy         : http://localhost:$PORT  (dashboard: /dashboard)"

exec "$BIN" proxy \
  --litellm-config "$LITELLM" \
  --router-config "$POOL" \
  --port "$PORT"
