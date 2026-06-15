#!/usr/bin/env bash
# Start the verified-trace router used for real Goose evaluation.
#
# This pins the checkpoint/pool trained from verified multi-model traces and
# leaves cache-aware switching controlled by the pool config. Set
# ROUTER_DISABLE_SWITCHING=1 to force pure per-turn routing.
set -euo pipefail
cd "$(dirname "$0")/.."

export POOL="${POOL:-configs/combined-pool.yaml}"
export LITELLM="${LITELLM:-configs/litellm-combined.yaml}"
export ROUTER_ROUTE_LOG="${ROUTER_ROUTE_LOG:-/tmp/router-routes-real.jsonl}"
export ROUTER_DISABLE_SWITCHING="${ROUTER_DISABLE_SWITCHING:-0}"
# This machine has the verified router encoder cached locally. Stay offline by
# default so test runs do not stall on Hugging Face metadata checks.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec ./scripts/run.sh
