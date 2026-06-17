#!/usr/bin/env zsh
# One-command local router restart.
#
# Defaults to the active public-trace router on port 4000:
#   ./scripts/restart-router.sh
#
# Optional overrides:
#   PORT=4100 ./scripts/restart-router.sh
#   ROUTER_SOURCE_ZSHRC=0 ./scripts/restart-router.sh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

export PORT="${PORT:-4000}"
export POOL="${POOL:-configs/combined-pool.yaml}"
export LITELLM="${LITELLM:-configs/litellm-combined.yaml}"
export ROUTER_LOG="${ROUTER_LOG:-/tmp/router-${PORT}.log}"
export ROUTER_ZSHRC_LOG="${ROUTER_ZSHRC_LOG:-/tmp/router-zshrc-source.log}"

if [[ "${ROUTER_SOURCE_ZSHRC:-1}" == "1" && -f "$HOME/.zshrc" ]]; then
  set +u
  source "$HOME/.zshrc" >"$ROUTER_ZSHRC_LOG" 2>&1 || {
    echo "WARN: ~/.zshrc returned non-zero; continuing. See $ROUTER_ZSHRC_LOG" >&2
  }
  set -u
  # Interactive zsh configs often install prompt/title hooks. They are useful in
  # a terminal, but make this non-interactive launcher print control sequences.
  precmd_functions=()
  preexec_functions=()
  chpwd_functions=()
  periodic_functions=()
  unfunction precmd preexec chpwd periodic 2>/dev/null || true
  [[ -s "$ROUTER_ZSHRC_LOG" ]] || rm -f "$ROUTER_ZSHRC_LOG"
fi

echo "Stopping router on port $PORT ..."
port_pids=("${(@f)$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)}")
if (( ${#port_pids[@]} )); then
  kill "${port_pids[@]}" 2>/dev/null || true
fi
pkill -f "model-router proxy .*--port $PORT" 2>/dev/null || true

for _ in {1..20}; do
  if ! lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done
if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "ERROR: port $PORT is still in use." >&2
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2 || true
  exit 1
fi

echo "Starting router..."
echo "  pool    : $POOL"
echo "  litellm : $LITELLM"
echo "  log     : $ROUTER_LOG"

pid=$(python3 - "$ROOT" "$ROUTER_LOG" <<'PY'
import os
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
log_path = pathlib.Path(sys.argv[2])
log = log_path.open("w")
proc = subprocess.Popen(
    ["./scripts/run.sh"],
    cwd=root,
    env=os.environ.copy(),
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(proc.pid)
PY
)

echo "  pid     : $pid"

for i in {1..45}; do
  if curl -fsS "http://localhost:$PORT/v1/models" >/tmp/router-models.json 2>/dev/null; then
    echo "Router ready: http://localhost:$PORT"
    python3 - <<'PY'
import json
with open("/tmp/router-models.json", "r", encoding="utf-8") as handle:
    models = json.load(handle)["data"]
print("Models:")
for model in models:
    print(f"  {model['id']}")
PY
    exit 0
  fi
  sleep 2
done

echo "ERROR: router did not become ready. Last log lines:" >&2
tail -160 "$ROUTER_LOG" >&2 || true
exit 1
