#!/usr/bin/env bash
# Run one Goose task through the verified router and print a gated trial report.
#
# Usage:
#   scripts/goose-routed-task.sh --name router-smoke -- "list files in this dir"
#   scripts/goose-routed-task.sh --dry-run -- "list files in this dir"
#   scripts/goose-routed-task.sh --report-only --marker /tmp/router-trial.marker
#
# This intentionally makes real provider calls via Goose. Run it from the repo
# whose context/AGENTS.md you want Goose to see.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-4000}"
ROUTER_URL="${ROUTER_URL:-http://localhost:$PORT}"
ROUTE_LOG="${ROUTER_ROUTE_LOG:-/tmp/router-routes-real.jsonl}"
MARKER="${ROUTER_TRIAL_MARKER:-/tmp/router-trial.marker}"
JSON_OUT="${ROUTER_TRIAL_JSON:-/tmp/router-trial.json}"
ARCHIVE_DIR="${ROUTER_EVAL_ARCHIVE_DIR:-/tmp/router-eval}"
WAIT_ROWS="${ROUTER_TRIAL_WAIT_ROWS:-1}"
WAIT_TIMEOUT="${ROUTER_TRIAL_WAIT_TIMEOUT:-300}"
SHOW="${ROUTER_TRIAL_SHOW:-8}"
MAX_VERIFIED_LOSS_PP="${ROUTER_TRIAL_MAX_VERIFIED_LOSS_PP:-1.5}"
MIN_VERIFIED_SAVINGS_PCT="${ROUTER_TRIAL_MIN_VERIFIED_SAVINGS_PCT:-50}"
NAME="routed-task"
MARKER_SET=0
WAIT_ROWS_SET=0
RESET=1
GATE=1
ENSURE_ROUTER=1
DRY_RUN=0
REPORT_ONLY=0

usage() {
  cat >&2 <<EOF
usage: $0 [--name NAME] [--no-reset] [--no-gate] [--no-ensure-router] [--dry-run] [--report-only] [--marker PATH] [--json-out PATH] [--archive-dir PATH] [--wait-rows N] [--wait-timeout SECONDS] [--show N] [--max-verified-loss-pp N] [--min-verified-savings-pct N] -- TASK

Runs a real Goose task through nvidia-routed, bounded by a marker timestamp, then
prints scripts/router_trial_report.py for just that task window.

By default this will start the verified router via scripts/router-service.sh if
localhost is not reachable. --dry-run and --report-only do not run Goose.

Default gate target: small-loss high-savings. It requires offline verified-label
calibration, <= $MAX_VERIFIED_LOSS_PP percentage points verified-label loss, and
>= $MIN_VERIFIED_SAVINGS_PCT% verified-label savings for the observed route
tolerance.

Real runs archive the JSON report under:
  $ARCHIVE_DIR
EOF
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

archive_json() {
  local path="$1"
  local kind="$2"
  local label="$3"
  if [[ ! -f "$path" ]]; then
    return
  fi
  mkdir -p "$ARCHIVE_DIR"
  local safe_label
  safe_label="${label//[^[:alnum:]._-]/_}"
  local dest="$ARCHIVE_DIR/$(date +%Y%m%d-%H%M%S)-$safe_label-$kind.json"
  cp "$path" "$dest"
  echo "Archived $kind JSON: $dest"
}

build_report_args() {
  REPORT_ARGS=(
    "$ROOT/scripts/router_trial_report.py"
    --route-log "$ROUTE_LOG"
    --wait-real-rows "$WAIT_ROWS"
    --wait-timeout "$WAIT_TIMEOUT"
    --json-out "$JSON_OUT"
    --show "$SHOW"
  )
  if [[ "$REPORT_ONLY" != "1" || "$MARKER_SET" == "1" ]]; then
    REPORT_ARGS+=(--marker "$MARKER")
  fi
  if [[ "$GATE" == "1" ]]; then
    REPORT_ARGS+=(
      --gate
      --require-calibration
      --max-verified-loss-pp "$MAX_VERIFIED_LOSS_PP"
      --min-verified-savings-pct "$MIN_VERIFIED_SAVINGS_PCT"
    )
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      NAME="$2"
      shift 2
      ;;
    --no-reset)
      RESET=0
      shift
      ;;
    --no-gate)
      GATE=0
      shift
      ;;
    --no-ensure-router)
      ENSURE_ROUTER=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --report-only)
      REPORT_ONLY=1
      shift
      ;;
    --marker)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      MARKER="$2"
      MARKER_SET=1
      shift 2
      ;;
    --json-out)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      JSON_OUT="$2"
      shift 2
      ;;
    --archive-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      ARCHIVE_DIR="$2"
      shift 2
      ;;
    --wait-rows)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      WAIT_ROWS="$2"
      WAIT_ROWS_SET=1
      shift 2
      ;;
    --wait-timeout)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      WAIT_TIMEOUT="$2"
      shift 2
      ;;
    --show)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      SHOW="$2"
      shift 2
      ;;
    --max-verified-loss-pp)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      MAX_VERIFIED_LOSS_PP="$2"
      shift 2
      ;;
    --min-verified-savings-pct)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      MIN_VERIFIED_SAVINGS_PCT="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

TASK="$*"
if [[ "$REPORT_ONLY" != "1" && -z "$TASK" ]]; then
  usage
  exit 2
fi
if [[ "$REPORT_ONLY" == "1" && "$WAIT_ROWS_SET" == "0" ]]; then
  WAIT_ROWS=0
fi

build_report_args

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Router URL: $ROUTER_URL"
  echo "Route log : $ROUTE_LOG"
  echo "Marker    : $MARKER"
  echo "JSON out  : $JSON_OUT"
  echo "Archive   : $ARCHIVE_DIR"
  if [[ "$GATE" == "1" ]]; then
    echo "Gate      : max verified loss ${MAX_VERIFIED_LOSS_PP}pp, min verified savings ${MIN_VERIFIED_SAVINGS_PCT}%"
  else
    echo "Gate      : disabled"
  fi
  echo
  echo "Would ensure router:"
  if [[ "$ENSURE_ROUTER" == "1" ]]; then
    quote_cmd "$ROOT/scripts/router-service.sh" start
  else
    echo "  disabled"
  fi
  if [[ "$REPORT_ONLY" != "1" ]]; then
    echo
    echo "Would preflight:"
    quote_cmd "$ROOT/.venv/bin/python" "$ROOT/scripts/preflight_verified_router.py" \
      --base-url "$ROUTER_URL" \
      --port "$PORT"
    echo
    echo "Would run Goose provider call:"
    printf 'LITELLM_HOST=%q LITELLM_API_KEY=%q GOOSE_PROVIDER=litellm GOOSE_MODEL=nvidia-routed ' \
      "$ROUTER_URL" "${LITELLM_API_KEY:-sk-local}"
    quote_cmd goose run --name "$NAME" -t "$TASK"
  fi
  echo
  echo "Would run report:"
  quote_cmd "$ROOT/.venv/bin/python" "${REPORT_ARGS[@]}"
  if [[ "$REPORT_ONLY" != "1" ]]; then
    echo
    echo "Would archive:"
    quote_cmd cp "$JSON_OUT" "$ARCHIVE_DIR/<timestamp>-$NAME-routed-trial.json"
  fi
  exit 0
fi

if ! curl -fsS "$ROUTER_URL/v1/models" >/dev/null; then
  if [[ "$ENSURE_ROUTER" == "1" ]]; then
    "$ROOT/scripts/router-service.sh" start
  fi
  if ! curl -fsS "$ROUTER_URL/v1/models" >/dev/null; then
    echo "router is not reachable at $ROUTER_URL" >&2
    echo "start it with: $ROOT/scripts/router-service.sh start" >&2
    exit 1
  fi
fi

if [[ "$REPORT_ONLY" == "1" ]]; then
  "$ROOT/.venv/bin/python" "${REPORT_ARGS[@]}"
  exit $?
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/preflight_verified_router.py" \
  --base-url "$ROUTER_URL" \
  --port "$PORT"

if [[ "$RESET" == "1" ]]; then
  curl -fsS -X POST "$ROUTER_URL/savings/reset" >/dev/null
  : > "$ROUTE_LOG"
fi

"$ROOT/.venv/bin/python" - "$MARKER" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"ts": time.time()}, sort_keys=True) + "\n")
print(f"Trial marker: {path}")
PY

set +e
LITELLM_HOST="$ROUTER_URL" \
LITELLM_API_KEY="${LITELLM_API_KEY:-sk-local}" \
GOOSE_PROVIDER=litellm \
GOOSE_MODEL=nvidia-routed \
goose run --name "$NAME" -t "$TASK"
goose_status=$?
set -e
if [[ "$goose_status" != "0" ]]; then
  echo "Goose run exited with status $goose_status; collecting route evidence anyway." >&2
fi

echo
echo "Trial report:"
set +e
"$ROOT/.venv/bin/python" "${REPORT_ARGS[@]}"
report_status=$?
set -e
archive_json "$JSON_OUT" "routed-trial" "$NAME"
if [[ "$goose_status" != "0" ]]; then
  exit "$goose_status"
fi
exit "$report_status"
