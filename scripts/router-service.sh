#!/usr/bin/env bash
# Manage the verified router proxy in a tmux session.
#
# This script is provider-free unless you later point Goose at the running proxy.
# It only starts/stops the local router, probes localhost health endpoints, and
# manages the local route log/savings counters.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${ROUTER_TMUX_SESSION:-nvidia-router}"
PORT="${PORT:-4000}"
ROUTER_URL="${ROUTER_URL:-http://localhost:$PORT}"
ROUTE_LOG="${ROUTER_ROUTE_LOG:-/tmp/router-routes-real.jsonl}"
SHOW="${ROUTER_TRIAL_SHOW:-8}"

usage() {
  cat >&2 <<EOF
usage: $0 COMMAND

Commands:
  start      Start the verified router in tmux session '$SESSION'
  stop       Stop the tmux-managed router
  restart    Stop then start
  status     Show tmux status, localhost health, route-log row count, and recent logs
  diagnose   Explain Goose/router wiring and whether route evidence exists
  preflight  Run provider-free router preflight against localhost
  clean      Reset savings and truncate the route log
  report     Run provider-free trial report for the current route log
  closeout   Gate/archive the current live route log at the knee setting
  shadow     Shadow-replay recent Goose sessions through the current router
  points     Report same-quality vs bounded-loss operating points
  points-current
             Report operating points on the latest all-Goose traffic slice
  points-current-savings
             Report latest all-Goose slice optimized for proxy savings
  summary    Aggregate routed trial + AB pair JSON evidence
  suggest    Suggest representative Goose tasks for real AB validation

Environment:
  ROUTER_TMUX_SESSION=$SESSION
  PORT=$PORT
  ROUTER_URL=$ROUTER_URL
  ROUTER_ROUTE_LOG=$ROUTE_LOG
  ROUTER_SELECTOR_SESSIONS=24
  ROUTER_SELECTOR_MAX_USER_TURNS=6
  ROUTER_SELECTOR_MODEL_LIKE=%nvidia-routed%
  ROUTER_SELECTOR_MAX_LOSS_PP=1.5
  ROUTER_SELECTOR_TIE_BREAKER=lower_loss
  ROUTER_SELECTOR_JSON=/tmp/router-operating-points.json
  ROUTER_SELECTOR_CURRENT_SESSIONS=12
  ROUTER_SELECTOR_CURRENT_MAX_USER_TURNS=2
  ROUTER_SELECTOR_CURRENT_MODEL_LIKE=%
  ROUTER_SELECTOR_CURRENT_JSON=/tmp/router-operating-points-current.json
  ROUTER_SELECTOR_CURRENT_SAVINGS_JSON=/tmp/router-operating-points-current-savings.json
  ROUTER_SELECTOR_CURRENT_SCORES_CACHE=/tmp/router-goose-current-route-scores.json
  ROUTER_EVAL_DIR=/tmp/router-eval
  ROUTER_EVAL_LABELS=/tmp/router-eval/labels.jsonl
  ROUTER_EVAL_JSON=/tmp/router-eval-summary.json
  ROUTER_DIAGNOSE_SESSIONS=80
  ROUTER_DIAGNOSE_SHOW=20
  ROUTER_LIVE_JSON=/tmp/router-live-trial.json
  ROUTER_LIVE_ARCHIVE_DIR=/tmp/router-eval
  ROUTER_LIVE_GATE=1
  ROUTER_LIVE_MIN_REAL_ROWS=1
  ROUTER_LIVE_MAX_VERIFIED_LOSS_PP=1.5
  ROUTER_LIVE_MIN_VERIFIED_SAVINGS_PCT=50
  ROUTER_LIVE_SINCE_MINUTES=0
  ROUTER_SHADOW_JSONL=/tmp/router-routes-shadow-knee.jsonl
  ROUTER_SHADOW_SESSIONS=12
  ROUTER_SHADOW_MAX_USER_TURNS=2
  ROUTER_SHADOW_MODEL_LIKE=%
  ROUTER_SHADOW_PROGRESS_EVERY=2
  ROUTER_SUGGEST_SESSIONS=80
  ROUTER_SUGGEST_COUNT=6
  ROUTER_SUGGEST_JSON=/tmp/router-validation-tasks.json
EOF
}

has_session() {
  tmux has-session -t "$SESSION" >/dev/null 2>&1
}

wait_ready() {
  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    if curl -fsS "$ROUTER_URL/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "router did not become ready at $ROUTER_URL within 60s" >&2
  return 1
}

start_router() {
  if has_session; then
    echo "tmux session '$SESSION' already exists"
    wait_ready
    return
  fi
  mkdir -p "$(dirname "$ROUTE_LOG")"
  local quoted_route_log
  printf -v quoted_route_log "%q" "$ROUTE_LOG"
  tmux new-session -d -s "$SESSION" -c "$ROOT" \
    "env HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0} ROUTER_ROUTE_LOG=$quoted_route_log ./scripts/run-verified-router.sh"
  echo "started tmux session '$SESSION'"
  wait_ready
}

stop_router() {
  if has_session; then
    tmux kill-session -t "$SESSION"
    echo "stopped tmux session '$SESSION'"
  else
    echo "tmux session '$SESSION' is not running"
  fi
}

status_router() {
  if has_session; then
    tmux ls | sed -n "/^$SESSION:/p"
  else
    echo "tmux session '$SESSION' is not running"
  fi

  if curl -fsS "$ROUTER_URL/v1/models" >/dev/null 2>&1; then
    echo "router health: OK at $ROUTER_URL"
  else
    echo "router health: not reachable at $ROUTER_URL"
  fi

  if [[ -f "$ROUTE_LOG" ]]; then
    wc -l "$ROUTE_LOG"
  else
    echo "route log missing: $ROUTE_LOG"
  fi

  if curl -fsS "$ROUTER_URL/savings" >/tmp/router-service-savings.json 2>/dev/null; then
    "$ROOT/.venv/bin/python" - /tmp/router-service-savings.json <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
print(
    "savings: "
    f"requests={data.get('requests', 0)} "
    f"saved_pct={float(data.get('saved_pct') or 0):.1f}% "
    f"saved_usd=${float(data.get('saved_usd') or 0):.4f}"
)
PY
  fi

  if has_session; then
    echo
    echo "recent tmux output:"
    tmux capture-pane -pt "$SESSION" -S -40
  fi
}

preflight_router() {
  "$ROOT/.venv/bin/python" "$ROOT/scripts/preflight_verified_router.py" \
    --base-url "$ROUTER_URL" \
    --port "$PORT"
}

diagnose_router() {
  "$ROOT/.venv/bin/python" "$ROOT/scripts/analyze_goose_router_trials.py" \
    --route-log "$ROUTE_LOG" \
    --sessions "${ROUTER_DIAGNOSE_SESSIONS:-80}" \
    --show "${ROUTER_DIAGNOSE_SHOW:-20}"
}

clean_router() {
  if curl -fsS -X POST "$ROUTER_URL/savings/reset" >/dev/null 2>&1; then
    echo "reset savings at $ROUTER_URL"
  else
    echo "warning: could not reset savings at $ROUTER_URL" >&2
  fi
  : > "$ROUTE_LOG"
  echo "truncated $ROUTE_LOG"
}

report_router() {
  "$ROOT/.venv/bin/python" "$ROOT/scripts/router_trial_report.py" \
    --route-log "$ROUTE_LOG" \
    --savings-timeout 1 \
    --show "$SHOW"
}

archive_live_json() {
  local path="$1"
  local label="$2"
  local archive_dir="${ROUTER_LIVE_ARCHIVE_DIR:-${ROUTER_EVAL_DIR:-/tmp/router-eval}}"
  if [[ ! -f "$path" ]]; then
    return
  fi
  mkdir -p "$archive_dir"
  local safe_label
  safe_label="${label//[^[:alnum:]._-]/_}"
  local dest="$archive_dir/$(date +%Y%m%d-%H%M%S)-$safe_label-routed-trial.json"
  cp "$path" "$dest"
  echo "Archived live routed-trial JSON: $dest"
}

closeout_live_router() {
  local json_out="${ROUTER_LIVE_JSON:-/tmp/router-live-trial.json}"
  local label="${ROUTER_LIVE_LABEL:-live-knee}"
  local args=(
    "$ROOT/scripts/router_trial_report.py"
    --route-log "$ROUTE_LOG"
    --savings-timeout 1
    --json-out "$json_out"
    --show "$SHOW"
    --min-real-rows "${ROUTER_LIVE_MIN_REAL_ROWS:-1}"
    --max-verified-loss-pp "${ROUTER_LIVE_MAX_VERIFIED_LOSS_PP:-1.5}"
    --min-verified-savings-pct "${ROUTER_LIVE_MIN_VERIFIED_SAVINGS_PCT:-50}"
    --require-calibration
  )
  if [[ "${ROUTER_LIVE_GATE:-1}" == "1" ]]; then
    args+=(--gate)
  fi
  if [[ -n "${ROUTER_LIVE_MARKER:-}" ]]; then
    args+=(--marker "$ROUTER_LIVE_MARKER")
  fi
  if [[ "${ROUTER_LIVE_SINCE_MINUTES:-0}" != "0" ]]; then
    args+=(--since-minutes "${ROUTER_LIVE_SINCE_MINUTES:-0}")
  fi

  local status=0
  "$ROOT/.venv/bin/python" "${args[@]}" || status=$?
  archive_live_json "$json_out" "$label"
  echo
  evaluation_summary || true
  return "$status"
}

shadow_replay_router() {
  local output="${ROUTER_SHADOW_JSONL:-/tmp/router-routes-shadow-knee.jsonl}"
  local shadow_args=(
    "$ROOT/scripts/shadow_goose_route_log.py"
    --model-like "${ROUTER_SHADOW_MODEL_LIKE:-%}"
    --sessions "${ROUTER_SHADOW_SESSIONS:-12}"
    --max-user-turns "${ROUTER_SHADOW_MAX_USER_TURNS:-2}"
    --output "$output"
    --progress-every "${ROUTER_SHADOW_PROGRESS_EVERY:-2}"
  )
  if [[ -n "${ROUTER_SHADOW_TOLERANCE:-}" ]]; then
    shadow_args+=(--tolerance "$ROUTER_SHADOW_TOLERANCE")
  fi

  "$ROOT/.venv/bin/python" "${shadow_args[@]}"
  echo
  "$ROOT/.venv/bin/python" "$ROOT/scripts/analyze_route_log.py" \
    --log "$output" \
    --show-expensive "${ROUTER_SHADOW_SHOW_EXPENSIVE:-8}"
}

operating_points_report() {
  "$ROOT/.venv/bin/python" "$ROOT/scripts/select_router_operating_point.py" \
    --sessions "${ROUTER_SELECTOR_SESSIONS:-24}" \
    --max-user-turns "${ROUTER_SELECTOR_MAX_USER_TURNS:-6}" \
    --model-like "${ROUTER_SELECTOR_MODEL_LIKE:-%nvidia-routed%}" \
    --max-loss-pp "${ROUTER_SELECTOR_MAX_LOSS_PP:-1.5}" \
    --objective "${ROUTER_SELECTOR_OBJECTIVE:-goose_cache_savings}" \
    --tie-breaker "${ROUTER_SELECTOR_TIE_BREAKER:-lower_loss}" \
    --objective-tie-slack "${ROUTER_SELECTOR_TIE_SLACK:-0}" \
    --loss-bands-pp "${ROUTER_SELECTOR_LOSS_BANDS_PP:-0,0.5,1,1.5,2}" \
    --summary-only \
    --json-out "${ROUTER_SELECTOR_JSON:-/tmp/router-operating-points.json}"
}

current_operating_points_report() {
  local objective="${1:-${ROUTER_SELECTOR_OBJECTIVE:-goose_cache_savings}}"
  local json_out="${2:-${ROUTER_SELECTOR_CURRENT_JSON:-/tmp/router-operating-points-current.json}}"
  "$ROOT/.venv/bin/python" "$ROOT/scripts/select_router_operating_point.py" \
    --sessions "${ROUTER_SELECTOR_CURRENT_SESSIONS:-12}" \
    --max-user-turns "${ROUTER_SELECTOR_CURRENT_MAX_USER_TURNS:-2}" \
    --model-like "${ROUTER_SELECTOR_CURRENT_MODEL_LIKE:-%}" \
    --max-loss-pp "${ROUTER_SELECTOR_MAX_LOSS_PP:-1.5}" \
    --objective "$objective" \
    --tie-breaker "${ROUTER_SELECTOR_TIE_BREAKER:-lower_loss}" \
    --objective-tie-slack "${ROUTER_SELECTOR_TIE_SLACK:-0}" \
    --loss-bands-pp "${ROUTER_SELECTOR_LOSS_BANDS_PP:-0,0.5,1,1.5,2}" \
    --scores-cache "${ROUTER_SELECTOR_CURRENT_SCORES_CACHE:-/tmp/router-goose-current-route-scores.json}" \
    --summary-only \
    --json-out "$json_out"
}

evaluation_summary() {
  local eval_dir="${ROUTER_EVAL_DIR:-/tmp/router-eval}"
  local args=(
    "$ROOT/scripts/router_eval_summary.py"
    --json /tmp/router-trial.json
    --json /tmp/router-ab-routed.json
    --json /tmp/router-ab-pair.json
    --labels "${ROUTER_EVAL_LABELS:-/tmp/router-eval/labels.jsonl}"
    --json-out "${ROUTER_EVAL_JSON:-/tmp/router-eval-summary.json}"
  )
  if [[ -n "$eval_dir" ]]; then
    args+=(--dir "$eval_dir")
  fi
  "$ROOT/.venv/bin/python" "${args[@]}"
}

suggest_validation_tasks() {
  local args=(
    "$ROOT/scripts/suggest_router_validation_tasks.py"
    --sessions "${ROUTER_SUGGEST_SESSIONS:-80}"
    --count "${ROUTER_SUGGEST_COUNT:-6}"
    --max-tokens "${ROUTER_SUGGEST_MAX_TOKENS:-60000}"
    --max-tool-errors "${ROUTER_SUGGEST_MAX_TOOL_ERRORS:-3}"
    --json-out "${ROUTER_SUGGEST_JSON:-/tmp/router-validation-tasks.json}"
  )
  if [[ "${ROUTER_SUGGEST_EMIT_COMMANDS:-0}" == "1" ]]; then
    args+=(--emit-commands)
  fi
  "$ROOT/.venv/bin/python" "${args[@]}"
}

cmd="${1:-}"
case "$cmd" in
  start)
    start_router
    ;;
  stop)
    stop_router
    ;;
  restart)
    stop_router
    start_router
    ;;
  status)
    status_router
    ;;
  diagnose)
    diagnose_router
    ;;
  preflight)
    preflight_router
    ;;
  clean)
    clean_router
    ;;
  report)
    report_router
    ;;
  closeout)
    closeout_live_router
    ;;
  shadow)
    shadow_replay_router
    ;;
  points)
    operating_points_report
    ;;
  points-current)
    current_operating_points_report
    ;;
  points-current-savings)
    current_operating_points_report \
      "goose_proxy_savings" \
      "${ROUTER_SELECTOR_CURRENT_SAVINGS_JSON:-/tmp/router-operating-points-current-savings.json}"
    ;;
  summary)
    evaluation_summary
    ;;
  suggest)
    suggest_validation_tasks
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
