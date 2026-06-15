#!/usr/bin/env bash
# Run a real Goose routed-vs-frontier AB task and summarize local evidence.
#
# This intentionally makes real provider calls unless --dry-run or --report-only
# is used. The routed side goes through scripts/goose-routed-task.sh so route
# rows, savings, and gate checks are captured. The direct side runs Goose against
# a fixed frontier model, then scripts/goose_pair_report.py compares local
# friction signals for the two sessions.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="router-ab"
ROUTED_NAME=""
OPUS_NAME=""
DIRECT_PROVIDER="${DIRECT_PROVIDER:-anthropic}"
DIRECT_MODEL="${DIRECT_MODEL:-claude-opus-4-8}"
MARKER="${ROUTER_AB_MARKER:-/tmp/router-ab.marker}"
ROUTED_MARKER="${ROUTER_AB_ROUTED_MARKER:-/tmp/router-ab-routed.marker}"
ROUTED_JSON="${ROUTER_AB_ROUTED_JSON:-/tmp/router-ab-routed.json}"
PAIR_JSON="${ROUTER_AB_PAIR_JSON:-/tmp/router-ab-pair.json}"
STATE_DIR="${ROUTER_AB_STATE_DIR:-/tmp/router-ab-state}"
WORKTREE_ROOT="${ROUTER_AB_WORKTREE_ROOT:-/tmp/router-ab-worktrees}"
ARCHIVE_DIR="${ROUTER_EVAL_ARCHIVE_DIR:-/tmp/router-eval}"
WAIT_ROWS="${ROUTER_TRIAL_WAIT_ROWS:-5}"
WAIT_TIMEOUT="${ROUTER_TRIAL_WAIT_TIMEOUT:-900}"
SHOW="${ROUTER_TRIAL_SHOW:-8}"
MARKER_SET=0
DRY_RUN=0
REPORT_ONLY=0
GATE=1
REQUIRE_CLEAN_WORKTREE=0
ISOLATE_WORKTREES=0
SOURCE_TOP=""
SOURCE_REL=""
ROUTED_WORKTREE=""
OPUS_WORKTREE=""
ROUTED_WORKDIR=""
OPUS_WORKDIR=""

usage() {
  cat >&2 <<EOF
usage: $0 [--name NAME] [--routed-name NAME] [--opus-name NAME] [--direct-provider PROVIDER] [--direct-model MODEL] [--marker PATH] [--routed-marker PATH] [--routed-json PATH] [--pair-json PATH] [--state-dir PATH] [--worktree-root PATH] [--archive-dir PATH] [--wait-rows N] [--wait-timeout SECONDS] [--show N] [--require-clean-worktree] [--isolate-worktrees] [--no-gate] [--dry-run] [--report-only] -- TASK

Runs the same Goose task through:
  1. nvidia-routed via the local router
  2. a direct frontier Goose provider/model

Then prints a local routed-vs-frontier pair report. --dry-run and --report-only
do not call providers.

For git repositories this records HEAD, status, and diffs before the AB run,
after the routed run, and after the direct frontier run into --state-dir. Use
--require-clean-worktree when you need a strict comparable starting state; in
that mode the wrapper also stops before Opus if the routed run changed files.

Use --isolate-worktrees for mutating tasks. It requires the source worktree to be
clean, creates separate detached git worktrees for routed and Opus under
--worktree-root, runs each side from the matching relative subdirectory, and
leaves the worktrees in place for diff review.

Real runs archive routed/pair JSON reports under:
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

write_marker() {
  "$ROOT/.venv/bin/python" - "$1" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"ts": time.time()}, sort_keys=True) + "\n")
print(f"AB marker: {path}")
PY
}

is_git_worktree() {
  local dir="${1:-.}"
  git -C "$dir" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

ensure_clean_worktree() {
  local dir="${1:-.}"
  if ! is_git_worktree "$dir"; then
    return
  fi
  if [[ -n "$(git -C "$dir" status --porcelain)" ]]; then
    echo "git worktree is dirty; AB runs may not be comparable." >&2
    echo "Commit/stash changes, or rerun without --require-clean-worktree." >&2
    git -C "$dir" status --short >&2
    exit 1
  fi
}

ensure_routed_did_not_change_worktree() {
  local dir="${1:-.}"
  if ! is_git_worktree "$dir"; then
    return
  fi
  if [[ -n "$(git -C "$dir" status --porcelain)" ]]; then
    echo "routed run changed the git worktree; Opus would not start from the same state." >&2
    echo "Captured routed changes under: $STATE_DIR/after-routed.*" >&2
    echo "Use a read-only task, isolate each side in separate worktrees, or rerun without --require-clean-worktree." >&2
    git -C "$dir" status --short >&2
    exit 1
  fi
}

capture_git_state() {
  local label="$1"
  local dir="${2:-.}"
  if ! is_git_worktree "$dir"; then
    return
  fi
  mkdir -p "$STATE_DIR"
  git -C "$dir" rev-parse HEAD > "$STATE_DIR/$label.head"
  git -C "$dir" status --short > "$STATE_DIR/$label.status"
  git -C "$dir" diff --binary > "$STATE_DIR/$label.diff"
  git -C "$dir" diff --cached --binary > "$STATE_DIR/$label.cached.diff"
  echo "Captured git state: $STATE_DIR/$label.{head,status,diff,cached.diff}"
}

print_git_state_note() {
  if ! is_git_worktree .; then
    echo "Git state   : not a git worktree"
    return
  fi
  local dirty
  dirty="$(git status --porcelain | wc -l | tr -d ' ')"
  echo "Git state   : $dirty dirty/untracked rows"
  echo "State dir   : $STATE_DIR"
  if [[ "$REQUIRE_CLEAN_WORKTREE" == "1" ]]; then
    echo "Clean gate  : enabled (also stops if routed changes files)"
  else
    echo "Clean gate  : disabled"
  fi
  if [[ "$ISOLATE_WORKTREES" == "1" ]]; then
    echo "Isolation   : enabled"
    echo "Worktree root: $WORKTREE_ROOT"
  else
    echo "Isolation   : disabled"
  fi
}

prepare_isolated_worktrees() {
  if ! is_git_worktree .; then
    echo "--isolate-worktrees requires running from inside a git worktree." >&2
    exit 1
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "source git worktree is dirty; isolated AB worktrees would not include these changes." >&2
    echo "Commit/stash changes, or run a read-only/non-isolated comparison." >&2
    git status --short >&2
    exit 1
  fi

  SOURCE_TOP="$(git rev-parse --show-toplevel)"
  SOURCE_REL="$(git rev-parse --show-prefix)"
  local head
  head="$(git rev-parse HEAD)"
  local stamp base
  stamp="$(date +%Y%m%d-%H%M%S)-$$"
  base="$WORKTREE_ROOT/$NAME-$stamp"
  ROUTED_WORKTREE="$base/routed"
  OPUS_WORKTREE="$base/opus"
  mkdir -p "$base"
  git -C "$SOURCE_TOP" worktree add --detach "$ROUTED_WORKTREE" "$head"
  git -C "$SOURCE_TOP" worktree add --detach "$OPUS_WORKTREE" "$head"

  ROUTED_WORKDIR="$ROUTED_WORKTREE/${SOURCE_REL%/}"
  OPUS_WORKDIR="$OPUS_WORKTREE/${SOURCE_REL%/}"
  if [[ -z "${SOURCE_REL%/}" ]]; then
    ROUTED_WORKDIR="$ROUTED_WORKTREE"
    OPUS_WORKDIR="$OPUS_WORKTREE"
  fi
  if [[ ! -d "$ROUTED_WORKDIR" || ! -d "$OPUS_WORKDIR" ]]; then
    echo "created worktrees, but relative source directory was not found:" >&2
    echo "  routed: $ROUTED_WORKDIR" >&2
    echo "  opus  : $OPUS_WORKDIR" >&2
    exit 1
  fi

  echo "Created isolated worktrees:"
  echo "  routed: $ROUTED_WORKTREE"
  echo "  opus  : $OPUS_WORKTREE"
  echo "Review/cleanup when done:"
  echo "  git -C $(printf '%q' "$SOURCE_TOP") worktree remove $(printf '%q' "$ROUTED_WORKTREE")"
  echo "  git -C $(printf '%q' "$SOURCE_TOP") worktree remove $(printf '%q' "$OPUS_WORKTREE")"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      NAME="$2"
      shift 2
      ;;
    --routed-name)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      ROUTED_NAME="$2"
      shift 2
      ;;
    --opus-name)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      OPUS_NAME="$2"
      shift 2
      ;;
    --direct-provider)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      DIRECT_PROVIDER="$2"
      shift 2
      ;;
    --direct-model)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      DIRECT_MODEL="$2"
      shift 2
      ;;
    --marker)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      MARKER="$2"
      MARKER_SET=1
      shift 2
      ;;
    --routed-marker)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      ROUTED_MARKER="$2"
      shift 2
      ;;
    --routed-json)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      ROUTED_JSON="$2"
      shift 2
      ;;
    --pair-json)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      PAIR_JSON="$2"
      shift 2
      ;;
    --state-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      STATE_DIR="$2"
      shift 2
      ;;
    --worktree-root)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      WORKTREE_ROOT="$2"
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
    --no-gate)
      GATE=0
      shift
      ;;
    --require-clean-worktree)
      REQUIRE_CLEAN_WORKTREE=1
      shift
      ;;
    --isolate-worktrees)
      ISOLATE_WORKTREES=1
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
ROUTED_NAME="${ROUTED_NAME:-$NAME-routed}"
OPUS_NAME="${OPUS_NAME:-$NAME-opus}"

if [[ "$REPORT_ONLY" != "1" && -z "$TASK" ]]; then
  usage
  exit 2
fi

ROUTED_ARGS=(
  "$ROOT/scripts/goose-routed-task.sh"
  --name "$ROUTED_NAME"
  --marker "$ROUTED_MARKER"
  --json-out "$ROUTED_JSON"
  --archive-dir "$ARCHIVE_DIR"
  --wait-rows "$WAIT_ROWS"
  --wait-timeout "$WAIT_TIMEOUT"
  --show "$SHOW"
)
if [[ "$GATE" != "1" ]]; then
  ROUTED_ARGS+=(--no-gate)
fi

PAIR_ARGS=(
  "$ROOT/scripts/goose_pair_report.py"
  --routed-name "$ROUTED_NAME"
  --opus-name "$OPUS_NAME"
  --opus-model "$DIRECT_MODEL"
  --json-out "$PAIR_JSON"
)
if [[ "$GATE" == "1" ]]; then
  PAIR_ARGS+=(--gate)
fi
if [[ "$REPORT_ONLY" != "1" || "$MARKER_SET" == "1" || -f "$MARKER" ]]; then
  PAIR_ARGS+=(--marker "$MARKER")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "AB marker    : $MARKER"
  echo "Routed name  : $ROUTED_NAME"
  echo "Opus name    : $OPUS_NAME"
  echo "Direct       : $DIRECT_PROVIDER / $DIRECT_MODEL"
  echo "Archive      : $ARCHIVE_DIR"
  print_git_state_note
  echo
  if [[ "$REPORT_ONLY" != "1" ]]; then
    if [[ "$ISOLATE_WORKTREES" == "1" ]]; then
      echo "Would create isolated worktrees under: $WORKTREE_ROOT"
      echo
    fi
    echo "Would write AB marker:"
    quote_cmd "$ROOT/.venv/bin/python" - "$MARKER"
    echo
    echo "Would run routed Goose provider call:"
    if [[ "$ISOLATE_WORKTREES" == "1" ]]; then
      echo "  from routed worktree relative directory"
    fi
    quote_cmd "${ROUTED_ARGS[@]}" -- "$TASK"
    echo
    echo "Would run direct Goose provider call:"
    if [[ "$ISOLATE_WORKTREES" == "1" ]]; then
      echo "  from opus worktree relative directory"
    fi
    printf 'GOOSE_PROVIDER=%q GOOSE_MODEL=%q ' "$DIRECT_PROVIDER" "$DIRECT_MODEL"
    quote_cmd goose run --name "$OPUS_NAME" -t "$TASK"
    echo
  fi
  echo "Would run local pair report:"
  quote_cmd "$ROOT/.venv/bin/python" "${PAIR_ARGS[@]}"
  echo
  echo "Would archive:"
  quote_cmd cp "$ROUTED_JSON" "$ARCHIVE_DIR/<timestamp>-$ROUTED_NAME-routed-trial.json"
  quote_cmd cp "$PAIR_JSON" "$ARCHIVE_DIR/<timestamp>-$NAME-ab-pair.json"
  exit 0
fi

if [[ "$REPORT_ONLY" != "1" ]]; then
  if [[ "$ISOLATE_WORKTREES" == "1" ]]; then
    capture_git_state source-before .
    prepare_isolated_worktrees
    capture_git_state before-routed "$ROUTED_WORKDIR"
    capture_git_state before-opus "$OPUS_WORKDIR"
  elif [[ "$REQUIRE_CLEAN_WORKTREE" == "1" ]]; then
    ensure_clean_worktree
  fi
  if [[ "$ISOLATE_WORKTREES" != "1" ]]; then
    capture_git_state before
  fi
  write_marker "$MARKER"
  if [[ "$ISOLATE_WORKTREES" == "1" ]]; then
    (
      cd "$ROUTED_WORKDIR"
      "${ROUTED_ARGS[@]}" -- "$TASK"
    )
    capture_git_state after-routed "$ROUTED_WORKDIR"
  else
    "${ROUTED_ARGS[@]}" -- "$TASK"
    capture_git_state after-routed
  fi
  if [[ "$ISOLATE_WORKTREES" != "1" && "$REQUIRE_CLEAN_WORKTREE" == "1" ]]; then
    ensure_routed_did_not_change_worktree
  fi
  if [[ "$ISOLATE_WORKTREES" == "1" ]]; then
    (
      cd "$OPUS_WORKDIR"
      unset LITELLM_HOST LITELLM_API_KEY
      GOOSE_PROVIDER="$DIRECT_PROVIDER" \
        GOOSE_MODEL="$DIRECT_MODEL" \
        goose run --name "$OPUS_NAME" -t "$TASK"
    )
    capture_git_state after-opus "$OPUS_WORKDIR"
  else
    (
      unset LITELLM_HOST LITELLM_API_KEY
      GOOSE_PROVIDER="$DIRECT_PROVIDER" \
        GOOSE_MODEL="$DIRECT_MODEL" \
        goose run --name "$OPUS_NAME" -t "$TASK"
    )
    capture_git_state after-opus
  fi
fi

echo
echo "AB pair report:"
set +e
"$ROOT/.venv/bin/python" "${PAIR_ARGS[@]}"
pair_status=$?
set -e
archive_json "$ROUTED_JSON" "routed-trial" "$ROUTED_NAME"
archive_json "$PAIR_JSON" "ab-pair" "$NAME"
exit "$pair_status"
