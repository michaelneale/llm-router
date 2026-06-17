#!/usr/bin/env bash
# Legacy compatibility wrapper. Use ./scripts/restart-router.sh directly.
set -euo pipefail
cd "$(dirname "$0")/.."

exec ./scripts/restart-router.sh
