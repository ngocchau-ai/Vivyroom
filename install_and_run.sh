#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="python3"
if ! command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

echo "[Vivyroom] Starting Autonomous OranPi-PocketFlow Single Chat Control Room..."
exec "$PYTHON_BIN" -m vivyroom "$@"
