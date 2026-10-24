#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="python3"
if ! command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

if [ ! -f ".vivyroom_installed" ]; then
    echo "[Vivyroom] First-time setup on Linux / OrangePi: checking runtime..."
    if [ ! -d ".venv" ] && "$PYTHON_BIN" -m venv .venv >/dev/null 2>&1; then
        .venv/bin/python -m pip install --quiet -r requirements.txt
    else
        "$PYTHON_BIN" -m pip install --quiet -r requirements.txt || true
    fi
    echo "installed=1" > .vivyroom_installed
fi

if [ -x ".venv/bin/python" ]; then
    exec .venv/bin/python -m vivyroom "$@"
else
    exec "$PYTHON_BIN" -m vivyroom "$@"
fi
