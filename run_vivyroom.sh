#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONIOENCODING="utf-8"
export PYTHONPATH="$SCRIPT_DIR"
echo "=========================================================================="
echo " Launching Vivyroom (Portable <2GB RAM Unified Vivy + Cautreo Cockpit)"
echo "=========================================================================="
python3 -m vivyroom.server "$@"
