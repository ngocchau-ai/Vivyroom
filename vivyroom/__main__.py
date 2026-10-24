"""Entrypoint for `python -m vivyroom` — 1-Time Dependency Verification + Unified Vivyroom Launch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ensure_one_time_dependencies() -> None:
    """Checks for `numpy` and installs it once if missing on a fresh machine."""
    root = Path(__file__).resolve().parents[1]
    stamp = root / ".vivyroom_installed"
    try:
        import numpy  # noqa: F401

        if not stamp.is_file():
            stamp.write_text("installed=1\n", encoding="utf-8")
    except ImportError:
        print("[Vivyroom] First-time setup: installing lightweight runtime requirement (numpy)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "numpy>=1.26.0"])
        stamp.write_text("installed=1\n", encoding="utf-8")


if __name__ == "__main__":
    ensure_one_time_dependencies()
    from .server import main

    sys.exit(main())
