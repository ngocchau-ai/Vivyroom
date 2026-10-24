"""Entrypoint for `python -m vivyroom` — Zero-Crash Dependency Resolver + Unified Vivyroom Launch.

Works seamlessly across:
- Standard Python (`pip` installed)
- `uv` / `hermes-agent` virtual environments where `pip` is NOT installed (`No module named pip`)
- Air-gapped / 100% offline machines (automatically falls back to `vivyroom._pure_numpy_fallback`)
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_one_time_dependencies() -> None:
    """Quietly attempts 1-time C `numpy` setup if missing, falling back to built-in pure engine."""
    root = Path(__file__).resolve().parents[1]
    stamp = root / ".vivyroom_installed"

    # If real C numpy is already available, mark installed and return immediately
    if importlib.util.find_spec("numpy") is not None:
        if not stamp.is_file():
            stamp.write_text("installed=c_numpy\n", encoding="utf-8")
        return

    if not stamp.is_file():
        # 1. If running inside a `uv` virtualenv (which lacks `pip`), try `uv pip install` first
        uv_bin = shutil.which("uv")
        if not uv_bin:
            for cand in (
                Path.home() / ".local" / "bin" / ("uv.exe" if sys.platform == "win32" else "uv"),
                Path.home() / ".cargo" / "bin" / ("uv.exe" if sys.platform == "win32" else "uv"),
            ):
                if cand.is_file():
                    uv_bin = str(cand)
                    break

        installed = False
        if uv_bin:
            res = subprocess.run(
                [uv_bin, "pip", "install", "--python", sys.executable, "--quiet", "numpy>=1.26.0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            installed = res.returncode == 0

        # 2. If `pip` module exists in this interpreter, try `python -m pip install`
        if not installed and importlib.util.find_spec("pip") is not None:
            res = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "numpy>=1.26.0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            installed = res.returncode == 0

        # 3. If neither `uv` nor `pip` is present, try `ensurepip` quietly
        if not installed and importlib.util.find_spec("ensurepip") is not None:
            ep = subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if ep.returncode == 0:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--quiet", "numpy>=1.26.0"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )

        stamp.write_text("installed=1\n", encoding="utf-8")

    # Ensure `numpy` (either newly installed C numpy, host site-packages, or built-in pure engine) is loaded
    from . import _bootstrap_numpy_or_pure_fallback

    _bootstrap_numpy_or_pure_fallback()


if __name__ == "__main__":
    ensure_one_time_dependencies()
    from .server import main

    sys.exit(main())
