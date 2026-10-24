"""Vivyroom — Portable Offline Agent Control Room & Encrypted 3-Cores-in-1 Vivy Core Runtime.

Designed for ultra-constrained hardware (<= 2,048 MB total RAM budget) with:
1. Proprietary Encrypted 3-Cores-in-1 `Vivy Core` (`.vcore` container format — NOT open-source weights):
   - Core 1: `NPS Decision & Deep-Soul Core` (`vivy_core.dll` + `vivy_v_soul.dll`)
   - Core 2: `Cautreo Nervous System & Finance-3B-MoE 8-Expert Core` (`cautreo.dll` + `cautreo_pager.dll`)
   - Core 3: `Qwen-Decomposed LLM Semantic Compiler Core` (Bidirectional NL <-> Decision AST Transducer)
2. Zero-Relay Custom Lightweight Agent Overlay (`OranPi × PocketFlow` state machine),
   preventing all external broker/Hermes relay hangs, deadlocks, or connection conflicts.
3. Single-Screen Button-Free Chat Cockpit (`ui/index.html` + `server.py`) with 100% Autonomous Skills.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_numpy_or_pure_fallback() -> None:
    """Ensures `numpy` is importable even in stripped `uv`/Hermes venvs without `pip` or offline."""
    try:
        import numpy  # noqa: F401

        return
    except ImportError:
        pass

    # 1. Search existing Python installations on the host machine for pre-installed numpy
    candidate_site_dirs: list[Path] = []
    base_exec = getattr(sys, "_base_executable", None)
    if base_exec:
        base_root = Path(base_exec).resolve().parent
        candidate_site_dirs.append(base_root / "Lib" / "site-packages")

    for env_key in ("LOCALAPPDATA", "APPDATA", "USERPROFILE", "HOME"):
        val = os.environ.get(env_key)
        if not val:
            continue
        p = Path(val)
        for sp in p.glob("Programs/Python/Python3*/Lib/site-packages"):
            candidate_site_dirs.append(sp)
        for sp in p.glob("Python/Python3*/site-packages"):
            candidate_site_dirs.append(sp)

    for sp_dir in candidate_site_dirs:
        if (sp_dir / "numpy" / "__init__.py").is_file():
            sp_str = str(sp_dir)
            if sp_str not in sys.path:
                sys.path.append(sp_str)
            try:
                import numpy  # noqa: F401

                return
            except Exception:
                if sp_str in sys.path:
                    sys.path.remove(sp_str)

    # 2. Activate built-in Zero-Dependency Pure-Python + ctypes numpy engine
    from ._pure_numpy_fallback import install_pure_numpy_fallback

    install_pure_numpy_fallback()


_bootstrap_numpy_or_pure_fallback()

__version__ = "1.1.0"

__all__ = [
    "__version__",
]
