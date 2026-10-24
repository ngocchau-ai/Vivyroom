"""Vivyroom — Portable Offline Agent Control Room & Encrypted 3-Cores-in-1 Vivy Core Runtime.

Designed for ultra-constrained hardware (<= 2,048 MB total RAM budget) with:
1. Proprietary Encrypted 3-Cores-in-1 `Vivy Core` (`.vcore` container format — NOT open-source weights):
   - Core 1: `NPS Decision & Deep-Soul Core` (`vivy_core.dll` + `vivy_v_soul.dll`)
   - Core 2: `Cautreo Nervous System & Finance-3B-MoE 8-Expert Core` (`cautreo.dll` + `cautreo_pager.dll`)
   - Core 3: `Qwen-Decomposed LLM Semantic Compiler Core` (Bidirectional NL <-> Decision AST Transducer)
2. Zero-Relay Custom Lightweight Agent Overlay (`PocketFlow` state machine + `Smol-ToolLoop`),
   preventing all external broker/Hermes relay hangs, deadlocks, or connection conflicts.
3. Single-Screen Unified Cockpit (`ui/index.html` + `server.py`) with 1-Touch Sovereign Unlock.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "__version__",
]
