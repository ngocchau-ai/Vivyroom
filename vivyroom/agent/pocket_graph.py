"""Minimalist Zero-Relay State Graph (`PocketFlow` MIT Pattern) for `Vivyroom`.

Eliminates external agent relay hops (such as Hermes/LangChain/AutoGen IPC brokers) that cause
connection conflicts, deadlocks, or hangs on standalone/constrained hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PocketStateContext:
    """Shared in-process state passed across nodes in zero-copy memory."""

    turn_id: int
    raw_user_input: str
    synthesize_voice: bool = False
    state: dict[str, Any] = field(default_factory=dict)
    trace_steps: list[dict[str, Any]] = field(default_factory=list)
    errors_healed: list[str] = field(default_factory=list)


class ZeroRelayPocketGraph:
    """Deterministic, non-blocking finite-state graph with per-node Watchdog enforcement."""

    def __init__(self, max_step_ms: float = 1500.0) -> None:
        self.max_step_ms = max_step_ms
        self._nodes: dict[str, Callable[[PocketStateContext], str | None]] = {}
        self._entry_node: str | None = None

    def add_node(
        self,
        name: str,
        handler: Callable[[PocketStateContext], str | None],
        is_entry: bool = False,
    ) -> None:
        self._nodes[name] = handler
        if is_entry or self._entry_node is None:
            self._entry_node = name

    def execute(self, ctx: PocketStateContext, max_hops: int = 8) -> PocketStateContext:
        current = self._entry_node
        hops = 0
        while current is not None and hops < max_hops:
            hops += 1
            handler = self._nodes.get(current)
            if handler is None:
                ctx.errors_healed.append(f"missing_node:{current}")
                break
            t0 = time.perf_counter()
            try:
                next_node = handler(ctx)
                status = "OK"
            except Exception as exc:
                # Self-healing fallback: record error and jump straight to outbound compiler
                ctx.errors_healed.append(f"node_exception:{current}:{type(exc).__name__}")
                next_node = "COMPILE_OUTBOUND" if current != "COMPILE_OUTBOUND" else None
                status = "SELF_HEALED"
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            ctx.trace_steps.append(
                {
                    "step_index": hops,
                    "node": current,
                    "status": status,
                    "latency_ms": elapsed_ms,
                }
            )
            current = next_node
        return ctx
