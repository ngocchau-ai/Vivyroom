"""Cautreo In-Process Context Memory, Score Graph & `VM-11` Falsification Tracker."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MemoryTurnRecord:
    turn_id: int
    timestamp: float
    user_input: str
    intent_domain: str
    active_experts: list[str]
    response_preview: str
    latency_ms: float
    ram_mb: float
    vm11_weight: float
    score_progress: float


class CautreoMemoryAndVM11Engine:
    """Bounded Ring-Buffer Context Memory (max 100 turns) & VM-11 Dampening Ledger."""

    def __init__(self, max_turns: int = 100) -> None:
        self.history: deque[MemoryTurnRecord] = deque(maxlen=max_turns)
        self.falsified_branches: dict[str, int] = {}
        self.quant_lessons: deque[dict[str, Any]] = deque(maxlen=50)
        self.coupled_external_models: list[dict[str, Any]] = []
        self.total_turns: int = 0
        self.dream_cycles_completed: int = 0

    def record_quant_lesson(self, lesson: dict[str, Any]) -> None:
        self.quant_lessons.append(lesson)

    def record_coupled_model(self, info: dict[str, Any]) -> None:
        self.coupled_external_models.append(info)

    def record_turn(
        self,
        turn_id: int,
        user_input: str,
        intent_domain: str,
        active_experts: list[str],
        response_text: str,
        latency_ms: float,
        ram_mb: float,
        vm11_weight: float = 1.0,
        score_progress: float = 0.98,
    ) -> MemoryTurnRecord:
        self.total_turns += 1
        rec = MemoryTurnRecord(
            turn_id=turn_id,
            timestamp=time.time(),
            user_input=user_input[:200],
            intent_domain=intent_domain,
            active_experts=list(active_experts),
            response_preview=response_text[:240],
            latency_ms=round(latency_ms, 3),
            ram_mb=round(ram_mb, 2),
            vm11_weight=round(vm11_weight, 4),
            score_progress=round(score_progress, 4),
        )
        self.history.append(rec)
        return rec

    def register_falsification(self, branch_key: str) -> float:
        """Applies VM-11 dampening (`0.5^k`) to a falsified branch."""
        k = self.falsified_branches.get(branch_key, 0) + 1
        self.falsified_branches[branch_key] = k
        return round(0.5 ** k, 4)

    def get_vm11_weight(self, branch_key: str) -> float:
        k = self.falsified_branches.get(branch_key, 0)
        return round(0.5 ** k, 4)

    def get_digest(self) -> dict[str, Any]:
        recent = [asdict(r) for r in list(self.history)[-10:]]
        mean_lat = (
            round(sum(r.latency_ms for r in self.history) / len(self.history), 3)
            if self.history
            else 0.0
        )
        return {
            "total_turns": self.total_turns,
            "stored_turns_in_ring_buffer": len(self.history),
            "mean_turn_latency_ms": mean_lat,
            "falsified_branches_count": len(self.falsified_branches),
            "dream_cycles_completed": self.dream_cycles_completed,
            "quant_lessons_count": len(self.quant_lessons),
            "recent_quant_lessons": list(self.quant_lessons)[-5:],
            "coupled_external_models": list(self.coupled_external_models)[-5:],
            "recent_turns": recent,
        }
