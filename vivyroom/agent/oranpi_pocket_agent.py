"""Open-Source `OranPi × PocketFlow` Autonomous Edge Agent Engine for `Vivyroom`.

Combines:
1. **PocketFlow (`The-Pocket/PocketFlow`, MIT License):** Ultra-lightweight `Node` -> `Action` -> `Flow`
   graph abstraction (`prep -> exec -> post`) with zero third-party bloat.
2. **OranPi Edge Agent Runtime:** Hardware-aware autonomous agent overlay tailored for
   OrangePi (`aarch64` / `armv7l` 1GB–2GB RAM SBCs), Linux (`x86_64`), and Windows (`AMD64`).
3. **100% Autonomous Skill Chaining (Zero Buttons Required):**
   - Users interact exclusively through a single clean chat box (and optional `SYSTEM_PROMPT.md` edit).
   - Vivy autonomously plans and chains multiple skills in a single conversational turn
     (e.g., `market_news_synthesizer` -> `websearch_zero_api` -> `finance_moe_analyze`, or
     `youtube_learning_ingestor` -> `finance_moe_analyze`, or live `SYSTEM_PROMPT.md` update via chat)
     and autonomously runs `LUCID_STANDBY` Dream RAM consolidation when needed.
"""

from __future__ import annotations

import os
import platform
import re
import time
from pathlib import Path
from typing import Any

from ..moe_engine.cautreo_moe_pager import get_process_working_set_mb


class PocketFlowNode:
    """Canonical MIT PocketFlow Node (`prep -> exec -> post`) with self-healing guard."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.successors: dict[str, PocketFlowNode] = {}

    def connect(self, action: str, next_node: PocketFlowNode) -> PocketFlowNode:
        self.successors[action] = next_node
        return next_node

    def prep(self, shared: dict[str, Any]) -> Any:
        return shared

    def exec(self, prep_res: Any) -> Any:
        return prep_res

    def post(self, shared: dict[str, Any], prep_res: Any, exec_res: Any) -> str | None:
        return "default"

    def run_step(self, shared: dict[str, Any]) -> str | None:
        p = self.prep(shared)
        e = self.exec(p)
        return self.post(shared, p, e)


class OranPiPocketFlow:
    """Zero-relay finite-state flow runner executing `PocketFlowNode` chains in-process."""

    def __init__(self, start_node: PocketFlowNode, max_hops: int = 10) -> None:
        self.start_node = start_node
        self.max_hops = max_hops

    def run(self, shared: dict[str, Any]) -> dict[str, Any]:
        curr: PocketFlowNode | None = self.start_node
        hops = 0
        trace: list[dict[str, Any]] = shared.setdefault("trace_steps", [])
        errors: list[str] = shared.setdefault("errors_healed", [])

        while curr is not None and hops < self.max_hops:
            hops += 1
            t0 = time.perf_counter()
            try:
                action = curr.run_step(shared)
                status = "OK"
            except Exception as exc:
                errors.append(f"oranpi_flow_healed:{curr.name}:{type(exc).__name__}")
                action = "compile_outbound" if curr.name != "COMPILE_OUTBOUND" else None
                status = "SELF_HEALED"
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            trace.append(
                {
                    "step_index": hops,
                    "node": curr.name,
                    "action": action or "END",
                    "status": status,
                    "latency_ms": elapsed_ms,
                }
            )
            if action is None:
                break
            curr = curr.successors.get(action)
        return shared


def detect_oranpi_and_host_hardware() -> dict[str, Any]:
    """Detects Windows, Linux x86_64, or OrangePi / ARM SBC hardware profile."""
    sys_name = platform.system()
    machine = platform.machine()
    board_model = f"{sys_name} {machine}"
    is_orangepi_or_sbc = False

    dt_model = Path("/proc/device-tree/model")
    if dt_model.is_file():
        try:
            raw_model = dt_model.read_text(encoding="utf-8", errors="ignore").strip("\x00 \n")
            if raw_model:
                board_model = raw_model
                if any(k in raw_model.lower() for k in ("orangepi", "orange pi", "rockchip", "allwinner", "raspberry")):
                    is_orangepi_or_sbc = True
        except Exception:
            pass
    elif machine.lower() in ("aarch64", "armv7l", "arm64"):
        is_orangepi_or_sbc = True

    return {
        "agent_framework": "OranPi-PocketFlow-v1.2 (MIT Open-Source Edge Agent)",
        "os_family": sys_name,
        "arch": machine,
        "board_model": board_model,
        "is_orangepi_or_arm_sbc": is_orangepi_or_sbc,
        "cpu_cores": os.cpu_count() or 2,
        "process_ram_mb": get_process_working_set_mb(),
    }


def build_autonomous_skill_plan(normalized_text: str, primary_tool: str, primary_args: dict[str, Any]) -> list[dict[str, Any]]:
    """Autonomously constructs a multi-skill execution plan from a single natural-language chat message.

    No manual UI buttons are needed: if a user asks a compound question (e.g., macro news + RoboForex search
    + XAUUSD chart analysis), Vivy chains all relevant skills automatically in one turn.
    """
    lower = normalized_text.lower()
    plan: list[dict[str, Any]] = []
    seen_tools: set[str] = set()

    def _add(tool_name: str, args: dict[str, Any]) -> None:
        if tool_name and tool_name != "none" and tool_name not in seen_tools:
            seen_tools.add(tool_name)
            plan.append({"tool_name": tool_name, "tool_args": args})

    # Detect symbol if any
    sym = str(primary_args.get("symbol", "XAUUSD")).upper()
    if "eur" in lower or "euusd" in lower:
        sym = "EURUSD"
    elif "jpy" in lower or "jpgusd" in lower:
        sym = "USDJPY"
    elif "btc" in lower:
        sym = "BTCUSD"

    # 1. Check if the user is updating SYSTEM_PROMPT.md directly via chat!
    m_prompt = re.match(
        r"^(?:/prompt|system\s*prompt\s*:|đổi\s+system\s*prompt\s*:|cập\s+nhật\s+system\s*prompt\s*:|set\s+prompt\s*:)\s*(.+)$",
        normalized_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_prompt:
        _add("update_system_prompt_inline", {"prompt_text": m_prompt.group(1).strip()})
        return plan

    if lower.strip() in ("/reset-prompt", "reset system prompt", "khôi phục system prompt gốc"):
        _add("update_system_prompt_inline", {"reset": True})
        return plan

    # 2. Autonomous Multi-Skill Discovery from natural language intent
    wants_news = any(
        k in lower
        for k in ("tin tức", "tổng hợp tin", "lịch kinh tế", "cpi", "nfp", "fomc", "market news", "tin vĩ mô", "điểm tin")
    )
    wants_web = any(
        k in lower
        for k in ("websearch", "tìm kiếm", "tra cứu", "roboforex", "forexfactory", "copyfx", "spread", "search web")
    )
    wants_yt = any(
        k in lower
        for k in ("youtube", "youtu.be", "xem video", "học từ youtube", "video bài giảng", "youtube learning")
    )
    wants_gh_hf = any(
        k in lower
        for k in ("github", "huggingface", "tải repo", "tải model", "ghép nối model", "clone repo", "hf.co")
    )
    has_other_skill = wants_news or wants_web or wants_yt or wants_gh_hf
    if has_other_skill:
        wants_finance = any(
            k in lower
            for k in ("phân tích", "biểu đồ", "chart", "lãi kép", "50 bot", "kèo")
        )
    else:
        wants_finance = any(
            k in lower
            for k in (
                "xauusd",
                "vàng",
                "gold",
                "eurusd",
                "euusd",
                "usdjpy",
                "jpgusd",
                "btcusd",
                "lãi kép",
                "50 bot",
                "kalman",
                "ofi",
                "vpin",
                "biểu đồ",
                "chart",
                "phân tích giá",
                "kèo",
            )
        )

    if wants_news:
        _add("market_news_synthesizer", {"symbol": sym})
    if wants_web:
        _add("websearch_zero_api", {"query": normalized_text})
    if wants_yt:
        _add("youtube_learning_ingestor", {"video_or_topic": normalized_text})
    if wants_gh_hf:
        _add("github_huggingface_connector", {"target": normalized_text, "action": "couple_and_download"})
    if wants_finance:
        _add("finance_moe_analyze", {"symbol": sym})

    # Always honor primary_tool if not yet in plan
    if primary_tool and primary_tool != "none":
        _add(primary_tool, primary_args)

    return plan
