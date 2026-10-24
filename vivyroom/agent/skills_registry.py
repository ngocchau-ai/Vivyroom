"""Customizable Skills Registry for `Vivyroom` (`<= 2 GB RAM`, Zero Paid API Keys).

Includes:
1. `websearch_zero_api` — Ultra-lightweight Zero-API WebSearch prioritizing **RoboForex** & **ForexFactory**
2. `github_huggingface_connector` — GitHub & HuggingFace repo/model downloader & Cautreo `.vcore` coupler
3. `youtube_learning_ingestor` — YouTube transcript & technical metadata ingestor for Bot-Trade/Quant-Trade learning
4. `market_news_synthesizer` — ForexFactory / RoboForex / RSS macro news synthesizer
5. `finance_moe_analyze` — Autonomous `Vivy-Finance-3B-MoE` quantitative analysis & Zero-Token SVG chart
6. `python_ast_math` — Safe AST mathematical expression evaluator
7. `workspace_list_files` & `workspace_read_file` — Local workspace file inspection
8. `trigger_lucid_dream` — 4-Stage `LUCID_STANDBY` Dream Consolidation & MoE RAM purge
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from ..config import REPO_ROOT
from ..moe_engine.cautreo_moe_pager import CautreoMoEDynamicPager
from ..moe_engine.finance_autonomous_desk import VivyFinanceMoEDesk
from .mandatory_trader_skills import (
    execute_github_huggingface_connector,
    execute_market_news_synthesizer,
    execute_websearch_zero_api,
    execute_youtube_learning_ingestor,
)

_SAFE_MATH_OPS: dict[type[Any], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_safe_ast_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_safe_ast_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_MATH_OPS:
        left = _eval_safe_ast_node(node.left)
        right = _eval_safe_ast_node(node.right)
        return float(_SAFE_MATH_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_MATH_OPS:
        return float(_SAFE_MATH_OPS[type(node.op)](_eval_safe_ast_node(node.operand)))
    raise ValueError(f"Unsupported AST math node: {type(node).__name__}")


class OfflineSkillsRegistry:
    """Manages customizable skills controlled directly by Vivy without external relay."""

    def __init__(self, finance_desk: VivyFinanceMoEDesk, moe_pager: CautreoMoEDynamicPager) -> None:
        self.finance_desk = finance_desk
        self.moe_pager = moe_pager
        self.last_macro_news_bias: float = 0.22
        self.enabled_skills: dict[str, bool] = {
            "websearch_zero_api": True,
            "github_huggingface_connector": True,
            "youtube_learning_ingestor": True,
            "market_news_synthesizer": True,
            "finance_moe_analyze": True,
            "python_ast_math": True,
            "workspace_list_files": True,
            "workspace_read_file": True,
            "trigger_lucid_dream": True,
        }

    def set_skill_enabled(self, skill_name: str, enabled: bool) -> dict[str, Any]:
        if skill_name in self.enabled_skills:
            self.enabled_skills[skill_name] = bool(enabled)
        return dict(self.enabled_skills)

    def toggle_skill(self, skill_name: str, enabled: bool) -> dict[str, Any]:
        return self.set_skill_enabled(skill_name, enabled)

    def execute_skill(self, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled_skills.get(tool_name, False):
            return {"executed": False, "reason": f"Skill '{tool_name}' is disabled in customization settings."}

        if tool_name == "websearch_zero_api":
            query = str(tool_args.get("query", "RoboForex ForexFactory XAUUSD quant algo"))
            return execute_websearch_zero_api(query=query)

        if tool_name == "github_huggingface_connector":
            target = str(tool_args.get("target", "ngocchau-ai/Vivyroom"))
            action = str(tool_args.get("action", "couple_and_download"))
            return execute_github_huggingface_connector(
                target=target,
                action=action,
                master_key=self.moe_pager.master_key,
            )

        if tool_name == "youtube_learning_ingestor":
            video_or_topic = str(tool_args.get("video_or_topic", "Quant Algo Bot Trading Kalman OU"))
            return execute_youtube_learning_ingestor(video_url_or_topic=video_or_topic)

        if tool_name == "market_news_synthesizer":
            sym = str(tool_args.get("symbol", "XAUUSD")).upper()
            news_res = execute_market_news_synthesizer(symbol=sym)
            self.last_macro_news_bias = float(news_res.get("macro_news_bias", 0.22))
            return news_res

        if tool_name == "finance_moe_analyze":
            sym = str(tool_args.get("symbol", "XAUUSD")).upper()
            res = self.finance_desk.analyze_market_offline_or_live(
                symbol=sym,
                macro_news_bias=self.last_macro_news_bias,
            )
            res["executed"] = True
            return res

        if tool_name == "python_ast_math":
            expr = str(tool_args.get("expression", "1+1")).replace("^", "**")
            tree = ast.parse(expr, mode="eval")
            val = _eval_safe_ast_node(tree)
            if math.isfinite(val) and val == int(val):
                val_out: int | float = int(val)
            else:
                val_out = round(val, 6)
            return {"executed": True, "expression": expr, "result": val_out}

        if tool_name == "workspace_list_files":
            sub = str(tool_args.get("subpath", "."))
            target = (REPO_ROOT / sub).resolve()
            if not str(target).startswith(str(REPO_ROOT.resolve())):
                target = REPO_ROOT
            items = sorted(p.name for p in target.iterdir() if not p.name.startswith(".")) if target.is_dir() else []
            return {"executed": True, "directory": str(target.name), "count": len(items), "files": items}

        if tool_name == "workspace_read_file":
            rel_path = str(tool_args.get("path", "README.md"))
            target = (REPO_ROOT / rel_path).resolve()
            if target.is_file() and str(target).startswith(str(REPO_ROOT.resolve())):
                content = target.read_text(encoding="utf-8", errors="replace")[:1024]
                return {"executed": True, "path": rel_path, "snippet": content}
            return {"executed": False, "reason": "File not found inside workspace."}

        if tool_name == "trigger_lucid_dream":
            rounds = int(tool_args.get("rounds", 2))
            purge_info = self.moe_pager.purge_all_expert_pages()
            return {
                "executed": True,
                "dream_status": "LUCID_STANDBY_COMPLETE",
                "rounds_completed": rounds,
                "evicted_experts": purge_info["evicted_experts"],
                "ram_after_mb": purge_info["process_ram_mb"],
            }

        return {"executed": False, "reason": f"Unknown skill: {tool_name}"}
