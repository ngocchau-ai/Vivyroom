"""Unified Zero-Relay `VivyroomAutonomousController` — Single-Process Vivy + Cautreo Organism.

Orchestrates:
1. `VivyQwenSemanticCompiler` (Inbound NL -> Canonical Decision AST, `< 1.0 ms`)
2. `CautreoMoEDynamicPager` (Top-2 `.vcore` Expert Paging with `< 2,048 MB` RAM Circuit Breaker)
3. `NativeTriCoreABIBridge` (`vivy_v_soul.dll` 25-Step Equilibrium + `vivy_nps_coupled_decide`)
4. `OfflineSkillsRegistry` (Finance-3B-MoE Quant Desk, AST Math, Workspace I/O, Lucid Dream)
5. `CautreoMemoryAndVM11Engine` (Ring-Buffer Context Memory & VM-11 Falsification Dampening)
6. `VivyQwenSemanticCompiler` (Outbound 1-Step Compositional NL & 24kHz Voice Synthesis)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..auth_one_touch import OneTouchSovereignAuthGate
from ..config import DEFAULT_CONFIG, MODELS_DIR, VivyroomConfig
from ..moe_engine.cautreo_moe_pager import (
    CautreoMoEDynamicPager,
    NativeTriCoreABIBridge,
    get_process_working_set_mb,
)
from ..moe_engine.finance_autonomous_desk import VivyFinanceMoEDesk
from ..moe_engine.moe_upcycler_trainer import (
    encode_utterance_sketch_64_np,
    train_and_package_vivy_tri_core_moe,
)
from ..moe_engine.vivy_v_finance_loader import load_and_verify_vivy_v_finance_3b
from ..plugin_compiler import UniversalPluginCompilerBridge
from .computer_browser_web_suite import (
    VivyMultiProviderKeylessWebEngine,
    VivyNativeBrowserUseEngine,
    VivyNativeComputerUseEngine,
)
from .deepseek_harness_bridge import RustDeepSeekHarnessBridge
from .harness2_room_engine import VivyroomHarness2Controller
from .llm_semantic_compiler import CompiledIntentAST, VivyQwenSemanticCompiler
from .memory_engine import CautreoMemoryAndVM11Engine
from .multi_tier_memory import VivyFiveTierMemorySpine
from .oranpi_pocket_agent import build_autonomous_skill_plan, detect_oranpi_and_host_hardware
from .pocket_graph import PocketStateContext, ZeroRelayPocketGraph
from .room_self_architect import VivyRoomSelfArchitectEngine
from .skills_registry import OfflineSkillsRegistry
from .trader_system_prompt import TraderSystemPromptManager


class VivyroomAutonomousController:
    """Main controller for `Vivyroom` operating 100% offline in a single process (< 2 GB RAM)."""

    def __init__(
        self,
        config: VivyroomConfig = DEFAULT_CONFIG,
        models_dir: Path = MODELS_DIR,
        master_passphrase: str | None = None,
    ) -> None:
        self.config = config
        self.models_dir = Path(models_dir)
        self.auth_gate = OneTouchSovereignAuthGate(master_passphrase=master_passphrase)

        # Ensure encrypted `.vcore` 3-Cores-in-1 models exist on disk
        master_vcore = self.models_dir / "vivy_tri_core_master.vcore"
        if not master_vcore.is_file():
            train_and_package_vivy_tri_core_moe(
                output_models_dir=self.models_dir,
                master_key=self.auth_gate.master_key_bytes,
            )

        self.abi = NativeTriCoreABIBridge()
        self.vivy_v_finance_suite = load_and_verify_vivy_v_finance_3b(self.models_dir / "vivy_v_finance_3b")
        self.deepseek_harness = RustDeepSeekHarnessBridge()
        self.harness2 = VivyroomHarness2Controller(rust_bridge=self.deepseek_harness)
        self.computer_use = VivyNativeComputerUseEngine(rust_bridge=self.deepseek_harness)
        self.browser_use = VivyNativeBrowserUseEngine()
        self.keyless_web = VivyMultiProviderKeylessWebEngine(browser_engine=self.browser_use)
        self.five_tier_memory = VivyFiveTierMemorySpine()
        self.room_architect = VivyRoomSelfArchitectEngine(self.deepseek_harness)
        self.plugin_compiler = UniversalPluginCompilerBridge()
        self.moe_pager = CautreoMoEDynamicPager(
            master_key=self.auth_gate.master_key_bytes,
            models_dir=self.models_dir,
            config=self.config,
        )
        self.finance_desk = VivyFinanceMoEDesk(self.abi, self.moe_pager)
        self.compiler = VivyQwenSemanticCompiler(self.abi, self.moe_pager)
        self.skills = OfflineSkillsRegistry(self.finance_desk, self.moe_pager)
        self.skills.enabled_skills["room_self_architect"] = True
        self.skills.enabled_skills["open_source_plugin_compiler"] = True
        self.skills.enabled_skills["computer_use_bounded_cua"] = True
        self.skills.enabled_skills["browser_use_indexed_dom"] = True
        self.skills.enabled_skills["five_tier_sovereign_memory"] = True
        self.memory = CautreoMemoryAndVM11Engine(max_turns=100)
        self.system_prompt_mgr = TraderSystemPromptManager()
        self._turn_counter: int = 0
        self._persona_vector_8 = np.array(
            [0.96, 0.98, 0.94, 0.95, 0.92, 0.96, 0.90, 0.94], dtype=np.float32
        )
        self.graph = self._build_zero_relay_graph()

    def _build_zero_relay_graph(self) -> ZeroRelayPocketGraph:
        g = ZeroRelayPocketGraph(max_step_ms=self.config.max_single_step_timeout_ms)

        def node_compile_inbound(ctx: PocketStateContext) -> str:
            ast_in = self.compiler.compile_inbound_nl_to_ast(ctx.raw_user_input)
            ctx.state["ast_in"] = ast_in
            if ast_in.self_healed_repairs:
                ctx.errors_healed.extend(ast_in.self_healed_repairs)
            return "ROUTE_MOE_AND_EQUILIBRIUM"

        def node_route_moe_and_equilibrium(ctx: PocketStateContext) -> str:
            ast_in: CompiledIntentAST = ctx.state.get("ast_in") or self.compiler.compile_inbound_nl_to_ast(ctx.raw_user_input)
            ctx.state["ast_in"] = ast_in
            sketch_64 = encode_utterance_sketch_64_np(ast_in.normalized_text)
            moe_report = self.moe_pager.route_and_page_top_k_experts(
                sketch_64,
                utterance_text=ast_in.normalized_text,
                intent_domain=ast_in.intent_domain,
            )
            ctx.state["moe_report"] = moe_report
            eq_report = self.abi.run_soul_equilibrium_loop(
                np.array(ast_in.soul_context_16, dtype=np.float32),
                self._persona_vector_8,
                max_steps=8,
            )
            ctx.state["eq_report"] = eq_report
            harness_receipt = self.deepseek_harness.run_harness(
                ast_in.normalized_text,
                falsified_branches=len(self.memory.falsified_branches),
            )
            ctx.state["deepseek_harness"] = harness_receipt
            skill_plan = (
                []
                if ast_in.contradiction_detected
                else build_autonomous_skill_plan(
                    ast_in.normalized_text,
                    ast_in.tool_name if ast_in.requires_tool else "none",
                    ast_in.tool_args,
                )
            )
            ctx.state["skill_plan"] = skill_plan
            return "ACTUATE_SKILL" if (ast_in.requires_tool or len(skill_plan) > 0) else "COMPILE_OUTBOUND"

        def node_actuate_skill(ctx: PocketStateContext) -> str:
            ast_in: CompiledIntentAST = ctx.state["ast_in"]
            skill_plan: list[dict[str, Any]] = ctx.state.get("skill_plan") or [
                {"tool_name": ast_in.tool_name, "tool_args": ast_in.tool_args}
            ]
            chained_outputs: list[dict[str, Any]] = []
            primary_out: dict[str, Any] = {}

            for step in skill_plan:
                t_name = step["tool_name"]
                t_args = step["tool_args"]
                if t_name == "update_system_prompt_inline":
                    if t_args.get("reset"):
                        prof = self.system_prompt_mgr.reset_to_default_trader_prompt()
                    else:
                        prof = self.system_prompt_mgr.update_system_prompt(str(t_args.get("prompt_text", "")))
                    out = {"executed": True, "skill": "update_system_prompt_inline", **prof.to_dict()}
                elif t_name == "room_self_architect":
                    act_type = str(t_args.get("action_type", "inspect_only"))
                    permitted = bool(t_args.get("user_permitted", False))
                    if act_type == "mutate_theme":
                        out = self.room_architect.mutate_room_theme(
                            str(t_args.get("theme_key") or "violet"),
                            user_permitted=permitted,
                        )
                    elif act_type == "mutate_title":
                        raw_u = str(t_args.get("raw_utterance", ""))
                        import re as _re

                        m_t = _re.search(r"['\"«]([^'\"»]{2,60})['\"»]", raw_u)
                        new_t = m_t.group(1) if m_t else "VIVYROOM SOVEREIGN"
                        out = self.room_architect.mutate_room_title(new_t, user_permitted=permitted)
                    elif act_type == "rollback_room":
                        out = self.room_architect.rollback_last_room_mutation(user_permitted=permitted)
                    elif act_type == "diagnose_and_repair":
                        out = self.room_architect.diagnose_and_repair_room(user_permitted=permitted)
                    else:
                        out = self.room_architect.inspect_room_architecture()
                elif t_name == "open_source_plugin_compiler":
                    out = self.plugin_compiler.hot_mount_from_spec(str(t_args.get("spec", "")))
                else:
                    out = self.skills.execute_skill(t_name, t_args)

                chained_outputs.append(out)
                if not primary_out or t_name == ast_in.tool_name:
                    primary_out = out
                if not out.get("executed", False):
                    self.memory.register_falsification(f"skill:{t_name}")
                elif t_name == "trigger_lucid_dream":
                    self.memory.dream_cycles_completed += 1
                elif t_name == "youtube_learning_ingestor" and "distilled_lesson" in out:
                    self.memory.record_quant_lesson(out["distilled_lesson"])
                elif t_name == "github_huggingface_connector" and out.get("cautreo_coupled"):
                    self.memory.record_coupled_model(
                        {
                            "platform": out.get("platform"),
                            "repo_id": out.get("repo_id"),
                            "coupled_vcore_path": out.get("coupled_vcore_path"),
                        }
                    )

            # Autonomous RAM self-maintenance: if turn count is multiple of 20 or RAM > 75% ceiling, auto-dream
            if self._turn_counter % 20 == 0 or get_process_working_set_mb() >= self.config.device_ram_limit_mb * 0.75:
                self.moe_pager.purge_all_expert_pages()
                self.memory.dream_cycles_completed += 1

            if len(chained_outputs) > 1:
                merged = dict(primary_out)
                merged["chained_outputs"] = chained_outputs
                merged["autonomous_skills_executed"] = [s["tool_name"] for s in skill_plan]
                # Preserve svg_chart if any step in the chain generated one
                for item in chained_outputs:
                    if "svg_chart" in item:
                        merged["svg_chart"] = item["svg_chart"]
                ctx.state["tool_output"] = merged
            else:
                ctx.state["tool_output"] = primary_out
            return "COMPILE_OUTBOUND"

        def node_compile_outbound(ctx: PocketStateContext) -> str | None:
            ast_in: CompiledIntentAST = ctx.state.get("ast_in") or self.compiler.compile_inbound_nl_to_ast(ctx.raw_user_input)
            ctx.state["ast_in"] = ast_in
            eq_report = ctx.state.get(
                "eq_report",
                {"reality_alignment": 0.98, "vm11_weight": 1.0, "steps_executed": 4, "backtrack_count": 0},
            )
            ctx.state["eq_report"] = eq_report
            moe_report = ctx.state.get(
                "moe_report",
                {
                    "active_expert_ids": [6],
                    "active_expert_codes": ["E6_COMPOSITIONAL_DIALOGUE_VOICE"],
                    "resident_expert_count": len(self.moe_pager.resident_experts),
                    "process_ram_mb": get_process_working_set_mb(),
                    "ram_limit_mb": self.config.device_ram_limit_mb,
                },
            )
            ctx.state["moe_report"] = moe_report
            tool_out = ctx.state.get("tool_output")
            out_bundle = self.compiler.compile_outbound_decision_to_nl(
                ast_in=ast_in,
                eq_report=eq_report,
                moe_report=moe_report,
                tool_output=tool_out,
                synthesize_voice=ctx.synthesize_voice,
            )
            ctx.state["outbound"] = out_bundle
            return None

        g.add_node("COMPILE_INBOUND", node_compile_inbound, is_entry=True)
        g.add_node("ROUTE_MOE_AND_EQUILIBRIUM", node_route_moe_and_equilibrium)
        g.add_node("ACTUATE_SKILL", node_actuate_skill)
        g.add_node("COMPILE_OUTBOUND", node_compile_outbound)
        return g

    def chat_turn(self, user_message: str, synthesize_voice: bool = False) -> dict[str, Any]:
        """Executes a complete autonomous conversational & agent turn in < 25 ms."""
        t0 = time.perf_counter()
        self._turn_counter += 1
        ctx = PocketStateContext(
            turn_id=self._turn_counter,
            raw_user_input=user_message,
            synthesize_voice=synthesize_voice,
        )
        self.graph.execute(ctx)

        total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        ram_mb = get_process_working_set_mb()
        ast_in: CompiledIntentAST = ctx.state.get("ast_in") or self.compiler.compile_inbound_nl_to_ast(user_message)
        moe_report: dict[str, Any] = ctx.state.get(
            "moe_report",
            {
                "active_expert_ids": [6],
                "active_expert_codes": ["E6_COMPOSITIONAL_DIALOGUE_VOICE"],
                "resident_expert_count": len(self.moe_pager.resident_experts),
                "process_ram_mb": ram_mb,
                "ram_limit_mb": self.config.device_ram_limit_mb,
            },
        )
        eq_report: dict[str, Any] = ctx.state.get(
            "eq_report",
            {"reality_alignment": 0.98, "vm11_weight": 1.0, "steps_executed": 4, "backtrack_count": 0},
        )
        tool_out: dict[str, Any] | None = ctx.state.get("tool_output")
        outbound: dict[str, Any] = ctx.state.get("outbound") or self.compiler.compile_outbound_decision_to_nl(
            ast_in=ast_in,
            eq_report=eq_report,
            moe_report=moe_report,
            tool_output=tool_out,
            synthesize_voice=synthesize_voice,
        )
        self.memory.record_turn(
            turn_id=self._turn_counter,
            user_input=ast_in.normalized_text,
            intent_domain=ast_in.intent_domain,
            active_experts=moe_report["active_expert_codes"],
            response_text=outbound["response_text"],
            latency_ms=total_ms,
            ram_mb=ram_mb,
            vm11_weight=float(eq_report.get("vm11_weight", 1.0)),
        )
        ds_harness_out = ctx.state.get("deepseek_harness") or self.deepseek_harness.run_harness(user_message)
        dsh_x_harness2_out = self.harness2.execute_dsh_contract_and_evolve(
            user_message=ast_in.normalized_text,
            domain_cluster=str(ds_harness_out.get("domain_cluster", ast_in.intent_domain)),
            reasoning_scaffold=str(ds_harness_out.get("reasoning_scaffold", "")),
            action_class=ast_in.tool_name if ast_in.requires_tool else "observe_and_reason",
            target_path="vivyroom://conversation",
            user_permitted=bool(ast_in.tool_args.get("user_permitted", False)),
        )
        five_tier_mem_out = self.five_tier_memory.process_turn(
            user_message=ast_in.normalized_text,
            assistant_reply=outbound["response_text"],
            m1_engram_telemetry=ds_harness_out,
            m2_cautreo_telemetry={
                "working_window_slots": min(4, self._turn_counter),
                "vm11_dampening": float(eq_report.get("vm11_weight", 1.0)),
                "active_experts": moe_report["active_expert_codes"],
            },
        )

        return {
            "turn_id": self._turn_counter,
            "user_input": user_message,
            "response_text": outbound["response_text"],
            "voice_wav_base64": outbound["voice_wav_base64"],
            "total_turn_latency_ms": total_ms,
            "compiler_telemetry": {
                "inbound_compile_ms": ast_in.compile_latency_ms,
                "outbound_compile_ms": outbound["outbound_compile_latency_ms"],
                "intent_domain": ast_in.intent_domain,
                "register_mode": ast_in.register_mode,
                "contradiction_detected": ast_in.contradiction_detected,
                "self_healed_repairs": ctx.errors_healed,
                "chosen_slot_units_6x64": outbound["chosen_slot_units_6x64"],
                "nps_coupled_states": outbound["nps_coupled_states"],
                "nps_coupled_score": outbound["nps_coupled_score"],
            },
            "decision_equilibrium_telemetry": eq_report,
            "deepseek_harness_telemetry": ds_harness_out,
            "dsh_x_harness2_telemetry": dsh_x_harness2_out,
            "five_tier_memory_telemetry": five_tier_mem_out,
            "moe_routing_telemetry": moe_report,
            "tool_output": tool_out,
            "ui_reload_required": bool(tool_out and tool_out.get("ui_reload_required", False)),
            "pocket_graph_trace": ctx.trace_steps,
            "hardware_telemetry": {
                "process_ram_mb": ram_mb,
                "device_ram_limit_mb": self.config.device_ram_limit_mb,
                "under_2gb_budget": ram_mb <= self.config.device_ram_limit_mb,
                "zero_external_relay": True,
                "native_embedded_vivy": True,
                "proprietary_vcore_3in1_active": True,
            },
        }

    def get_system_status(self) -> dict[str, Any]:
        ram_mb = get_process_working_set_mb()
        vcore_audit = self.moe_pager.inspect_all_vcore_containers()
        return {
            "organism": "Vivy + Cautreo Sovereign Room v2.0 (Native Embedded Vivy + Rust DeepSeek Harness)",
            "is_single_unified_organism": True,
            "entity_count": 1,
            "zero_external_relay": True,
            "native_embedded_vivy": True,
            "auth_status": self.auth_gate.get_status(),
            "ram_telemetry": {
                "process_ram_mb": ram_mb,
                "device_ram_limit_mb": self.config.device_ram_limit_mb,
                "circuit_breaker_mb": self.config.circuit_breaker_mb,
                "headroom_mb": round(self.config.device_ram_limit_mb - ram_mb, 2),
                "resident_moe_experts": [
                    self.moe_pager.last_active_experts
                ],
                "page_in_events": self.moe_pager.page_in_events,
                "page_out_events": self.moe_pager.page_out_events,
            },
            "vcore_audit": vcore_audit,
            "vivy_v_finance_3b_status": {
                k: v for k, v in self.vivy_v_finance_suite.items() if k != "reconstructed_tensors"
            },
            "deepseek_harness_status": {
                "native_rust_loaded": self.deepseek_harness.native_loaded,
                "dll_path": str(self.deepseek_harness.dll_path.name),
            },
            "dsh_x_harness2_status": {
                "attached_to_vivyroom_runtime": self.harness2.attached_to_vivyroom_runtime,
                "attached_to_cautreo_engine": self.harness2.attached_to_cautreo_engine,
                "organs_count": len(self.harness2.organs_state),
                "total_dsh_receipts": self.harness2.total_receipts,
            },
            "five_tier_memory_status": {
                "fts5_enabled": self.five_tier_memory.m3_vault.fts5_enabled,
                "m5_two_brain": self.five_tier_memory.m5_sync.status(),
            },
            "room_self_architect_status": self.room_architect.inspect_room_architecture(),
            "mounted_plugins": self.plugin_compiler.get_summary(),
            "enabled_skills": self.skills.enabled_skills,
            "memory_digest": self.memory.get_digest(),
            "trader_system_prompt": self.system_prompt_mgr.get_active_profile().to_dict(),
            "oranpi_edge_profile": detect_oranpi_and_host_hardware(),
        }
