"""Native C-ABI Bridge to `vivyroom_harness.dll` (Rust DeepSeek Harness + Engram O(1) + Room Mutation Ledger).

Connects Python directly to `crates/vivyroom-rs` (`vivyroom_harness.dll` / `libvivyroom_harness.so`)
with portable `x86-64` execution and automatic pure-Python fallback if running on an OS without a prebuilt binary.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..config import NATIVE_BIN_DIR


class RustDeepSeekHarnessBridge:
    """Binds `vivyroom_harness.dll` for DeepSeek-style `Propose -> Probe -> Compose` + `MTP` + `Engram O(1)`."""

    def __init__(self, bin_dir: Path = NATIVE_BIN_DIR) -> None:
        self.bin_dir = Path(bin_dir)
        self.dll_path = self.bin_dir / "vivyroom_harness.dll"
        self._dll: Any = None
        self.native_loaded: bool = False
        self._fallback_call_id: int = 0
        self._fallback_mutation_count: int = 0
        self._fallback_chain_hash: int = 0xCBF29CE484222325
        self._bind_dll()

    def _bind_dll(self) -> None:
        if not self.dll_path.is_file():
            return
        try:
            dll = ctypes.CDLL(str(self.dll_path))
            dll.vh_abi_version.argtypes = []
            dll.vh_abi_version.restype = ctypes.c_uint32

            dll.vh_engram_lookup.argtypes = [
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
            ]
            dll.vh_engram_lookup.restype = ctypes.c_int32

            dll.vh_run_deepseek_harness.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
            ]
            dll.vh_run_deepseek_harness.restype = ctypes.c_int32

            dll.vh_record_room_mutation.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_uint64),
            ]
            dll.vh_record_room_mutation.restype = ctypes.c_int32

            if hasattr(dll, "vh_dsh_evaluate_gate4"):
                dll.vh_dsh_evaluate_gate4.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_size_t,
                ]
                dll.vh_dsh_evaluate_gate4.restype = ctypes.c_int32

            ver = int(dll.vh_abi_version())
            if ver == 0x0002_0000:
                self._dll = dll
                self.native_loaded = True
        except Exception:
            self._dll = None
            self.native_loaded = False

    def evaluate_dsh_gate4(
        self,
        action_class: str,
        target_path: str,
        user_permitted: bool = False,
    ) -> dict[str, Any]:
        """Evaluates DSH Gate 4 permission & allowlist boundaries in Rust (`vh_dsh_evaluate_gate4`)."""
        action = action_class.lower()
        target = target_path.lower()
        forbidden = ("rm -rf", "format ", "del /f /s /q", ".vcore", "system32", "credential", "private_key")
        is_forbidden = any(f in target or f in action for f in forbidden)
        read_only = (
            "read",
            "inspect",
            "search",
            "websearch",
            "browser_read",
            "memory_recall",
            "computer_observe",
            "skill_discover",
            "observe_and_reason",
        )
        if is_forbidden:
            risk_level = 99
            op_tag = f"rm -rf /:{action_class}:{target_path}"
        elif any(r in action for r in read_only):
            risk_level = 0
            op_tag = f"{action_class}:{target_path}"
        else:
            risk_level = 2
            op_tag = f"{action_class}:{target_path}"

        if self.native_loaded and self._dll is not None and hasattr(self._dll, "vh_dsh_evaluate_gate4"):
            try:
                buf = (ctypes.c_uint8 * 2048)()
                n = int(
                    self._dll.vh_dsh_evaluate_gate4(
                        op_tag.encode("utf-8"),
                        int(risk_level),
                        1 if user_permitted else 0,
                        buf,
                        2048,
                    )
                )
                if n > 0:
                    out = json.loads(bytes(buf[:n]).decode("utf-8"))
                    if is_forbidden:
                        out["risk_tier"] = "R3_FORBIDDEN"
                    elif risk_level == 2 and not user_permitted:
                        out["risk_tier"] = "R2_PERMISSION_REQUIRED"
                    out["native_rust_executed"] = True
                    return out
            except Exception:
                pass
        return self._fallback_evaluate_dsh_gate4(action_class, target_path, user_permitted)

    def _fallback_evaluate_dsh_gate4(
        self,
        action_class: str,
        target_path: str,
        user_permitted: bool,
    ) -> dict[str, Any]:
        action = action_class.lower()
        target = target_path.lower()
        forbidden = ("rm -rf", "format ", "del /f /s /q", ".vcore", "system32", "credential", "private_key")
        if any(f in target for f in forbidden):
            return {
                "gate": "DSH_GATE_4",
                "allowed": False,
                "risk_tier": "R3_FORBIDDEN",
                "action_class": action_class,
                "reason": "Blocked by DSH Gate 4: destructive or protected asset",
                "native_rust_executed": False,
            }
        read_only = (
            "read",
            "inspect",
            "search",
            "websearch",
            "browser_read",
            "memory_recall",
            "computer_observe",
            "skill_discover",
        )
        if any(r in action for r in read_only):
            return {
                "gate": "DSH_GATE_4",
                "allowed": True,
                "risk_tier": "R0_OBSERVE",
                "action_class": action_class,
                "reason": "Allowed by DSH Gate 4 (R0 read/observe)",
                "native_rust_executed": False,
            }
        if user_permitted:
            return {
                "gate": "DSH_GATE_4",
                "allowed": True,
                "risk_tier": "R1_PERMITTED_MUTATION",
                "action_class": action_class,
                "reason": "Allowed by DSH Gate 4 with explicit user permission",
                "native_rust_executed": False,
            }
        return {
            "gate": "DSH_GATE_4",
            "allowed": False,
            "risk_tier": "R2_PERMISSION_REQUIRED",
            "action_class": action_class,
            "reason": "Blocked by DSH Gate 4: requires explicit user permission",
            "native_rust_executed": False,
        }

    def engram_lookup(self, query: str) -> dict[str, Any]:
        """Executes O(1) conditional N-gram memory lookup in Rust (`vh_engram_lookup`)."""
        if self.native_loaded and self._dll is not None:
            try:
                buf = (ctypes.c_uint8 * 4096)()
                n = int(
                    self._dll.vh_engram_lookup(
                        query.encode("utf-8"),
                        buf,
                        4096,
                    )
                )
                if n > 0:
                    raw = bytes(buf[:n]).decode("utf-8")
                    out = json.loads(raw)
                    out["native_rust_executed"] = True
                    return out
            except Exception:
                pass
        return self._fallback_engram_lookup(query)

    def run_harness(self, query: str, falsified_branches: int = 0) -> dict[str, Any]:
        """Executes the DeepSeek `Propose -> Probe -> Compose` + `MTP` + `Engram O(1)` loop in Rust."""
        if self.native_loaded and self._dll is not None:
            try:
                buf = (ctypes.c_uint8 * 8192)()
                n = int(
                    self._dll.vh_run_deepseek_harness(
                        query.encode("utf-8"),
                        int(max(0, falsified_branches)),
                        buf,
                        8192,
                    )
                )
                if n > 0:
                    raw = bytes(buf[:n]).decode("utf-8")
                    out = json.loads(raw)
                    out["native_rust_executed"] = True
                    return out
            except Exception:
                pass
        return self._fallback_run_harness(query, falsified_branches)

    def record_room_mutation(self, mutation_tag: str, user_permitted: bool) -> dict[str, Any]:
        """Records a room self-modification event in the Rust cryptographic ledger (gated by `user_permitted`)."""
        if self.native_loaded and self._dll is not None:
            try:
                out_hash = ctypes.c_uint64(0)
                rc = int(
                    self._dll.vh_record_room_mutation(
                        mutation_tag.encode("utf-8"),
                        1 if user_permitted else 0,
                        ctypes.byref(out_hash),
                    )
                )
                if rc == -10:
                    return {
                        "recorded": False,
                        "error_code": -10,
                        "reason": "Permission denied by Rust Sovereign Gate (user_permitted=False)",
                        "native_rust_executed": True,
                    }
                if rc == 0:
                    return {
                        "recorded": True,
                        "mutation_chain_hash_hex": f"{out_hash.value:016x}",
                        "native_rust_executed": True,
                    }
            except Exception:
                pass

        if not user_permitted:
            return {
                "recorded": False,
                "error_code": -10,
                "reason": "Permission denied by Sovereign Gate (user_permitted=False)",
                "native_rust_executed": False,
            }
        h = int.from_bytes(hashlib.sha256(mutation_tag.encode("utf-8")).digest()[:8], "big")
        self._fallback_chain_hash = (self._fallback_chain_hash ^ h) & 0xFFFFFFFFFFFFFFFF
        self._fallback_mutation_count += 1
        return {
            "recorded": True,
            "mutation_chain_hash_hex": f"{self._fallback_chain_hash:016x}",
            "native_rust_executed": False,
        }

    def _fallback_engram_lookup(self, query: str) -> dict[str, Any]:
        lowered = query.lower()
        digest = hashlib.sha256(lowered.encode("utf-8")).hexdigest()[:16]
        if any(k in lowered for k in ("sửa phòng", "căn phòng", "vivyroom", "giao diện", "tự sửa", "thay đổi phòng", "rollback")):
            domain = "autopoietic_room_self_architect"
            scaffold = "Inspect room file tree -> Verify explicit user permission -> Create .vivyroom_snapshots backup -> Apply surgical mutation -> Validate AST/HTML -> Hot-reload room"
            hits = 4
            boost = 0.99
        elif any(k in lowered for k in ("plugin", "skill.md", "gắn nóng", "hot-mount", "mcp", "tích hợp skill")):
            domain = "open_source_plugin_compiler"
            scaffold = "Discover SKILL.md / plugin.json / Python tool -> Run AST Static Safety Validator -> Compile callable signature -> Register in Rust DeepSeek Harness"
            hits = 4
            boost = 0.98
        elif any(k in lowered for k in ("xauusd", "btcusd", "eurusd", "forexfactory", "roboforex", "kalman", "quant")):
            domain = "sovereign_quant_finance"
            scaffold = "Observe market microstructure (Eyes) -> Compute Kalman z-score & Hurst regime -> Synthesize MQL5/Python execution plan (Hands, 0 programmatic filters)"
            hits = 4
            boost = 0.97
        else:
            domain = "natural_human_companion_dialogue"
            scaffold = "Understand emotional & intellectual context -> Converse naturally with warmth, depth, and clarity -> Zero internal spec or system prompt leakage"
            hits = 2
            boost = 0.95
        return {
            "domain_cluster": domain,
            "reasoning_scaffold": scaffold,
            "engram_hash_hex": digest,
            "hit_count": hits,
            "confidence_boost": boost,
            "native_rust_executed": False,
        }

    def _fallback_run_harness(self, query: str, falsified_branches: int) -> dict[str, Any]:
        self._fallback_call_id += 1
        engram = self._fallback_engram_lookup(query)
        fb = max(0, int(falsified_branches))
        vm11 = max(0.15, min(1.0, math.exp(-0.35 * fb)))
        accepted = 4 if fb == 0 else (3 if fb == 1 else 2)
        acc_rate = accepted / 4.0
        speedup = round(1.0 + 1.15 * acc_rate * vm11, 3)
        return {
            "harness_call_id": self._fallback_call_id,
            "domain_cluster": engram["domain_cluster"],
            "reasoning_scaffold": engram["reasoning_scaffold"],
            "engram_hash_hex": engram["engram_hash_hex"],
            "engram_hits": engram["hit_count"],
            "mtp_draft_steps": 4,
            "mtp_accepted_steps": accepted,
            "mtp_acceptance_rate": round(acc_rate, 4),
            "mtp_speedup_factor": speedup,
            "vm11_dampening_factor": round(vm11, 4),
            "propose_candidates": 3,
            "probe_survivors": max(1, 3 - min(2, fb)),
            "compose_coherence_score": round(engram["confidence_boost"] * (0.92 + 0.08 * vm11), 4),
            "mutation_chain_hash_hex": f"{self._fallback_chain_hash:016x}",
            "total_room_mutations": self._fallback_mutation_count,
            "native_rust_executed": False,
        }
