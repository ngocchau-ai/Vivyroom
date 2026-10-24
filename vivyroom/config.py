"""Hardware & Memory Budget Configuration for `Vivyroom` (Strict <= 2,048 MB RAM Ceiling)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = PACKAGE_ROOT
REPO_ROOT = PACKAGE_ROOT.parent
NATIVE_BIN_DIR = PACKAGE_ROOT / "native" / "bin"
MODELS_DIR = REPO_ROOT / "models"
UI_DIR = REPO_ROOT / "ui"
DATA_DIR = REPO_ROOT / ".vivyroom_data"


@dataclass
class VivyroomConfig:
    """Runtime configuration enforcing the strict 2 GB (2,048 MB) RAM envelope."""

    # Strict Hardware Memory Ceilings
    device_ram_limit_mb: int = 2048
    agent_and_moe_budget_mb: int = 1500
    circuit_breaker_ratio: float = 0.85  # Triggers Top-1 Ultra-Sparse fallback at 1,740 MB
    pager_max_ram_bytes: int = 256 * 1024 * 1024  # 256 MB dedicated C11 MoE Expert slice window

    # MoE Routing Configuration (Vivy-Finance-3B-MoE)
    total_moe_experts: int = 8
    default_top_k_experts: int = 2
    low_memory_top_k_experts: int = 1
    sticky_task_affinity_bonus: float = 0.18

    # Zero-Relay Autonomous Agent Guardrails
    zero_external_relay: bool = True
    max_single_step_timeout_ms: float = 1500.0
    max_agent_loop_steps: int = 6
    enable_self_healing_compiler: bool = True
    programmatic_finance_filters: int = 0  # Strictly 0 per VyTrading Core Rule §5

    # Proprietary Model Encryption Policy
    proprietary_vcore_encryption_mandatory: bool = True
    allow_unencrypted_model_weights: bool = False
    tri_core_architecture_id: str = "VIVY_CORE_3_IN_1_PROPRIETARY_V1"

    # Server Settings
    host: str = "127.0.0.1"
    port: int = 7860

    @property
    def circuit_breaker_mb(self) -> float:
        return round(self.device_ram_limit_mb * self.circuit_breaker_ratio, 2)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["circuit_breaker_mb"] = self.circuit_breaker_mb
        return data


DEFAULT_CONFIG = VivyroomConfig()
