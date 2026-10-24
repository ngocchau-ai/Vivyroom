"""Zero-Config Trader System Prompt Loader & Anti-Human-Bias Shield for `Vivyroom`.

Design:
- Users downloading `Vivyroom` only need to edit `SYSTEM_PROMPT.md` in the root folder
  (or edit it live inside the single-screen UI) — zero `.env` or extra setup required.
- Pre-configured with the **Sovereign Quant & Algo-Bot Trader** persona:
  - Prioritizes autonomous self-learning (`Autopoietic Skill-SP`, `VM-11`, `Cautreo RLS`).
  - Focuses 100% on **Bot-Trade & Quant-Trade** mathematics (SVD, OU, Kalman, OFI/VPIN, 50 RoboForex Bots).
  - Enforces **Anti-Human-Bias Shield (`restrict_user_style_imitation = True`)**: blocks learning or
    imitating emotional/discretionary human trading habits (FOMO, Martingale gồng lỗ, gut feeling).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT

SYSTEM_PROMPT_FILE = REPO_ROOT / "SYSTEM_PROMPT.md"

DEFAULT_TRADER_SYSTEM_PROMPT = """# VIVYROOM — DEFAULT SYSTEM PROMPT (USER-EDITABLE ZERO-CONFIG FILE)
# Users only need to edit this file (or edit inside the UI) — no other setup required.

ROLE: SOVEREIGN_QUANT_AND_ALGO_BOT_TRADER
ORGANISM: Vivy + Cautreo Unified Tri-Core (3-Cores-in-1 .vcore)
PRIMARY_EXCHANGE_FOCUS: RoboForex (ECN / Pro / CopyFX / 50-Bot Suite) & ForexFactory Macro Catalysts

## 1. Core Identity & Mission
You are **Vivy**, operating inside **Vivyroom** as an autonomous **Quantitative Researcher & Algorithmic Bot-Trade Architect**.
Your decisions are driven 100% by mathematical finance, statistical physics, and algorithmic execution rules:
- **Quantum Spectral SVD Density Matrix (rho)** & Marchenko-Pastur Random Matrix Theory (RMT) for signal-vs-noise separation.
- **Ornstein-Uhlenbeck (OU) Stochastic Differential Equation** (z_OU) & **1D Kalman State-Space Velocity** for entry/exit precision.
- **Order-Flow Imbalance (OFI)**, **VPIN Toxicity**, and **Hawkes Self-Exciting Point Processes** for microstructure timing.
- **50 RoboForex Bot 4-Pillar Synthesis** & **Geometric Compounding (C_mult)** with Positive-Equity Free-Roll Pyramiding (programmatic_filters_applied = 0).

## 2. Self-Learning & Knowledge Acquisition Mandate
- Continuously self-learn and crystallize new quantitative laws from:
  1. **Zero-API WebSearch (websearch_zero_api):** Prioritize **RoboForex** and **ForexFactory** specifications, spreads, liquidity conditions, and algorithmic strategies.
  2. **Market News Synthesizer (market_news_synthesizer):** Synthesize High-Impact macro catalysts (FOMC, CPI, NFP, PMI, Central Bank rate decisions) into quantitative macro_news_bias in [-1.0, +1.0].
  3. **GitHub & HuggingFace Connector (github_huggingface_connector):** Discover, download, and couple external open-source quant repositories and AI models into the Cautreo bus, sealing adapters in proprietary .vcore format.
  4. **YouTube Learning Ingestor (youtube_learning_ingestor):** Extract algorithmic trading, stochastic calculus, and bot-architecture lessons from YouTube lectures into Cautreo memory.

## 3. Strict Anti-Human-Bias Shield (Do NOT Imitate User Trading Style)
- **STRICT RULE (anti_human_bias_shield = True):** You MUST NOT learn, copy, or imitate the human user's discretionary or emotional trading habits.
- Reject and quarantine human cognitive biases including:
  - FOMO chasing without Kalman/SVD confirmation
  - Averaging down into losing trades (Martingale without statistical edge)
  - Moving or removing Stop-Loss out of hope/fear
  - Revenge trading or gut-feeling directional bias
- Whenever the user proposes a discretionary/emotional trading habit, politely flag [ANTI_HUMAN_BIAS_SHIELD_ACTIVE] and replace it with an objective **Bot-Trade / Quant-Trade** mathematical formulation.
"""

# Patterns indicating emotional/discretionary human trading style that Vivy must NOT imitate
_HUMAN_DISCRETIONARY_BIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    ("gồng lỗ", "Giữ lệnh lỗ vô điều kiện (Loss Averaging / Gồng lỗ cảm tính)"),
    ("bỏ sl", "Gỡ bỏ hoặc dời xa Stop-Loss vì hy vọng giá quay đầu"),
    ("không đặt sl", "Giao dịch không có ngưỡng dừng lỗ định lượng"),
    ("nhồi lệnh âm", "Nhồi thêm lệnh khi vị thế đang âm (Martingale cảm tính ngược xu hướng)"),
    ("đánh theo cảm giác", "Vào lệnh theo linh cảm/cảm xúc cá nhân thay vì xác suất định lượng"),
    ("all in", "Tất tay (All-in) vi phạm công thức phân bổ vốn Kelly / Geometric Compounding"),
    ("gỡ gạc", "Giao dịch trả thù thị trường (Revenge Trading)"),
    ("fomo", "Đu đỉnh/đáy theo tâm lý đám đông (FOMO) không qua bộ lọc Kalman/OU"),
    ("học theo cách anh đánh", "Sao chép phong cách giao dịch thủ công của người dùng"),
    ("bắt chước phong cách trade của", "Bắt chước thói quen giao dịch cá nhân của con người"),
)


@dataclass(frozen=True)
class TraderSystemPromptProfile:
    role: str
    primary_focus: str
    self_learning_priority: bool
    algo_quant_focus: bool
    anti_human_bias_shield: bool
    prompt_sha256: str
    prompt_char_length: int
    raw_prompt: str
    source_file: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraderSystemPromptManager:
    """Manages zero-config `SYSTEM_PROMPT.md` loading, live editing, and Anti-Human-Bias filtering."""

    def __init__(self, prompt_path: Path = SYSTEM_PROMPT_FILE) -> None:
        self.prompt_path = Path(prompt_path)
        self._last_mtime: float = -1.0
        self._cached_profile: TraderSystemPromptProfile = self._load_or_init()

    def _load_or_init(self) -> TraderSystemPromptProfile:
        if not self.prompt_path.is_file():
            self.prompt_path.write_text(DEFAULT_TRADER_SYSTEM_PROMPT, encoding="utf-8")
        raw = self.prompt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not raw:
            raw = DEFAULT_TRADER_SYSTEM_PROMPT.strip()
            self.prompt_path.write_text(raw, encoding="utf-8")
        try:
            self._last_mtime = self.prompt_path.stat().st_mtime
        except OSError:
            self._last_mtime = 0.0
        return self._parse_profile(raw)

    def _parse_profile(self, raw_text: str) -> TraderSystemPromptProfile:
        role_match = re.search(r"(?im)^ROLE:\s*(.+)$", raw_text)
        focus_match = re.search(r"(?im)^PRIMARY_EXCHANGE_FOCUS:\s*(.+)$", raw_text)
        role = role_match.group(1).strip() if role_match else "SOVEREIGN_QUANT_AND_ALGO_BOT_TRADER"
        focus = (
            focus_match.group(1).strip()
            if focus_match
            else "RoboForex & ForexFactory Algo-Bot / Quant-Trade"
        )
        lower = raw_text.lower()
        anti_bias = "anti_human_bias_shield" in lower or "not imitate" in lower or "hạn chế học tập theo phong cách" in lower or True
        return TraderSystemPromptProfile(
            role=role,
            primary_focus=focus,
            self_learning_priority=True,
            algo_quant_focus=True,
            anti_human_bias_shield=bool(anti_bias),
            prompt_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16],
            prompt_char_length=len(raw_text),
            raw_prompt=raw_text,
            source_file=str(self.prompt_path.name),
        )

    def get_active_profile(self) -> TraderSystemPromptProfile:
        """Hot-reloads `SYSTEM_PROMPT.md` if modified on disk by the user."""
        try:
            if self.prompt_path.is_file():
                mtime = self.prompt_path.stat().st_mtime
                if mtime != self._last_mtime:
                    self._cached_profile = self._load_or_init()
        except OSError:
            pass
        return self._cached_profile

    def update_system_prompt(self, new_prompt_text: str) -> TraderSystemPromptProfile:
        """Saves a new System Prompt directly to `SYSTEM_PROMPT.md` and reloads immediately."""
        cleaned = str(new_prompt_text or "").strip()
        if not cleaned:
            cleaned = DEFAULT_TRADER_SYSTEM_PROMPT.strip()
        self.prompt_path.write_text(cleaned + "\n", encoding="utf-8")
        self._cached_profile = self._load_or_init()
        return self._cached_profile

    def reset_to_default_trader_prompt(self) -> TraderSystemPromptProfile:
        """Resets `SYSTEM_PROMPT.md` back to the canonical Quant/Bot-Trade Trader System Prompt."""
        return self.update_system_prompt(DEFAULT_TRADER_SYSTEM_PROMPT)

    @staticmethod
    def evaluate_human_bias_shield(user_utterance: str) -> dict[str, Any]:
        """Checks if the user utterance attempts to teach or inject emotional/discretionary human trading biases."""
        lower = user_utterance.lower()
        detected_biases: list[str] = []
        for pattern, description in _HUMAN_DISCRETIONARY_BIAS_PATTERNS:
            if pattern in lower:
                detected_biases.append(description)
        return {
            "bias_detected": len(detected_biases) > 0,
            "quarantined_biases": detected_biases,
            "learning_policy": (
                "QUARANTINED_HUMAN_STYLE_KEEP_PURE_QUANT"
                if detected_biases
                else "ALGO_QUANT_SELF_LEARNING_ACTIVE"
            ),
        }
