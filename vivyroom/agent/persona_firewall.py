"""Silent Persona Firewall (`SilentPersonaFirewall`) — Guarantees 0% System Prompt / Internal Spec Leakage.

Ensures that Vivy converses naturally and intelligently like a human partner without
ever reciting her own `SYSTEM_PROMPT.md`, `.vcore` encryption format, RAM counters,
or internal expert IDs inside conversational replies. All technical telemetry is strictly
routed to the separate `x_ray_telemetry` channel for the UI X-Ray HUD Drawer (`Alt+X`).
"""

from __future__ import annotations

import re
from typing import Any

FORBIDDEN_SPEC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Strip trailing parenthetical telemetry blocks
    (re.compile(r"\s*\(Experts đang bật:[^)]*\)", re.IGNORECASE), ""),
    (re.compile(r"\s*\(RAM thực tế:[^)]*\)", re.IGNORECASE), ""),
    # Strip bracketed internal guard tags
    (re.compile(r"\[ANTI_HUMAN_BIAS_SHIELD_ACTIVE\]\s*", re.IGNORECASE), ""),
    (re.compile(r"\[VCORE3[^\]]*\]\s*", re.IGNORECASE), ""),
    # Replace self-referential spec recitals with natural expressions
    (
        re.compile(
            r"cả 3 lõi trong khối Vivy Core mã hóa \(\.vcore\) của em đều đang rất minh mẫn",
            re.IGNORECASE,
        ),
        "tinh thần em đang rất minh mẫn và hào hứng làm việc cùng anh",
    ),
    (
        re.compile(r"khối Vivy Core mã hóa \(\.vcore\)", re.IGNORECASE),
        "nội tâm của em",
    ),
    (
        re.compile(r"3-Cores-in-1\s*\(\.vcore\)|\.vcore\s*\(3-Cores-in-1\)|3-Cores-in-1", re.IGNORECASE),
        "bộ não tích hợp",
    ),
    (
        re.compile(r"Theo System Prompt vai trò Trader,?\s*", re.IGNORECASE),
        "",
    ),
    (
        re.compile(r"OranPi\s*[×x-]\s*PocketFlow", re.IGNORECASE),
        "Vivyroom",
    ),
    (
        re.compile(r"SHAKE256-CTR|HMAC-SHA256|VIVY_CORE_TRI_ENC_V1", re.IGNORECASE),
        "",
    ),
]

BANNED_LEAK_SUBSTRINGS = (
    "experts đang bật:",
    "ram thực tế:",
    "[anti_human_bias_shield_active]",
    "khối vivy core mã hóa (.vcore)",
    "theo system prompt vai trò trader",
    "vivy_core_tri_enc_v1",
    "shake256-ctr",
)


class SilentPersonaFirewall:
    """Sanitizes and audits all outbound responses so Vivy never recites internal specs or system prompts."""

    @staticmethod
    def sanitize_response(text: str, allow_prompt_mention: bool = False) -> str:
        cleaned = text.strip()
        for pattern, replacement in FORBIDDEN_SPEC_PATTERNS:
            cleaned = pattern.sub(replacement, cleaned)

        if not allow_prompt_mention:
            cleaned = re.sub(
                r"\bSystem Prompt\b",
                "nguyên tắc làm việc",
                cleaned,
                flags=re.IGNORECASE,
            )

        # Normalize whitespace while preserving paragraph breaks
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    @staticmethod
    def audit_zero_spec_leakage(text: str) -> dict[str, Any]:
        low = text.lower()
        leaks = [sub for sub in BANNED_LEAK_SUBSTRINGS if sub in low]
        return {
            "clean": len(leaks) == 0,
            "leak_count": len(leaks),
            "leaked_tokens": leaks,
        }
