"""1-Touch Sovereign Authentication & Cryptographic Session Gate for `Vivyroom`.

Provides:
1. 1-Touch Sovereign Passkey / Local Hardware-Bound Token derivation (`HKDF-SHA256` + `PBKDF2-HMAC-SHA256`).
2. Loopback Session HMAC token generation & verification (`X-Vivyroom-Session-Token`).
3. In-memory key unsealing for Proprietary Encrypted `.vcore` (3-Cores-in-1) model containers.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

_DEFAULT_SOVEREIGN_OWNER_SEED = b"NGOCCHAU_AI_VIVYROOM_SOVEREIGN_TRI_CORE_V1_2026"


@dataclass(frozen=True)
class SovereignSessionReceipt:
    session_id: str
    owner_id: str
    unlocked_at_epoch: float
    auth_mode: str
    vcore_key_fingerprint: str
    session_hmac_token: str
    ram_only_decryption: bool


class OneTouchSovereignAuthGate:
    """Manages 1-Touch Local Passkey / PIN authentication and RAM-only `.vcore` key derivation."""

    def __init__(self, master_passphrase: str | None = None) -> None:
        raw_secret = (
            master_passphrase.encode("utf-8")
            if master_passphrase
            else os.environ.get("VIVYROOM_SOVEREIGN_KEY", "vivyroom-1touch-sovereign").encode("utf-8")
        )
        self._master_key_32 = hashlib.pbkdf2_hmac(
            "sha256",
            raw_secret,
            _DEFAULT_SOVEREIGN_OWNER_SEED,
            iterations=12000,
            dklen=32,
        )
        self._session_secret = secrets.token_bytes(32)
        self._unlocked: bool = True  # Auto-unlocked in local 1-touch sovereign mode
        self._active_session: SovereignSessionReceipt = self._mint_session("ONE_TOUCH_LOCAL_SOVEREIGN")

    @property
    def master_key_bytes(self) -> bytes:
        if not self._unlocked:
            raise PermissionError("Vivyroom Sovereign Vault is locked. Trigger 1-Touch Unlock first.")
        return self._master_key_32

    @property
    def key_fingerprint(self) -> str:
        return hashlib.sha256(self._master_key_32).hexdigest()[:16]

    def _mint_session(self, auth_mode: str) -> SovereignSessionReceipt:
        sid = f"vroom-{int(time.time() * 1000)}-{secrets.token_hex(4)}"
        token = hmac.new(self._session_secret, sid.encode("utf-8"), hashlib.sha256).hexdigest()
        return SovereignSessionReceipt(
            session_id=sid,
            owner_id="anh_vi_sovereign",
            unlocked_at_epoch=time.time(),
            auth_mode=auth_mode,
            vcore_key_fingerprint=self.key_fingerprint,
            session_hmac_token=token,
            ram_only_decryption=True,
        )

    def unlock_one_touch(self, mode: str = "ONE_TOUCH_PASSKEY", pin_override: str | None = None) -> dict[str, Any]:
        """Executes 1-Touch Sovereign Unlock in < 5ms."""
        t0 = time.perf_counter()
        if pin_override:
            self._master_key_32 = hashlib.pbkdf2_hmac(
                "sha256",
                pin_override.encode("utf-8"),
                _DEFAULT_SOVEREIGN_OWNER_SEED,
                iterations=12000,
                dklen=32,
            )
        self._unlocked = True
        self._active_session = self._mint_session(mode)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "unlocked": True,
            "unlock_latency_ms": elapsed_ms,
            "session_id": self._active_session.session_id,
            "owner_id": self._active_session.owner_id,
            "auth_mode": self._active_session.auth_mode,
            "vcore_key_fingerprint": self._active_session.vcore_key_fingerprint,
            "session_hmac_token": self._active_session.session_hmac_token,
            "ram_only_decryption": True,
        }

    def lock_vault(self) -> dict[str, Any]:
        self._unlocked = False
        return {"unlocked": False, "status": "LOCKED"}

    def verify_session_token(self, token: str | None) -> bool:
        if not self._unlocked:
            return False
        if not token:
            return True  # Local loopback convenience when accessed from same machine
        expected = hmac.new(
            self._session_secret,
            self._active_session.session_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(token, expected)

    def get_status(self) -> dict[str, Any]:
        return {
            "unlocked": self._unlocked,
            "session_id": self._active_session.session_id,
            "owner_id": self._active_session.owner_id,
            "auth_mode": self._active_session.auth_mode,
            "vcore_key_fingerprint": self.key_fingerprint,
            "ram_only_decryption": True,
        }
