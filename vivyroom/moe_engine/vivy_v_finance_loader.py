"""Native Loader & In-RAM Architecture Decryptor for `Vivy-V-Finance-3B` (`Receipt R-105`).

Loads `models/vivy_v_finance_3b/`:
- `vivy-v-finance-3b_arch.vcore` (`VIVY_V_ARCH_ENC_V1`: `architecture_encrypted = True`,
  `open_source_architecture = False`, `open_source_weights = True`)
- `vivy-v-finance-3b_weights.safetensors` (`263` open-source anonymized weight tensors `weights.layer_0000..0262`)
- Reconstructs all `263` internal `Vivy-V` Tri-Core tensors + `3` routing matrices strictly in RAM
  with zero external `torch`/`safetensors` C-extension dependency.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import struct
from typing import Any
import zlib

import numpy as np

from ..config import MODELS_DIR

VCORE_MAGIC_HEADER = b"VCORE3\x01\x00"
_SOVEREIGN_ARCH_SALT = b"NGOCCHAU_AI_VIVY_V_ARCH_ENC_SOVEREIGN_V1_2026"
VIVY_V_FINANCE_DIR = MODELS_DIR / "vivy_v_finance_3b"


def derive_vivy_v_arch_key(passphrase: str | None = None) -> bytes:
    default_key = "vivy-v-architecture-sovereign-master-key-2026"
    raw = (
        passphrase.encode("utf-8")
        if passphrase
        else os.environ.get("VIVY_V_ARCH_SOVEREIGN_KEY", default_key).encode("utf-8")
    )
    return hashlib.pbkdf2_hmac("sha256", raw, _SOVEREIGN_ARCH_SALT, iterations=12000, dklen=32)


def _derive_enc_mac_keys(master_key: bytes, salt: bytes, nonce: bytes) -> tuple[bytes, bytes]:
    prk = hmac.new(salt, master_key + nonce, hashlib.sha256).digest()
    enc_key = hmac.new(prk, b"VIVY_V_ARCH_ENC_KEY\x01", hashlib.sha256).digest()
    mac_key = hmac.new(prk, b"VIVY_V_ARCH_MAC_KEY\x02", hashlib.sha256).digest()
    return enc_key, mac_key


def _xor_shake256_stream(data: bytes, enc_key: bytes, nonce: bytes) -> bytes:
    if not data:
        return b""
    stream = hashlib.shake_256(enc_key + nonce + b"VIVY_V_ARCH_STREAM").digest(len(data))
    if hasattr(np, "frombuffer") and hasattr(np, "bitwise_xor"):
        try:
            a = np.frombuffer(data, dtype=np.uint8)
            b = np.frombuffer(stream, dtype=np.uint8)
            return np.bitwise_xor(a, b).tobytes()
        except Exception:
            pass
    return bytes(x ^ y for x, y in zip(data, stream))


def decrypt_vivy_v_architecture_from_vcore(
    blob: bytes,
    master_key: bytes | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Decrypts `vivy-v-finance-3b_arch.vcore` (`VIVY_V_ARCH_ENC_V1`) strictly in RAM."""
    if len(blob) < 76 or blob[:8] != VCORE_MAGIC_HEADER:
        raise ValueError("Invalid .vcore magic header.")
    key = master_key or derive_vivy_v_arch_key()
    salt = blob[8:24]
    nonce = blob[24:40]
    expected_tag = blob[40:72]
    (h_len,) = struct.unpack("<I", blob[72:76])
    h_bytes = blob[76 : 76 + h_len]
    ciphertext = blob[76 + h_len :]

    enc_key, mac_key = _derive_enc_mac_keys(key, salt, nonce)
    mac_input = blob[:8] + salt + nonce + blob[72:76] + h_bytes + ciphertext
    actual_tag = hmac.new(mac_key, mac_input, hashlib.sha256).digest()
    if not hmac.compare_digest(expected_tag, actual_tag):
        raise PermissionError("Vivy-V Architecture HMAC-SHA256 verification failed.")

    public_header = json.loads(h_bytes.decode("utf-8"))
    if public_header.get("architecture_encrypted") is not True:
        raise ValueError("Policy violation: architecture_encrypted must be True.")
    if public_header.get("open_source_architecture") is not False:
        raise ValueError("Policy violation: open_source_architecture must be False.")
    if public_header.get("open_source_weights") is not True:
        raise ValueError("Policy violation: open_source_weights must be True.")

    comp_blob = _xor_shake256_stream(ciphertext, enc_key, nonce)
    raw_blob = zlib.decompress(comp_blob)
    if hashlib.sha256(raw_blob).hexdigest() != public_header["content_sha256"]:
        raise ValueError("Decrypted architecture SHA-256 mismatch.")

    (inner_json_len,) = struct.unpack("<I", raw_blob[:4])
    inner_json_bytes = raw_blob[4 : 4 + inner_json_len]
    tensor_payload = raw_blob[4 + inner_json_len :]
    inner_pkg = json.loads(inner_json_bytes.decode("utf-8"))

    tensors: dict[str, Any] = {}
    for item in inner_pkg["routing_tensor_directory"]:
        off = int(item["byte_offset"])
        ln = int(item["byte_length"])
        shp = tuple(int(x) for x in item["shape"])
        chunk = tensor_payload[off : off + ln]
        arr = np.frombuffer(chunk, dtype=np.float32).reshape(shp).copy()
        tensors[item["name"]] = arr
    return inner_pkg, tensors, public_header


def load_safetensors_pure_f32(path: Path) -> dict[str, Any]:
    """Zero-dependency pure-Python `.safetensors` float32 reader (`0` torch/safetensors C-ext required)."""
    raw = path.read_bytes()
    if len(raw) < 8:
        return {}
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    data_start = 8 + header_len
    out: dict[str, Any] = {}
    for k, meta in header.items():
        if k == "__metadata__":
            continue
        offsets = meta["data_offsets"]
        start, end = int(offsets[0]), int(offsets[1])
        shp = tuple(int(x) for x in meta["shape"])
        chunk = raw[data_start + start : data_start + end]
        arr = np.frombuffer(chunk, dtype=np.float32)
        if shp:
            arr = arr.reshape(shp)
        out[k] = arr.copy()
    return out


def load_and_verify_vivy_v_finance_3b(suite_dir: Path = VIVY_V_FINANCE_DIR) -> dict[str, Any]:
    """Loads `Vivy-V-Finance-3B` (`vivy-v-finance-3b_arch.vcore` + `vivy-v-finance-3b_weights.vcore` / `.safetensors`)
    and reconstructs the internal 263-tensor architecture mapping in RAM.
    """
    from .vcore_crypto import decrypt_vcore_bytes_to_memory, encrypt_tensors_to_vcore_file

    suite_dir = Path(suite_dir)
    vcore_path = suite_dir / "vivy-v-finance-3b_arch.vcore"
    weights_vcore_path = suite_dir / "vivy-v-finance-3b_weights.vcore"
    weights_st_path = suite_dir / "vivy-v-finance-3b_weights.safetensors"
    weights_key = hashlib.sha256(b"vivyroom-vivy-v-finance-3b-weights-vcore-key").digest()

    if weights_st_path.is_file() and not weights_vcore_path.is_file():
        st_tensors = load_safetensors_pure_f32(weights_st_path)
        encrypt_tensors_to_vcore_file(
            tensors=st_tensors,
            output_path=weights_vcore_path,
            master_key=weights_key,
            container_role="VIVY_V_FINANCE_3B_263_WEIGHTS",
        )
        weights_st_path.unlink()

    if not vcore_path.is_file() or not weights_vcore_path.is_file():
        return {
            "ready": False,
            "model_id": "Vivy-V-Finance-3B",
            "reason": "Missing vivy-v-finance-3b_arch.vcore or vivy-v-finance-3b_weights.vcore",
        }

    inner_pkg, routing_tensors, pub_hdr = decrypt_vivy_v_architecture_from_vcore(vcore_path.read_bytes())
    anon_weights, _ = decrypt_vcore_bytes_to_memory(weights_vcore_path.read_bytes(), weights_key)
    bp = inner_pkg.get("architecture_blueprint", {})
    raw_map = bp.get("open_weight_key_to_internal_architecture_map", [])
    key_map: dict[str, str] = {}
    if isinstance(raw_map, list):
        for entry in raw_map:
            if isinstance(entry, dict) and "open_weight_key" in entry and "internal_vivy_v_symbol" in entry:
                sym = str(entry["internal_vivy_v_symbol"])
                if sym.startswith("vivy_r_finance_3b_v74_tricore_decision."):
                    sym = sym[len("vivy_r_finance_3b_v74_tricore_decision.") :]
                key_map[str(entry["open_weight_key"])] = sym
    elif isinstance(raw_map, dict):
        key_map = {str(k): str(v) for k, v in raw_map.items()}

    reconstructed_tensors: dict[str, Any] = {}
    for anon_key, arr in anon_weights.items():
        internal_key = key_map.get(anon_key, anon_key)
        reconstructed_tensors[internal_key] = arr
    for r_key, r_arr in routing_tensors.items():
        reconstructed_tensors[r_key] = r_arr

    return {
        "ready": True,
        "model_id": pub_hdr.get("model_id", "Vivy-V-Finance-3B"),
        "generation": pub_hdr.get("generation", "Vivy-V"),
        "format_version": pub_hdr.get("format_version", "VIVY_V_ARCH_ENC_V1"),
        "architecture_encrypted": pub_hdr.get("architecture_encrypted", True),
        "open_source_architecture": pub_hdr.get("open_source_architecture", False),
        "open_source_weights": pub_hdr.get("open_source_weights", True),
        "tri_core_components": inner_pkg.get("vivy_v_tri_core_components", []),
        "open_weights_tensor_count": len(anon_weights),
        "routing_tensor_count": len(routing_tensors),
        "total_reconstructed_tensors": len(reconstructed_tensors),
        "legacy_v6_replaced": bp.get("legacy_v6_replaced", "Vivy-R-Finance-3B (v6.6..v7.4)"),
        "reconstructed_tensors": reconstructed_tensors,
    }
