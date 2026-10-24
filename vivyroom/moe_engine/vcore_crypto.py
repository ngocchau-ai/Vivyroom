"""Proprietary `Vivy Core` (3-Cores-in-1 `.vcore`) Cryptographic Container Format.

Enforces the mandatory directive:
"Từ giờ toàn bộ model Vivy phải được mã hóa Vivy core - không opensource (kiến trúc 3 core trong 1)"

Binary Layout of `.vcore` (`VIVY_CORE_TRI_ENC_V1`):
- Bytes [0..8]   : Magic Header `b"VCORE3\\x01\\x00"` (8 bytes)
- Bytes [8..24]  : 16-byte Cryptographic Salt
- Bytes [24..40] : 16-byte Stream Nonce
- Bytes [40..72] : 32-byte Encrypt-then-MAC `HMAC-SHA256` Tag over Header + Ciphertext
- Bytes [72..76] : uint32 Little-Endian length of JSON Metadata Header (`H_len`)
- Bytes [76..76+H_len] : UTF-8 JSON Metadata Header declaring the 3-Cores-in-1 anatomy,
  tensor descriptors (`name`, `core_id`, `shape`, `dtype`, `byte_offset`, `byte_length`),
  and proprietary closed-weight license (`open_source_weights = False`).
- Bytes [76+H_len..] : Authenticated `SHAKE256-CTR` + `zlib` encrypted tensor payload.

Decryption occurs strictly in RAM (`RAM-only zero-disk-leakage`).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

VCORE_MAGIC_HEADER = b"VCORE3\x01\x00"
TRI_CORE_REQUIRED_IDS = (
    "CORE_1_NPS_DECISION_SOUL_500M",
    "CORE_2_CAUTREO_FINANCE_MOE_VESSEL",
    "CORE_3_QWEN_DECOMPOSED_LLM_COMPILER_3B",
)


@dataclass(frozen=True)
class VivyCoreContainerHeader:
    format_version: str
    architecture: str
    open_source_weights: bool
    license_id: str
    container_role: str
    tri_core_components: tuple[str, str, str]
    tensor_count: int
    uncompressed_bytes: int
    encrypted_payload_bytes: int
    content_sha256: str
    tensor_directory: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _derive_encryption_and_mac_keys(master_key: bytes, salt: bytes, nonce: bytes) -> tuple[bytes, bytes]:
    """Derives separate 32-byte Encryption Key and 32-byte MAC Key via HKDF-SHA256."""
    prk = hmac.new(salt, master_key + nonce, hashlib.sha256).digest()
    enc_key = hmac.new(prk, b"VIVY_CORE_3IN1_ENC_KEY\x01", hashlib.sha256).digest()
    mac_key = hmac.new(prk, b"VIVY_CORE_3IN1_MAC_KEY\x02", hashlib.sha256).digest()
    return enc_key, mac_key


def _xor_shake256_keystream(data: bytes, enc_key: bytes, nonce: bytes) -> bytes:
    """Applies a fast C-accelerated SHAKE-256 counter-mode keystream via native C-long XOR."""
    n_bytes = len(data)
    if n_bytes == 0:
        return b""
    stream = hashlib.shake_256(enc_key + nonce + b"VCORE3_STREAM").digest(n_bytes)
    return (int.from_bytes(data, "little") ^ int.from_bytes(stream, "little")).to_bytes(n_bytes, "little")


def encrypt_tensors_to_vcore_bytes(
    tensors: dict[str, np.ndarray],
    master_key: bytes,
    container_role: str = "VIVY_TRI_CORE_MASTER",
    core_assignment_map: dict[str, str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> bytes:
    """Packs and encrypts a dictionary of numpy tensors into the proprietary `.vcore` 3-Cores-in-1 format."""
    if not tensors:
        raise ValueError("Cannot create an empty .vcore container.")

    raw_chunks: list[bytes] = []
    tensor_dir: list[dict[str, Any]] = []
    current_offset = 0

    for name in sorted(tensors.keys()):
        arr = np.ascontiguousarray(tensors[name], dtype=np.float32)
        b_data = arr.tobytes()
        core_id = "CORE_2_CAUTREO_FINANCE_MOE_VESSEL"
        if core_assignment_map and name in core_assignment_map:
            core_id = core_assignment_map[name]
        elif any(k in name for k in ("qwen", "cortical", "compositional", "voice", "text_slot", "southern")):
            core_id = "CORE_3_QWEN_DECOMPOSED_LLM_COMPILER_3B"
        elif any(k in name for k in ("soul", "equilibrium", "dmcv", "contradiction", "persona", "tricore.soul")):
            core_id = "CORE_1_NPS_DECISION_SOUL_500M"

        tensor_dir.append(
            {
                "name": name,
                "core_id": core_id,
                "shape": list(arr.shape),
                "dtype": "float32",
                "byte_offset": current_offset,
                "byte_length": len(b_data),
            }
        )
        raw_chunks.append(b_data)
        current_offset += len(b_data)

    uncompressed_blob = b"".join(raw_chunks)
    content_sha256 = hashlib.sha256(uncompressed_blob).hexdigest()
    compressed_blob = zlib.compress(uncompressed_blob, level=6)

    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(16)
    enc_key, mac_key = _derive_encryption_and_mac_keys(master_key, salt, nonce)
    ciphertext = _xor_shake256_keystream(compressed_blob, enc_key, nonce)

    header_dict: dict[str, Any] = {
        "format_version": "VIVY_CORE_TRI_ENC_V1",
        "architecture": "3_CORES_IN_1_UNIFIED_ORGANISM",
        "open_source_weights": False,
        "license_id": "VIVY-CORE-PROPRIETARY-CLOSED-WEIGHTS-V1",
        "container_role": container_role,
        "tri_core_components": list(TRI_CORE_REQUIRED_IDS),
        "tensor_count": len(tensor_dir),
        "uncompressed_bytes": len(uncompressed_blob),
        "encrypted_payload_bytes": len(ciphertext),
        "content_sha256": content_sha256,
        "tensor_directory": tensor_dir,
    }
    if extra_metadata:
        header_dict["extra_metadata"] = extra_metadata

    header_bytes = json.dumps(header_dict, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    header_len_packed = struct.pack("<I", len(header_bytes))

    mac_input = VCORE_MAGIC_HEADER + salt + nonce + header_len_packed + header_bytes + ciphertext
    hmac_tag = hmac.new(mac_key, mac_input, hashlib.sha256).digest()

    return VCORE_MAGIC_HEADER + salt + nonce + hmac_tag + header_len_packed + header_bytes + ciphertext


def encrypt_tensors_to_vcore_file(
    tensors: dict[str, np.ndarray],
    output_path: Path,
    master_key: bytes,
    container_role: str = "VIVY_TRI_CORE_MASTER",
    core_assignment_map: dict[str, str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Writes an encrypted `.vcore` container to `output_path` and returns its cryptographic receipt."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blob = encrypt_tensors_to_vcore_bytes(
        tensors=tensors,
        master_key=master_key,
        container_role=container_role,
        core_assignment_map=core_assignment_map,
        extra_metadata=extra_metadata,
    )
    output_path.write_bytes(blob)
    file_sha256 = hashlib.sha256(blob).hexdigest()
    hdr = inspect_vcore_header(output_path)
    return {
        "path": str(output_path),
        "size_bytes": len(blob),
        "file_sha256": file_sha256,
        "format_version": hdr["format_version"],
        "architecture": hdr["architecture"],
        "open_source_weights": hdr["open_source_weights"],
        "container_role": hdr["container_role"],
        "tensor_count": hdr["tensor_count"],
        "uncompressed_bytes": hdr["uncompressed_bytes"],
    }


def inspect_vcore_header(vcore_source: Path | bytes) -> dict[str, Any]:
    """Reads and validates the `.vcore` public metadata header without decrypting the secret weights."""
    blob = vcore_source.read_bytes() if isinstance(vcore_source, Path) else vcore_source
    if len(blob) < 76:
        raise ValueError("Invalid .vcore container: truncated header (< 76 bytes).")
    magic = blob[:8]
    if magic != VCORE_MAGIC_HEADER:
        raise ValueError(f"Invalid .vcore magic header: {magic!r}")
    (h_len,) = struct.unpack("<I", blob[72:76])
    if len(blob) < 76 + h_len:
        raise ValueError("Invalid .vcore container: truncated JSON metadata header.")
    header_json = blob[76 : 76 + h_len].decode("utf-8")
    data: dict[str, Any] = json.loads(header_json)
    if data.get("open_source_weights") is not False:
        raise ValueError("Security Policy Violation: .vcore container must have open_source_weights == False.")
    return data


def decrypt_vcore_bytes_to_memory(
    blob: bytes,
    master_key: bytes,
    filter_names: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Verifies HMAC-SHA256 authenticity and decrypts `.vcore` tensors strictly in RAM."""
    if len(blob) < 76:
        raise ValueError("Truncated .vcore container.")
    magic = blob[:8]
    if magic != VCORE_MAGIC_HEADER:
        raise ValueError("Corrupted .vcore magic header.")
    salt = blob[8:24]
    nonce = blob[24:40]
    expected_tag = blob[40:72]
    (h_len,) = struct.unpack("<I", blob[72:76])
    header_bytes = blob[76 : 76 + h_len]
    ciphertext = blob[76 + h_len :]

    enc_key, mac_key = _derive_encryption_and_mac_keys(master_key, salt, nonce)
    mac_input = magic + salt + nonce + blob[72:76] + header_bytes + ciphertext
    actual_tag = hmac.new(mac_key, mac_input, hashlib.sha256).digest()
    if not hmac.compare_digest(expected_tag, actual_tag):
        raise PermissionError("VivyCore HMAC-SHA256 verification failed: invalid key or tampered .vcore file.")

    header: dict[str, Any] = json.loads(header_bytes.decode("utf-8"))
    compressed_blob = _xor_shake256_keystream(ciphertext, enc_key, nonce)
    uncompressed_blob = zlib.decompress(compressed_blob)

    actual_sha256 = hashlib.sha256(uncompressed_blob).hexdigest()
    if actual_sha256 != header["content_sha256"]:
        raise ValueError("VivyCore payload SHA-256 mismatch after decompression.")

    tensors: dict[str, np.ndarray] = {}
    for entry in header["tensor_directory"]:
        t_name = entry["name"]
        if filter_names is not None and t_name not in filter_names:
            continue
        b_off = int(entry["byte_offset"])
        b_len = int(entry["byte_length"])
        shape = tuple(int(x) for x in entry["shape"])
        arr = np.frombuffer(uncompressed_blob, dtype=np.float32, count=b_len // 4, offset=b_off).reshape(shape).copy()
        tensors[t_name] = arr

    return tensors, header


def decrypt_vcore_file_to_memory(
    vcore_path: Path,
    master_key: bytes,
    filter_names: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Decrypts a `.vcore` file from disk strictly into RAM."""
    blob = Path(vcore_path).read_bytes()
    return decrypt_vcore_bytes_to_memory(blob, master_key=master_key, filter_names=filter_names)
