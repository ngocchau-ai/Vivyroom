"""Proprietary Encrypted `Vivy Core` (3-Cores-in-1 `.vcore`) & `Vivy-Finance-3B-MoE` Engine."""

from __future__ import annotations

from .vcore_crypto import (
    VCORE_MAGIC_HEADER,
    VivyCoreContainerHeader,
    decrypt_vcore_bytes_to_memory,
    decrypt_vcore_file_to_memory,
    encrypt_tensors_to_vcore_bytes,
    encrypt_tensors_to_vcore_file,
    inspect_vcore_header,
)

__all__ = [
    "VCORE_MAGIC_HEADER",
    "VivyCoreContainerHeader",
    "encrypt_tensors_to_vcore_bytes",
    "encrypt_tensors_to_vcore_file",
    "decrypt_vcore_bytes_to_memory",
    "decrypt_vcore_file_to_memory",
    "inspect_vcore_header",
]
