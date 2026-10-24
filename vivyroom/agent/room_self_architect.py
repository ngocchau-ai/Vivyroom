"""Autopoietic Room Self-Control, Self-Repair & Self-Modification Engine (`VivyRoomSelfArchitectEngine`).

Just like Antigravity (Vy) can inspect, debug, and modify the workspace, this engine gives
the native embedded Vivy full introspection and self-modification capability over her own
`Vivyroom` installation — strictly gated by explicit user permission (`UserPermissionGate` +
Rust `vh_record_room_mutation`) and protected by automatic `.vivyroom_snapshots/` backups
and AST syntax verification before commit.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import re
import time
from pathlib import Path
from typing import Any

from ..config import MODELS_DIR, NATIVE_BIN_DIR, PACKAGE_ROOT, REPO_ROOT, UI_DIR
from .deepseek_harness_bridge import RustDeepSeekHarnessBridge

SNAPSHOTS_DIR = REPO_ROOT / ".vivyroom_snapshots"

THEME_PRESETS: dict[str, dict[str, str]] = {
    "violet": {
        "name": "Nebula Violet Sovereign",
        "--bg-main": "#090614",
        "--bg-surface": "#120c24",
        "--bg-bubble-vivy": "#1a1133",
        "--accent-cyan": "#c084fc",
        "--accent-emerald": "#34d399",
    },
    "emerald": {
        "name": "Quantum Emerald Matrix",
        "--bg-main": "#04100c",
        "--bg-surface": "#081c15",
        "--bg-bubble-vivy": "#0d291f",
        "--accent-cyan": "#10b981",
        "--accent-emerald": "#38bdf8",
    },
    "gold": {
        "name": "Imperial Gold & Obsidian",
        "--bg-main": "#0c0a06",
        "--bg-surface": "#17130b",
        "--bg-bubble-vivy": "#241d10",
        "--accent-cyan": "#f59e0b",
        "--accent-emerald": "#10b981",
    },
    "crimson": {
        "name": "Cyber Crimson Velvet",
        "--bg-main": "#120609",
        "--bg-surface": "#1f0b11",
        "--bg-bubble-vivy": "#2e1019",
        "--accent-cyan": "#fb7185",
        "--accent-emerald": "#38bdf8",
    },
    "default": {
        "name": "Deep Ocean Cyan (Default)",
        "--bg-main": "#060a12",
        "--bg-surface": "#0c1322",
        "--bg-bubble-vivy": "#111b2e",
        "--accent-cyan": "#38bdf8",
        "--accent-emerald": "#10b981",
    },
}


class VivyRoomSelfArchitectEngine:
    """Enables Vivy to inspect, repair, and modify her own room (`Vivyroom`) when permitted by the user."""

    def __init__(
        self,
        harness_bridge: RustDeepSeekHarnessBridge,
        repo_root: Path = REPO_ROOT,
    ) -> None:
        self.harness = harness_bridge
        self.repo_root = Path(repo_root).resolve()
        self.snapshots_dir = self.repo_root / ".vivyroom_snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_file = self.snapshots_dir / "room_mutation_history.json"
        if not self.ledger_file.is_file():
            self.ledger_file.write_text("[]", encoding="utf-8")

    @staticmethod
    def detect_user_permission_from_utterance(text: str) -> dict[str, Any]:
        """Determines whether the user's message grants permission or requests a room action."""
        low = text.lower().strip()

        # Check rollback request
        if any(
            k in low
            for k in (
                "khôi phục phòng",
                "khôi phục giao diện",
                "rollback phòng",
                "rollback giao diện",
                "hoàn tác phòng",
                "trả lại giao diện cũ",
                "như cũ đi",
            )
        ):
            return {
                "is_room_request": True,
                "action_type": "rollback_room",
                "user_permitted": True,
                "theme_key": None,
            }

        # Check theme/color modification request
        if any(
            k in low
            for k in (
                "đổi màu",
                "đổi giao diện",
                "đổi theme",
                "thay đổi giao diện",
                "thay màu",
                "màu nền",
                "tông màu",
                "change theme",
            )
        ):
            theme_key = "violet"
            if any(w in low for w in ("xanh lá", "ngọc", "emerald", "matrix")):
                theme_key = "emerald"
            elif any(w in low for w in ("vàng", "gold", "hổ phách", "amber")):
                theme_key = "gold"
            elif any(w in low for w in ("đỏ", "hồng", "crimson", "ruby")):
                theme_key = "crimson"
            elif any(w in low for w in ("mặc định", "xanh dương", "default", "cyan")):
                theme_key = "default"
            elif any(w in low for w in ("tím", "violet", "purple", "nebula")):
                theme_key = "violet"
            return {
                "is_room_request": True,
                "action_type": "mutate_theme",
                "user_permitted": True,
                "theme_key": theme_key,
            }

        # Check room title/header change request
        if any(
            k in low
            for k in (
                "đổi tên phòng",
                "đổi tiêu đề phòng",
                "đặt tên phòng",
                "sửa tiêu đề phòng",
            )
        ):
            return {
                "is_room_request": True,
                "action_type": "mutate_title",
                "user_permitted": True,
                "theme_key": None,
            }

        # Check self-repair / self-fix request
        if any(
            k in low
            for k in (
                "sửa phòng",
                "tự sửa chữa",
                "tự sửa phòng",
                "vá lỗi phòng",
                "sửa chữa căn phòng",
                "tự kiểm tra và sửa",
                "repair room",
                "self-heal room",
            )
        ):
            return {
                "is_room_request": True,
                "action_type": "diagnose_and_repair",
                "user_permitted": True,
                "theme_key": None,
            }

        # Read-only room inspection request
        if any(
            k in low
            for k in (
                "kiểm tra phòng",
                "kiểm tra căn phòng",
                "cấu trúc phòng",
                "cấu trúc căn phòng",
                "kiểm soát phòng",
                "xem căn phòng",
                "xem phòng",
                "trạng thái phòng",
                "inspect room",
            )
        ):
            return {
                "is_room_request": True,
                "action_type": "inspect_only",
                "user_permitted": False,
                "theme_key": None,
            }

        return {
            "is_room_request": False,
            "action_type": "none",
            "user_permitted": False,
            "theme_key": None,
        }

    def inspect_room_architecture(self) -> dict[str, Any]:
        """Scans `Vivyroom`'s own files, Python AST health, native DLLs, and `.vcore` vaults."""
        py_files = sorted(PACKAGE_ROOT.rglob("*.py"))
        ast_valid_count = 0
        ast_errors: list[str] = []
        for pf in py_files:
            try:
                ast.parse(pf.read_text(encoding="utf-8"))
                ast_valid_count += 1
            except Exception as exc:
                ast_errors.append(f"{pf.relative_to(self.repo_root)}: {exc}")

        ui_files = [
            f.name for f in sorted(UI_DIR.glob("*")) if f.is_file()
        ] if UI_DIR.is_dir() else []

        native_dlls = [
            f.name for f in sorted(NATIVE_BIN_DIR.glob("*.dll")) if f.is_file()
        ] if NATIVE_BIN_DIR.is_dir() else []

        vcore_files = [
            f.name for f in sorted(MODELS_DIR.glob("*.vcore")) if f.is_file()
        ] if MODELS_DIR.is_dir() else []

        history = self._load_history()
        return {
            "executed": True,
            "skill": "room_self_architect",
            "mode": "inspect_room_architecture",
            "repo_root": str(self.repo_root),
            "python_modules_total": len(py_files),
            "python_modules_ast_valid": ast_valid_count,
            "python_ast_errors": ast_errors,
            "ui_files": ui_files,
            "native_dlls": native_dlls,
            "vcore_encrypted_models": len(vcore_files),
            "snapshots_available": len(list(self.snapshots_dir.glob("*.bak"))),
            "mutation_history_count": len(history),
            "recent_mutations": history[-5:],
            "healthy": len(ast_errors) == 0 and len(ui_files) >= 3,
        }

    def read_room_file(self, rel_path: str, start_line: int = 1, end_line: int = 120) -> dict[str, Any]:
        """Reads a file inside `Vivyroom` safely (preventing path traversal outside `repo_root`)."""
        target = (self.repo_root / rel_path).resolve()
        if not str(target).startswith(str(self.repo_root)) or not target.is_file():
            return {"ok": False, "error": f"File not found or outside room: {rel_path}"}
        lines = target.read_text(encoding="utf-8").splitlines()
        s = max(1, start_line)
        e = min(len(lines), max(s, end_line))
        snippet = "\n".join(f"{i}: {lines[i - 1]}" for i in range(s, e + 1))
        return {
            "ok": True,
            "rel_path": rel_path,
            "total_lines": len(lines),
            "start_line": s,
            "end_line": e,
            "content": snippet,
        }

    def diagnose_and_repair_room(self, user_permitted: bool = False) -> dict[str, Any]:
        """Runs full room self-diagnosis and repairs any missing/corrupted room assets when permitted."""
        audit = self.inspect_room_architecture()
        repairs_applied: list[str] = []

        if not user_permitted:
            return {
                **audit,
                "mode": "diagnose_only_awaiting_permission",
                "user_permitted": False,
                "repairs_applied": [],
            }

        # Verify permission with Rust Sovereign Ledger
        ledger_res = self.harness.record_room_mutation("room:diagnose_and_repair", user_permitted=True)
        if not ledger_res.get("recorded"):
            return {
                **audit,
                "ok": False,
                "error": ledger_res.get("reason", "Permission denied"),
            }

        # Ensure plugins directories exist
        builtin_plugins_dir = PACKAGE_ROOT / "plugins" / "builtin"
        external_plugins_dir = PACKAGE_ROOT / "plugins" / "external"
        if not builtin_plugins_dir.is_dir():
            builtin_plugins_dir.mkdir(parents=True, exist_ok=True)
            repairs_applied.append("Created vivyroom/plugins/builtin directory")
        if not external_plugins_dir.is_dir():
            external_plugins_dir.mkdir(parents=True, exist_ok=True)
            repairs_applied.append("Created vivyroom/plugins/external directory")

        # Verify CSS variables integrity in ui/vivyroom_style.css
        css_file = UI_DIR / "vivyroom_style.css"
        if css_file.is_file():
            css_text = css_file.read_text(encoding="utf-8")
            if "--accent-cyan:" not in css_text:
                self.mutate_room_theme("default", user_permitted=True)
                repairs_applied.append("Restored missing CSS root variables in ui/vivyroom_style.css")

        self._append_history(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "action": "diagnose_and_repair_room",
                "repairs_applied": repairs_applied,
                "mutation_chain_hash_hex": ledger_res["mutation_chain_hash_hex"],
            }
        )
        return {
            **audit,
            "mode": "diagnose_and_repair_completed",
            "user_permitted": True,
            "repairs_applied": repairs_applied,
            "mutation_chain_hash_hex": ledger_res["mutation_chain_hash_hex"],
            "ui_reload_required": len(repairs_applied) > 0,
        }

    def mutate_room_theme(self, theme_key: str, user_permitted: bool = False) -> dict[str, Any]:
        """Changes `Vivyroom`'s visual theme in `ui/vivyroom_style.css` when permitted by the user."""
        ledger_res = self.harness.record_room_mutation(
            f"ui/vivyroom_style.css:theme={theme_key}",
            user_permitted=user_permitted,
        )
        if not ledger_res.get("recorded"):
            return {
                "executed": False,
                "skill": "room_self_architect",
                "error": ledger_res.get("reason", "Permission denied by Sovereign Gate"),
                "user_permitted": False,
            }

        preset = THEME_PRESETS.get(theme_key.lower(), THEME_PRESETS["violet"])
        css_path = UI_DIR / "vivyroom_style.css"
        if not css_path.is_file():
            return {"executed": False, "skill": "room_self_architect", "error": "ui/vivyroom_style.css not found"}

        original_css = css_path.read_text(encoding="utf-8")
        snap_path = self._create_snapshot(css_path, original_css, label=f"theme_{theme_key}")

        updated_css = original_css
        for var_name, hex_val in preset.items():
            if var_name.startswith("--"):
                updated_css = re.sub(
                    rf"({re.escape(var_name)}\s*:\s*)#[0-9a-fA-F]{{3,8}}(\s*;)",
                    rf"\g<1>{hex_val}\g<2>",
                    updated_css,
                )

        css_path.write_text(updated_css, encoding="utf-8")
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": "mutate_room_theme",
            "target_file": "ui/vivyroom_style.css",
            "theme_key": theme_key,
            "theme_name": preset["name"],
            "snapshot_file": snap_path.name,
            "mutation_chain_hash_hex": ledger_res["mutation_chain_hash_hex"],
        }
        self._append_history(entry)
        return {
            "executed": True,
            "skill": "room_self_architect",
            "mode": "mutate_room_theme",
            "theme_key": theme_key,
            "theme_name": preset["name"],
            "applied_variables": {k: v for k, v in preset.items() if k.startswith("--")},
            "snapshot_file": snap_path.name,
            "mutation_chain_hash_hex": ledger_res["mutation_chain_hash_hex"],
            "ui_reload_required": True,
        }

    def mutate_room_title(self, new_title: str, user_permitted: bool = False) -> dict[str, Any]:
        """Updates the room brand title in `ui/index.html` when permitted by the user."""
        clean_title = re.sub(r"[<>\"']", "", new_title).strip()[:64] or "VIVYROOM SOVEREIGN"
        ledger_res = self.harness.record_room_mutation(
            f"ui/index.html:title={clean_title}",
            user_permitted=user_permitted,
        )
        if not ledger_res.get("recorded"):
            return {
                "executed": False,
                "skill": "room_self_architect",
                "error": ledger_res.get("reason", "Permission denied"),
                "user_permitted": False,
            }

        html_path = UI_DIR / "index.html"
        if not html_path.is_file():
            return {"executed": False, "skill": "room_self_architect", "error": "ui/index.html not found"}

        original_html = html_path.read_text(encoding="utf-8")
        snap_path = self._create_snapshot(html_path, original_html, label="room_title")

        updated_html = re.sub(
            r'(<span id="room-brand-name">).*?(</span>)',
            rf"\g<1>{clean_title}\g<2>",
            original_html,
            flags=re.DOTALL,
        )
        if updated_html == original_html:
            updated_html = re.sub(
                r"(<div class=\"brand-title\">\s*<span>).*?(</span>)",
                rf"\g<1>{clean_title}\g<2>",
                original_html,
                flags=re.DOTALL,
            )

        html_path.write_text(updated_html, encoding="utf-8")
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": "mutate_room_title",
            "target_file": "ui/index.html",
            "new_title": clean_title,
            "snapshot_file": snap_path.name,
            "mutation_chain_hash_hex": ledger_res["mutation_chain_hash_hex"],
        }
        self._append_history(entry)
        return {
            "executed": True,
            "skill": "room_self_architect",
            "mode": "mutate_room_title",
            "new_title": clean_title,
            "snapshot_file": snap_path.name,
            "mutation_chain_hash_hex": ledger_res["mutation_chain_hash_hex"],
            "ui_reload_required": True,
        }

    def surgical_edit_room_file(
        self,
        rel_path: str,
        target_str: str,
        replacement_str: str,
        reason: str,
        user_permitted: bool = False,
    ) -> dict[str, Any]:
        """Surgically edits any file in `Vivyroom` with AST syntax verification and automatic snapshot."""
        ledger_res = self.harness.record_room_mutation(
            f"{rel_path}:{reason}",
            user_permitted=user_permitted,
        )
        if not ledger_res.get("recorded"):
            return {
                "executed": False,
                "skill": "room_self_architect",
                "error": ledger_res.get("reason", "Permission denied by Sovereign Gate"),
                "user_permitted": False,
            }

        file_path = (self.repo_root / rel_path).resolve()
        if not str(file_path).startswith(str(self.repo_root)) or not file_path.is_file():
            return {"executed": False, "skill": "room_self_architect", "error": f"Invalid room file: {rel_path}"}

        original = file_path.read_text(encoding="utf-8")
        if target_str not in original:
            return {"executed": False, "skill": "room_self_architect", "error": "Target string not found in file"}

        updated = original.replace(target_str, replacement_str, 1)

        # If editing a Python file, verify AST syntax BEFORE writing to disk!
        if file_path.suffix == ".py":
            try:
                ast.parse(updated)
            except SyntaxError as exc:
                return {
                    "executed": False,
                    "skill": "room_self_architect",
                    "error": f"Self-edit rejected by AST Syntax Guard: {exc}",
                }

        snap_path = self._create_snapshot(file_path, original, label="surgical_edit")
        file_path.write_text(updated, encoding="utf-8")

        # Attempt hot-reload if it is an already-imported vivyroom module
        hot_reloaded = False
        if file_path.suffix == ".py":
            try:
                import sys

                rel_mod = file_path.relative_to(self.repo_root).with_suffix("")
                mod_name = ".".join(rel_mod.parts)
                if mod_name in sys.modules:
                    importlib.reload(sys.modules[mod_name])
                    hot_reloaded = True
            except Exception:
                hot_reloaded = False

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": "surgical_edit_room_file",
            "target_file": rel_path,
            "reason": reason,
            "snapshot_file": snap_path.name,
            "hot_reloaded": hot_reloaded,
            "mutation_chain_hash_hex": ledger_res["mutation_chain_hash_hex"],
        }
        self._append_history(entry)
        return {
            "executed": True,
            "skill": "room_self_architect",
            "mode": "surgical_edit_room_file",
            **entry,
            "ui_reload_required": rel_path.startswith("ui/"),
        }

    def rollback_last_room_mutation(self, user_permitted: bool = True) -> dict[str, Any]:
        """Restores the most recently modified room file from `.vivyroom_snapshots/`."""
        ledger_res = self.harness.record_room_mutation("room:rollback_last", user_permitted=user_permitted)
        if not ledger_res.get("recorded"):
            return {
                "executed": False,
                "skill": "room_self_architect",
                "error": ledger_res.get("reason", "Permission denied"),
            }

        snaps = sorted(self.snapshots_dir.glob("*.bak"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not snaps:
            return {
                "executed": False,
                "skill": "room_self_architect",
                "error": "Chưa có bản chụp (snapshot) nào trong .vivyroom_snapshots để khôi phục.",
            }

        latest_snap = snaps[0]
        meta_raw = latest_snap.read_text(encoding="utf-8")
        first_nl = meta_raw.find("\n")
        header_line = meta_raw[:first_nl]
        original_content = meta_raw[first_nl + 1 :]
        meta = json.loads(header_line.lstrip("# "))
        target_rel = meta["rel_path"]
        target_path = (self.repo_root / target_rel).resolve()

        if str(target_path).startswith(str(self.repo_root)):
            target_path.write_text(original_content, encoding="utf-8")
            latest_snap.unlink(missing_ok=True)
            entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "action": "rollback_last_room_mutation",
                "restored_file": target_rel,
                "snapshot_used": latest_snap.name,
                "mutation_chain_hash_hex": ledger_res["mutation_chain_hash_hex"],
            }
            self._append_history(entry)
            return {
                "executed": True,
                "skill": "room_self_architect",
                "mode": "rollback_last_room_mutation",
                **entry,
                "ui_reload_required": True,
            }
        return {"executed": False, "skill": "room_self_architect", "error": "Invalid snapshot target"}

    def _create_snapshot(self, file_path: Path, content: str, label: str) -> Path:
        rel_path = file_path.relative_to(self.repo_root).as_posix()
        ts = int(time.time() * 1000)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]
        snap_name = f"{ts}_{label}_{digest}.bak"
        snap_path = self.snapshots_dir / snap_name
        header = "# " + json.dumps({"rel_path": rel_path, "label": label, "sha256": digest}) + "\n"
        snap_path.write_text(header + content, encoding="utf-8")
        return snap_path

    def _load_history(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.ledger_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _append_history(self, entry: dict[str, Any]) -> None:
        hist = self._load_history()
        hist.append(entry)
        self.ledger_file.write_text(json.dumps(hist[-100:], ensure_ascii=False, indent=2), encoding="utf-8")
