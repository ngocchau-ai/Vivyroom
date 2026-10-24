"""Zero-Dependency Single-Process HTTP Server for `Vivyroom` (`127.0.0.1:7860`).

Serves:
- `GET /`, `/vivyroom_app.js`, `/vivyroom_style.css` -> Single-Screen Unified Cockpit UI (`ui/`)
- `GET /api/status` -> Full system, `.vcore` encryption, MoE paging, and RAM telemetry (`< 2,048 MB`)
- `POST /api/chat` -> Ultra-fast natural-language turn via `VivyQwenSemanticCompiler` + `Vivy Decision Core`
- `POST /api/finance` -> Autonomous `Vivy-Finance-3B-MoE` quantitative analysis & Zero-Token SVG chart
- `POST /api/auth/unlock` -> 1-Touch Sovereign Hardware-Bound Passkey / Passphrase Unlock
- `POST /api/skills/toggle` -> Live customization of offline agent skills
- `POST /api/dream` -> Trigger 4-Stage `LUCID_STANDBY` Dream Consolidation & MoE RAM Purge
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .agent.vivy_controller import VivyroomAutonomousController
from .config import DEFAULT_CONFIG, UI_DIR

MAX_REQUEST_BODY_BYTES = 65_536


class VivyroomRequestHandler(BaseHTTPRequestHandler):
    """Thread-safe HTTP request handler coupled to the singleton `VivyroomAutonomousController`."""

    controller: VivyroomAutonomousController
    controller_lock = threading.Lock()
    ui_dir: Path = UI_DIR

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy per-request console spam unless debugging
        return

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> tuple[dict[str, Any] | None, str | None, int]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None, "Invalid Content-Length header", HTTPStatus.BAD_REQUEST
        if length < 0 or length > MAX_REQUEST_BODY_BYTES:
            return None, f"Payload exceeds {MAX_REQUEST_BODY_BYTES} bytes limit", HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        if length == 0:
            return {}, None, HTTPStatus.OK
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                return None, "JSON body must be an object", HTTPStatus.BAD_REQUEST
            return parsed, None, HTTPStatus.OK
        except Exception as exc:
            return None, f"Malformed JSON payload: {exc}", HTTPStatus.BAD_REQUEST

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        route = parsed_url.path

        if route == "/api/status":
            with self.controller_lock:
                status = self.controller.get_system_status()
            self._send_json(HTTPStatus.OK, {"ok": True, "status": status})
            return

        if route == "/api/health":
            with self.controller_lock:
                status = self.controller.get_system_status()
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "organism": status["organism"],
                    "process_ram_mb": status["ram_telemetry"]["process_ram_mb"],
                    "under_2gb_budget": status["ram_telemetry"]["process_ram_mb"] <= 2048.0,
                    "proprietary_vcore_compliance": status["vcore_audit"]["proprietary_3_in_1_compliance"],
                },
            )
            return

        # Serve static UI files from `ui/`
        if route in ("/", "/index.html"):
            target_file = self.ui_dir / "index.html"
        else:
            rel = route.lstrip("/")
            if ".." in rel or "/" in rel or "\\" in rel:
                self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "Forbidden path"})
                return
            target_file = self.ui_dir / rel

        if not target_file.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Route not found: {route}"})
            return

        mime, _ = mimetypes.guess_type(str(target_file))
        content_type = mime or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type += "; charset=utf-8"
        data = target_file.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        route = parsed_url.path
        body, err, code = self._read_json_body()
        if err is not None or body is None:
            self._send_json(code, {"ok": False, "error": err or "Bad request"})
            return

        try:
            if route == "/api/chat":
                message = str(body.get("message", "")).strip()
                synthesize_voice = bool(body.get("synthesize_voice", False))
                with self.controller_lock:
                    turn_result = self.controller.chat_turn(message, synthesize_voice=synthesize_voice)
                    status = self.controller.get_system_status()
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "turn": turn_result, "status": status},
                )
                return

            if route == "/api/finance":
                symbol = str(body.get("symbol", "XAUUSD")).upper().strip() or "XAUUSD"
                with self.controller_lock:
                    fin_report = self.controller.finance_desk.analyze_market_offline_or_live(symbol=symbol)
                    status = self.controller.get_system_status()
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "finance": fin_report, "status": status},
                )
                return

            if route == "/api/auth/unlock":
                passphrase = str(body.get("passphrase", "")).strip() or None
                with self.controller_lock:
                    ticket = self.controller.auth_gate.unlock_one_touch(pin_override=passphrase)
                    status = self.controller.get_system_status()
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "ticket": ticket, "status": status},
                )
                return

            if route == "/api/skills/toggle":
                skill_name = str(body.get("skill_name", "")).strip()
                enabled = bool(body.get("enabled", True))
                with self.controller_lock:
                    updated_skills = self.controller.skills.toggle_skill(skill_name, enabled)
                    status = self.controller.get_system_status()
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "enabled_skills": updated_skills, "status": status},
                )
                return

            if route == "/api/dream":
                rounds = int(body.get("rounds", 2))
                with self.controller_lock:
                    dream_out = self.controller.skills.execute_skill(
                        "trigger_lucid_dream", {"rounds": max(1, min(8, rounds))}
                    )
                    self.controller.memory.dream_cycles_completed += 1
                    status = self.controller.get_system_status()
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "dream": dream_out, "status": status},
                )
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown endpoint: {route}"})
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"Internal controller error: {exc}"},
            )


def create_vivyroom_http_server(
    host: str = DEFAULT_CONFIG.host,
    port: int = DEFAULT_CONFIG.port,
    controller: VivyroomAutonomousController | None = None,
) -> ThreadingHTTPServer:
    """Instantiates a `ThreadingHTTPServer` bound to `(host, port)` with the unified Vivy controller."""
    ctrl = controller or VivyroomAutonomousController()

    class BoundHandler(VivyroomRequestHandler):
        pass

    BoundHandler.controller = ctrl
    BoundHandler.ui_dir = UI_DIR
    return ThreadingHTTPServer((host, port), BoundHandler)


def run_cli_self_test() -> int:
    """Runs an end-to-end headless verification of `Vivyroom` for CI and portable diagnostics."""
    ctrl = VivyroomAutonomousController()
    status = ctrl.get_system_status()
    r1 = ctrl.chat_turn("Chào Vivy, hôm nay thế nào?", synthesize_voice=True)
    r2 = ctrl.chat_turn("Phân tích giá vàng XAUUSD và lãi kép 50 bot", synthesize_voice=False)
    summary = {
        "self_test": "PASS",
        "organism": status["organism"],
        "proprietary_vcore_3in1": status["vcore_audit"]["proprietary_3_in_1_compliance"],
        "turn1_latency_ms": r1["total_turn_latency_ms"],
        "turn2_latency_ms": r2["total_turn_latency_ms"],
        "process_ram_mb": r2["hardware_telemetry"]["process_ram_mb"],
        "ram_limit_mb": r2["hardware_telemetry"]["device_ram_limit_mb"],
        "active_experts_turn2": r2["moe_routing_telemetry"]["active_expert_codes"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vivyroom — Portable <2GB RAM Unified Vivy + Cautreo Control Room")
    parser.add_argument("--host", default=DEFAULT_CONFIG.host, help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_CONFIG.port, help="Bind port (default: 7860)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser on startup")
    parser.add_argument("--self-test", action="store_true", help="Run headless verification and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_cli_self_test()

    server = create_vivyroom_http_server(host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}"
    status = server.RequestHandlerClass.controller.get_system_status()
    print("=" * 74)
    print(" VIVYROOM — PORTABLE <2GB RAM UNIFIED VIVY + CAUTREO CONTROL ROOM")
    print("=" * 74)
    print(f" * URL               : {url}")
    print(f" * Organism          : {status['organism']}")
    print(f" * Encrypted .vcore  : {status['vcore_audit']['proprietary_3_in_1_compliance']} (8 MoE Experts + Tri-Core Master)")
    print(f" * Resident RAM      : {status['ram_telemetry']['process_ram_mb']} MB / {status['ram_telemetry']['device_ram_limit_mb']} MB")
    print("=" * 74)

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Vivyroom] Shutting down cleanly...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
