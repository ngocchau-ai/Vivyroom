/**
 * Vivyroom v2.0 — Sovereign All-Purpose Agent UI Controller
 * - Keeps chat stream 100% natural and human-grade (zero telemetry/spec clutter inside chat bubbles)
 * - Isolates all technical telemetry into the collapsible X-Ray Drawer (Alt+X)
 * - Supports live hot-reload when Vivy modifies her own room theme, title, or layout!
 */

let isBusy = false;
let promptSaveTimer = null;

function showToast(msg, type = "success") {
  const banner = document.getElementById("status-toast");
  if (!banner) return;
  banner.textContent = msg;
  banner.className = `status-toast ${type}`;
  setTimeout(() => {
    banner.className = "status-toast";
  }, 3600);
}

function hotReloadRoomStylesheet(turn = null) {
  const linkEl = document.getElementById("vivyroom-stylesheet");
  if (linkEl) {
    linkEl.href = `/vivyroom_style.css?t=${Date.now()}`;
  }
  const toolOut = turn && turn.tool_output ? turn.tool_output : null;
  if (toolOut && toolOut.new_title) {
    const titleEl = document.getElementById("room-brand-name");
    if (titleEl) titleEl.textContent = toolOut.new_title;
  }
}

function updateTopTelemetry(status, turn = null) {
  if (!status) return;
  const ram = status.ram_telemetry || {};
  const ramMb = Number(ram.process_ram_mb || 0).toFixed(1);

  const ramValEl = document.getElementById("ram-value");
  if (ramValEl) ramValEl.textContent = `RAM: ${ramMb} MB`;

  const plugins = status.mounted_plugins || {};
  const plugPill = document.getElementById("active-plugins-pill");
  if (plugPill && plugins.total_mounted_plugins !== undefined) {
    plugPill.textContent = `🧩 ${plugins.total_mounted_plugins} Plugins`;
  }

  const roomStatus = status.room_self_architect_status || {};
  const snapPill = document.getElementById("room-snapshots-pill");
  if (snapPill && roomStatus.snapshots_available !== undefined) {
    snapPill.textContent = `Snapshots: ${roomStatus.snapshots_available}`;
  }

  if (turn) {
    const latEl = document.getElementById("turn-latency");
    if (latEl) latEl.textContent = `⚡ ${turn.total_turn_latency_ms} ms`;

    const moe = turn.moe_routing_telemetry || {};
    const moePill = document.getElementById("active-moe-pill");
    if (moePill && moe.active_expert_codes) {
      moePill.textContent = `MoE: ${moe.active_expert_codes.map((c) => c.slice(0, 2)).join("+")}`;
    }

    const dsh = turn.deepseek_harness_telemetry || {};
    const hPill = document.getElementById("harness-pill");
    if (hPill && dsh.mtp_speedup_factor) {
      hPill.textContent = `Rust Harness: MTP ${dsh.mtp_speedup_factor}x | Engram Hits: ${dsh.engram_hits}`;
    }

    if (turn.ui_reload_required) {
      hotReloadRoomStylesheet(turn);
      showToast("Vivy vừa tự cập nhật trực tiếp giao diện căn phòng!", "success");
    }
  }

  if (status.trader_system_prompt) {
    const sp = status.trader_system_prompt;
    const editorEl = document.getElementById("system-prompt-editor");
    if (editorEl && document.activeElement !== editorEl) {
      editorEl.value = sp.raw_prompt || "";
    }
  }
}

function formatSimpleMarkdown(text) {
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br />");
}

function appendChatMessage(role, text, turn = null) {
  const stream = document.getElementById("chat-stream");
  if (!stream) return;

  const div = document.createElement("div");
  div.className = `msg-card ${role === "user" ? "msg-user" : "msg-vivy"}`;

  const bodyP = document.createElement("div");
  bodyP.innerHTML = formatSimpleMarkdown(text);
  div.appendChild(bodyP);

  if (turn) {
    const toolOut = turn.tool_output || null;

    // Render SVG Chart inline only when quantitative chart analysis was requested
    if (toolOut && toolOut.svg_chart) {
      const chartBox = document.createElement("div");
      chartBox.className = "inline-svg-chart";
      chartBox.innerHTML = toolOut.svg_chart;
      div.appendChild(chartBox);
    }

    if (turn.voice_wav_base64) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.style.marginTop = "8px";
      audio.style.height = "28px";
      audio.style.width = "240px";
      audio.src = "data:audio/wav;base64," + turn.voice_wav_base64;
      div.appendChild(audio);
      audio.play().catch(() => {});
    }
  }

  stream.appendChild(div);
  stream.scrollTop = stream.scrollHeight;
}

async function sendChatMessage() {
  if (isBusy) return;
  const inputEl = document.getElementById("chat-input");
  if (!inputEl) return;

  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = "";
  appendChatMessage("user", text);

  isBusy = true;
  inputEl.placeholder = "Vivy đang suy nghĩ và thực thi...";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        synthesize_voice: false,
      }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    appendChatMessage("vivy", data.turn.response_text, data.turn);
    updateTopTelemetry(data.status, data.turn);
  } catch (err) {
    showToast(`Lỗi kết nối cục bộ: ${err.message}`, "error");
    appendChatMessage("vivy", `Em gặp chút gián đoạn khi xử lý yêu cầu vừa rồi (${err.message}). Anh thử gửi lại giúp em nhé!`);
  } finally {
    isBusy = false;
    inputEl.placeholder =
      "Trò chuyện tự nhiên với Vivy (VD: Chào Vivy, Đổi giao diện phòng sang màu tím, Phân tích XAUUSD, Nạp skill từ GitHub...)...";
    inputEl.focus();
  }
}

async function autoSaveSystemPrompt() {
  const editorEl = document.getElementById("system-prompt-editor");
  if (!editorEl) return;
  try {
    const res = await fetch("/api/system-prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt_text: editorEl.value, reset: false }),
    });
    const data = await res.json();
    if (data.ok) {
      updateTopTelemetry(data.status);
      showToast("Đã lưu thiết lập vai trò mới", "success");
    }
  } catch (err) {
    showToast(`Không thể lưu thiết lập: ${err.message}`, "error");
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  const inputEl = document.getElementById("chat-input");
  const editorEl = document.getElementById("system-prompt-editor");
  const xrayDrawer = document.getElementById("xray-drawer");

  window.addEventListener("keydown", (e) => {
    if (e.altKey && (e.key === "x" || e.key === "X")) {
      e.preventDefault();
      if (xrayDrawer) xrayDrawer.open = !xrayDrawer.open;
    }
  });

  if (inputEl) {
    inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });
  }

  if (editorEl) {
    editorEl.addEventListener("blur", () => autoSaveSystemPrompt());
    editorEl.addEventListener("input", () => {
      if (promptSaveTimer) clearTimeout(promptSaveTimer);
      promptSaveTimer = setTimeout(() => autoSaveSystemPrompt(), 1200);
    });
  }

  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.ok) {
      updateTopTelemetry(data.status);
    }
  } catch (err) {
    showToast(`Status check failed: ${err.message}`, "error");
  }
});
