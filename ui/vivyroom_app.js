/**
 * Vivyroom — Button-Free Single-Chat Autonomous Controller (OranPi × PocketFlow Edge Shell)
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

function updateTopTelemetry(status, turn = null) {
  if (!status) return;
  const ram = status.ram_telemetry || {};
  const ramMb = Number(ram.process_ram_mb || 0).toFixed(1);
  const limitMb = Number(ram.device_ram_limit_mb || 2048);

  const ramValEl = document.getElementById("ram-value");
  if (ramValEl) ramValEl.textContent = `RAM: ${ramMb} / ${limitMb} MB`;

  const pageInEl = document.getElementById("moe-page-events");
  if (pageInEl) {
    pageInEl.textContent = `MoE Pager IN:${ram.page_in_events || 0} | OUT:${ram.page_out_events || 0}`;
  }

  const vcoreEl = document.getElementById("vcore-compliance");
  if (vcoreEl && status.vcore_audit) {
    vcoreEl.textContent = status.vcore_audit.proprietary_3_in_1_compliance
      ? "VCORE3 (3-IN-1 LOCKED)"
      : "UNVERIFIED";
  }

  const oranpiEl = document.getElementById("oranpi-badge");
  if (oranpiEl && status.oranpi_edge_profile) {
    oranpiEl.textContent = `OranPi-PocketFlow (${status.oranpi_edge_profile.os_family}/${status.oranpi_edge_profile.arch})`;
  }

  if (turn) {
    const latEl = document.getElementById("turn-latency");
    if (latEl) latEl.textContent = `⚡ ${turn.total_turn_latency_ms} ms`;

    const moe = turn.moe_routing_telemetry || {};
    const moePill = document.getElementById("active-moe-pill");
    if (moePill && moe.active_expert_codes) {
      moePill.textContent = `MoE: ${moe.active_expert_codes.map((c) => c.slice(0, 2)).join(" + ")}`;
    }
  }

  if (status.trader_system_prompt) {
    const sp = status.trader_system_prompt;
    const editorEl = document.getElementById("system-prompt-editor");
    const badgeEl = document.getElementById("prompt-role-badge");
    if (editorEl && document.activeElement !== editorEl) {
      editorEl.value = sp.raw_prompt || "";
    }
    if (badgeEl) {
      badgeEl.textContent = `${sp.role.slice(0, 26)} • ANTI-BIAS: ${sp.anti_human_bias_shield ? "ON" : "OFF"}`;
    }
  }
}

function appendChatMessage(role, text, turn = null) {
  const stream = document.getElementById("chat-stream");
  if (!stream) return;

  const div = document.createElement("div");
  div.className = `msg-card ${role === "user" ? "msg-user" : "msg-vivy"}`;

  const bodyP = document.createElement("div");
  bodyP.textContent = text;
  div.appendChild(bodyP);

  if (turn) {
    const comp = turn.compiler_telemetry || {};
    const moe = turn.moe_routing_telemetry || {};
    const hw = turn.hardware_telemetry || {};
    const toolOut = turn.tool_output || null;

    // Render Zero-Token SVG Chart inline inside the chat bubble whenever Finance-3B-MoE executed
    if (toolOut && toolOut.svg_chart) {
      const chartBox = document.createElement("div");
      chartBox.className = "inline-svg-chart";
      chartBox.innerHTML = toolOut.svg_chart;
      div.appendChild(chartBox);
    }

    const meta = document.createElement("div");
    meta.className = "msg-meta";
    const autoChain =
      toolOut && toolOut.autonomous_skills_executed
        ? `<span class="meta-tag" style="border-color:#10b981;color:#34d399">🔗 Auto-Chain: ${toolOut.autonomous_skills_executed.join(" → ")}</span>`
        : toolOut && toolOut.skill
          ? `<span class="meta-tag" style="border-color:#10b981;color:#34d399">🔧 Auto-Skill: ${toolOut.skill}</span>`
          : "";
    const repairs =
      (comp.self_healed_repairs || []).length > 0
        ? `<span class="meta-tag" style="border-color:#f59e0b;color:#fcd34d">🛡️ ${comp.self_healed_repairs.join(",")}</span>`
        : "";

    meta.innerHTML = `
      <span class="meta-tag">⚡ ${turn.total_turn_latency_ms} ms</span>
      <span class="meta-tag">🧠 ${(moe.active_expert_codes || []).join(" + ")}</span>
      <span class="meta-tag">💾 ${hw.process_ram_mb} / 2048 MB</span>
      ${autoChain}
      ${repairs}
    `;
    div.appendChild(meta);

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
  inputEl.placeholder = "Vivy đang tư duy & tự động thực thi kỹ năng (<15ms)...";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        synthesize_voice: true,
      }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    appendChatMessage("vivy", data.turn.response_text, data.turn);
    updateTopTelemetry(data.status, data.turn);
  } catch (err) {
    showToast(`Communication error: ${err.message}`, "error");
    appendChatMessage("vivy", `[Self-Healed Local Guard] ${err.message}`);
  } finally {
    isBusy = false;
    inputEl.placeholder =
      "Nhập yêu cầu tự nhiên và nhấn Enter (VD: Phân tích XAUUSD kèm tin ForexFactory & RoboForex, học YouTube Kalman, tải model HF, hoặc /prompt ...)...";
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
      showToast(`Đã tự động lưu SYSTEM_PROMPT.md (${data.system_prompt.prompt_sha256})`, "success");
    }
  } catch (err) {
    showToast(`Auto-save System Prompt failed: ${err.message}`, "error");
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  const inputEl = document.getElementById("chat-input");
  const editorEl = document.getElementById("system-prompt-editor");

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
