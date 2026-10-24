/**
 * Vivyroom Single-Screen Unified Cockpit Controller (100% Offline, Zero External Deps)
 */

const EXPERT_META = [
  { id: 0, code: "E0_QUANTUM_MICROSTRUCTURE", desc: "Quantum SVD Density rho, RMT, OFI/VPIN" },
  { id: 1, code: "E1_OU_KALMAN_VELOCITY", desc: "Ornstein-Uhlenbeck & 1D Kalman Velocity" },
  { id: 2, code: "E2_50BOT_4PILLAR_COMPOUND", desc: "50 RoboForex Bots & Compounding C_mult" },
  { id: 3, code: "E3_MULTI_ASSET_XAUUSD", desc: "XAUUSD/EURUSD/USDJPY Sovereign Matrix" },
  { id: 4, code: "E4_MTF_RANGE_SVG_VISUAL", desc: "M15/M30/H1 Range & Zero-Token SVG Chart" },
  { id: 5, code: "E5_SWE_CODE_AGENT_TOOLS", desc: "Python AST Math, Workspace I/O & Tools" },
  { id: 6, code: "E6_COMPOSITIONAL_DIALOGUE_VOICE", desc: "Qwen-Decomposed Compiler & 24kHz Voice" },
  { id: 7, code: "E7_VM11_DREAM_EVOLUTION", desc: "VM-11 Dampening & LUCID_STANDBY Dream" },
];

let isBusy = false;

function showToast(msg, type = "success") {
  const banner = document.getElementById("status-toast");
  if (!banner) return;
  banner.textContent = msg;
  banner.className = `status-toast ${type}`;
  setTimeout(() => {
    banner.className = "status-toast";
  }, 4200);
}

function renderMoEGrid(activeCodes = [], probs = {}) {
  const container = document.getElementById("moe-grid");
  if (!container) return;
  container.innerHTML = EXPERT_META.map((exp) => {
    const isActive = activeCodes.includes(exp.code);
    const probPct = probs[exp.code] !== undefined ? (probs[exp.code] * 100).toFixed(1) + "%" : (isActive ? "TOP-2" : "DISK");
    return `
      <div class="expert-box ${isActive ? "active" : ""}">
        <div class="expert-top">
          <span><span class="led-dot"></span> ${exp.code.slice(0, 18)}</span>
          <span style="color:${isActive ? "#34d399" : "#64748b"}">${probPct}</span>
        </div>
        <div class="expert-sub">${exp.desc}</div>
      </div>
    `;
  }).join("");
}

function updateTopTelemetry(status, turn = null) {
  if (!status) return;
  const ram = status.ram_telemetry || {};
  const ramMb = Number(ram.process_ram_mb || 0).toFixed(1);
  const limitMb = Number(ram.device_ram_limit_mb || 2048);
  const pct = Math.min(100, Math.max(2, (ramMb / limitMb) * 100)).toFixed(1);

  const ramValEl = document.getElementById("ram-value");
  const ramBarEl = document.getElementById("ram-bar");
  if (ramValEl) ramValEl.textContent = `${ramMb} / ${limitMb} MB (${pct}%)`;
  if (ramBarEl) ramBarEl.style.width = `${pct}%`;

  const pageInEl = document.getElementById("moe-page-events");
  if (pageInEl) {
    pageInEl.textContent = `IN:${ram.page_in_events || 0} | OUT:${ram.page_out_events || 0}`;
  }

  const vcoreEl = document.getElementById("vcore-compliance");
  if (vcoreEl && status.vcore_audit) {
    vcoreEl.textContent = status.vcore_audit.proprietary_3_in_1_compliance
      ? "VCORE3 (3-IN-1 LOCKED)"
      : "UNVERIFIED";
  }

  if (turn) {
    const latEl = document.getElementById("turn-latency");
    if (latEl) latEl.textContent = `${turn.total_turn_latency_ms} ms`;

    const comp = turn.compiler_telemetry || {};
    const eq = turn.decision_equilibrium_telemetry || {};
    const moe = turn.moe_routing_telemetry || {};
    renderMoEGrid(moe.active_expert_codes || [], moe.gating_probabilities || {});

    const xrayDomain = document.getElementById("xray-domain");
    const xrayCompiler = document.getElementById("xray-compiler");
    const xrayEquil = document.getElementById("xray-equil");
    const xraySlots = document.getElementById("xray-slots");
    if (xrayDomain) xrayDomain.textContent = `${comp.intent_domain} (${comp.register_mode})`;
    if (xrayCompiler) xrayCompiler.textContent = `In: ${comp.inbound_compile_ms}ms | Out: ${comp.outbound_compile_ms}ms`;
    if (xrayEquil) xrayEquil.textContent = `Align: ${eq.reality_alignment} | VM-11: ${eq.vm11_weight} (${eq.steps_executed} steps)`;
    if (xraySlots) xraySlots.textContent = `[${(comp.chosen_slot_units_6x64 || []).join(", ")}]`;

    if (turn.tool_output && turn.tool_output.svg_chart) {
      renderFinanceReport(turn.tool_output);
    }
  }

  if (status.enabled_skills) {
    renderSkillsToggles(status.enabled_skills);
  }
}

function renderFinanceReport(fin) {
  if (!fin) return;
  const symEl = document.getElementById("fin-symbol");
  const actEl = document.getElementById("fin-action");
  const svdEl = document.getElementById("fin-svd");
  const multEl = document.getElementById("fin-mult");
  const svgWrap = document.getElementById("fin-svg-container");

  if (symEl) symEl.textContent = `${fin.symbol} @ ${fin.last_price}`;
  if (actEl) actEl.textContent = `${fin.ai_selected_action} (${fin.market_regime})`;
  if (svdEl) svdEl.textContent = `γ=${fin.svd_purity_gamma} | z=${fin.ou_zscore}`;
  if (multEl) multEl.textContent = `${fin.geometric_compounding_c_mult}x (Filters=${fin.programmatic_filters_applied})`;
  if (svgWrap && fin.svg_chart) {
    svgWrap.innerHTML = fin.svg_chart;
  }
}

function renderSkillsToggles(skillsMap) {
  const row = document.getElementById("skills-row");
  if (!row) return;
  row.innerHTML = Object.entries(skillsMap).map(([name, enabled]) => `
    <label class="skill-toggle">
      <input type="checkbox" ${enabled ? "checked" : ""} onchange="window.toggleSkill('${name}', this.checked)" />
      <span>${name}</span>
    </label>
  `).join("");
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
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    const repairs = (comp.self_healed_repairs || []).length > 0
      ? `<span class="meta-tag" style="border-color:#f59e0b;color:#fcd34d">Healed: ${comp.self_healed_repairs.join(",")}</span>`
      : "";
    meta.innerHTML = `
      <span class="meta-tag">⚡ ${turn.total_turn_latency_ms} ms</span>
      <span class="meta-tag">🧠 ${(moe.active_expert_codes || []).join(" + ")}</span>
      <span class="meta-tag">💾 ${hw.process_ram_mb} MB / 2048 MB</span>
      <span class="meta-tag">🎯 Slots [${(comp.chosen_slot_units_6x64 || []).join(",")}]</span>
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

async function sendChatMessage(customText = null) {
  if (isBusy) return;
  const inputEl = document.getElementById("chat-input");
  const voiceChk = document.getElementById("voice-toggle");
  const sendBtn = document.getElementById("send-btn");

  const text = (customText !== null ? customText : inputEl.value).trim();
  if (!text) return;

  if (customText === null) inputEl.value = "";
  appendChatMessage("user", text);

  isBusy = true;
  if (sendBtn) {
    sendBtn.disabled = true;
    sendBtn.textContent = "Compiling...";
  }

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        synthesize_voice: voiceChk ? voiceChk.checked : false,
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
    appendChatMessage("vivy", `[Self-Healed Local Guard] Unable to reach endpoint: ${err.message}`);
  } finally {
    isBusy = false;
    if (sendBtn) {
      sendBtn.disabled = false;
      sendBtn.textContent = "Send (⚡ <15ms)";
    }
  }
}

window.runQuickPrompt = function (text) {
  sendChatMessage(text);
};

window.requestFinanceSymbol = async function (symbol) {
  try {
    const res = await fetch("/api/finance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol }),
    });
    const data = await res.json();
    if (data.ok) {
      renderFinanceReport(data.finance);
      updateTopTelemetry(data.status);
      showToast(`Finance-3B-MoE analyzed ${symbol} (0 programmatic filters)`, "success");
    }
  } catch (err) {
    showToast(`Finance analysis failed: ${err.message}`, "error");
  }
};

window.triggerOneTouchUnlock = async function () {
  try {
    const res = await fetch("/api/auth/unlock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const data = await res.json();
    if (data.ok) {
      updateTopTelemetry(data.status);
      showToast(`1-Touch Sovereign Vault Unlocked: ${data.ticket.vcore_key_fingerprint}`, "success");
    }
  } catch (err) {
    showToast(`Unlock error: ${err.message}`, "error");
  }
};

window.triggerLucidDream = async function () {
  try {
    const res = await fetch("/api/dream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rounds: 2 }),
    });
    const data = await res.json();
    if (data.ok) {
      updateTopTelemetry(data.status);
      renderMoEGrid([], {});
      showToast(`LUCID_STANDBY Dream completed: RAM=${data.dream.ram_after_mb} MB`, "success");
    }
  } catch (err) {
    showToast(`Dream error: ${err.message}`, "error");
  }
};

window.toggleSkill = async function (skillName, enabled) {
  try {
    const res = await fetch("/api/skills/toggle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skill_name: skillName, enabled }),
    });
    const data = await res.json();
    if (data.ok) {
      updateTopTelemetry(data.status);
      showToast(`Skill '${skillName}' ${enabled ? "ENABLED" : "DISABLED"}`, "success");
    }
  } catch (err) {
    showToast(`Skill toggle failed: ${err.message}`, "error");
  }
};

window.addEventListener("DOMContentLoaded", async () => {
  renderMoEGrid(["E6_COMPOSITIONAL_DIALOGUE_VOICE", "E7_VM11_DREAM_EVOLUTION"], {});

  const inputEl = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  if (inputEl) {
    inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });
  }
  if (sendBtn) {
    sendBtn.addEventListener("click", () => sendChatMessage());
  }

  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.ok) {
      updateTopTelemetry(data.status);
    }
    await window.requestFinanceSymbol("XAUUSD");
  } catch (err) {
    showToast(`Initial status check failed: ${err.message}`, "error");
  }
});
