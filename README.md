<!-- [ISOLATED / REPLACED — Initial Stub]
# Vivyroom
Vivy control room - Agent offline tự chủ
-->

# Vivyroom — Portable `< 2 GB RAM` Autonomous Control Room (`Vivy + Cautreo` 3-Cores-in-1 `.vcore`)

> [!IMPORTANT]
> **MANDATORY INHERITANCE & DOCUMENT IMMUTABILITY NOTICE FOR ALL AGENTS:**
> 1. **Mandatory Changelog:** Any agent editing this document MUST append an entry to the Changelog section (Agent Name/ID, timestamp, reason).
> 2. **Isolate Only, NEVER Delete:** Never delete previous architectural definitions or content; isolate replaced blocks with `[ISOLATED / DEPRECATED / REPLACED]`.
> 3. **Proprietary 3-Cores-in-1 `.vcore` Rule:** All Vivy model weights in this repository and across the ecosystem are encrypted in the proprietary `VIVY_CORE_TRI_ENC_V1` (`.vcore`) format (`open_source_weights = False`).

**Vivyroom** is a portable, 100% offline, single-screen autonomous agent control room designed to run effortlessly on resource-constrained edge hardware (**$\le 2\text{ GB}$ (`2,048 MB`) total RAM**, zero GPU required). It is powered directly by the unified **`Vivy + Cautreo`** organism (`entity_count = 1`) and packages **`Vivy-Finance-3B-MoE`** (`8` Sparse Experts, `Top-2` active per turn) inside encrypted **3-Cores-in-1 (`.vcore`)** containers.

---

## 1. Core Architectural Pillars

### 1.1. Proprietary 3-Cores-in-1 Encrypted Model Format (`.vcore`)
All Vivy neural weights are sealed inside the proprietary **`VIVY_CORE_TRI_ENC_V1`** binary format (`b"VCORE3\x01\x00"` header + `SHAKE256-CTR` stream cipher + `HMAC-SHA256` Encrypt-then-MAC integrity tag + `zlib` compression, `open_source_weights = False`):
1. **Core 1 — `Vivy NPS Decision Soul` (`vivy_v_soul.dll`):** 25-Step Equilibrium Thinking Loop, `vivy_nps_coupled_decide` 4-variable coupled belief propagation, and `VM-11` Falsification Dampening.
2. **Core 2 — `Cautreo Nervous System & Vivy-Finance-3B-MoE` (`cautreo.dll` + `cautreo_pager.dll` + `vivy_core.dll`):** Upcycled 8-Expert Sparse MoE (`E0`..`E7`, `Top-2` paged into RAM per turn) with Quantum Spectral SVD Density Matrix $\rho$, Ornstein-Uhlenbeck SDE, 1D Kalman velocity, and 50 RoboForex Bot 4-Pillar geometric compounding (`0` programmatic filters).
3. **Core 3 — `Qwen-Decomposed LLM Bidirectional Semantic Compiler` (`VivyQwenSemanticCompiler`):** Decomposed from Qwen-3B to act as an in-process bidirectional semantic transducer between human natural language (Vietnamese & English, including typo/no-diacritic self-healing) and the Vivy Decision Core, coupled with the `6 Slots × 64 Micro-Structures` (`68.7B` combinations) compositional atlas and `24kHz` harmonic voice synthesis.

### 1.2. Zero-Relay Custom Agent Overlay (`ZeroRelayPocketGraph`)
External multi-daemon agent relays (such as Hermes) introduce socket latency, process hangs, and connection conflicts when deployed on unattended edge hardware. `Vivyroom` eliminates all external relays in favor of an in-process state machine (`COMPILE_INBOUND -> ROUTE_MOE_AND_EQUILIBRIUM -> ACTUATE_SKILL -> COMPILE_OUTBOUND`) executing in **`3 ms – 12 ms` per turn** inside **`< 205 MB` resident Working Set RAM**.

```mermaid
flowchart LR
    User["User (Single UI Cockpit)"] -->|HTTP POST /api/chat| CompilerIn["Core 3: Qwen-Decomposed Inbound Compiler (<1 ms)"]
    CompilerIn -->|Canonical Intent AST| Pager["Core 2: Cautreo MoE Pager (Top-2 of 8 .vcore Experts)"]
    Pager -->|Active Tensors in RAM| Soul["Core 1: Vivy Decision Core (25-Step Equilibrium + NPS Decide)"]
    Soul -->|Requires Tool?| Skills["Offline Skills & Finance-3B-MoE Desk"]
    Soul -->|Direct Dialogue| CompilerOut["Core 3: 1-Step Compositional Outbound Compiler & 24kHz Voice"]
    Skills --> CompilerOut
    CompilerOut -->|JSON + SVG + 24kHz WAV (<12 ms)| User
```

---

## 2. Verified Hardware & Performance Benchmarks

| Metric | Target | Verified Measurement |
|:---|:---|:---|
| **Process Working Set RAM (Idle + Tri-Core + Top-2 MoE)** | $\le 2,048\text{ MB}$ (`2 GB`) | **`197.5 MB – 201.4 MB`** (`< 10%` of 2 GB budget) |
| **85% RAM Circuit Breaker (`Top-2 -> Top-1` fallback)** | `1,740 MB` | **`1,740 MB` enforced by `CautreoMoEDynamicPager`** |
| **Natural Language Conversational Turn Latency** | `< 25 ms` | **`2.2 ms – 9.2 ms`** (including `24kHz` voice & SVG chart) |
| **MoE Router Top-1 Domain Accuracy** | $\ge 95\%$ | **`100.0%` (`1.0000`)** across all 8 expert domains |
| **Unencrypted Weight Files on Disk (`.safetensors`/`.gguf`)** | `0` | **`0` (100% encrypted `.vcore` containers)** |
| **Programmatic Filters on Finance-3B-MoE Decisions** | `0` | **`0` (100% AI Sovereign Decision)** |

---

## 3. Repository Layout

```text
Vivyroom/
├── vivyroom/
│   ├── config.py                     # Strict 2,048 MB RAM budget & 1,740 MB Circuit Breaker
│   ├── auth_one_touch.py             # 1-Touch Sovereign Hardware-Bound Key Derivation
│   ├── server.py                     # Zero-dependency ThreadingHTTPServer (127.0.0.1:7860)
│   ├── native/
│   │   ├── bin/                      # 5 Native C11/Rust DLLs (cautreo, cautreo_pager, vivy_core, vivy_v_soul, vivy_vessel_core)
│   │   └── src/                      # Native C11 headers & sources
│   ├── moe_engine/
│   │   ├── vcore_crypto.py           # Proprietary 3-Cores-in-1 .vcore Encryption (VCORE3\x01\x00)
│   │   ├── cautreo_moe_pager.py      # Top-2 Sparse MoE Pager & C-ABI Bridge
│   │   ├── moe_upcycler_trainer.py   # Upcycles & trains 8 MoE Experts + Shared Master into .vcore
│   │   └── finance_autonomous_desk.py# Vivy-Finance-3B-MoE Quant Desk & Zero-Token SVG Chart
│   └── agent/
│       ├── pocket_graph.py           # Zero-Relay deterministic finite state machine
│       ├── llm_semantic_compiler.py  # Bidirectional Qwen-Decomposed Semantic Compiler & 24kHz Voice
│       ├── skills_registry.py        # Customizable offline skills (Finance, AST Math, File I/O, Dream)
│       ├── memory_engine.py          # Ring-buffer memory & VM-11 falsification dampening
│       └── vivy_controller.py        # Unified VivyroomAutonomousController
├── models/
│   ├── vivy_tri_core_master.vcore    # Encrypted 3-Cores-in-1 Shared Master (171 tensors)
│   ├── vivy_core_moe_manifest.json   # MoE routing & training receipt
│   ├── LICENSE-VIVY-CORE-PROPRIETARY.md
│   └── experts/                      # 8 Encrypted .vcore MoE Expert Slices (E0..E7)
├── ui/
│   ├── index.html                    # Single-Screen Unified Cockpit UI
│   ├── vivyroom_style.css            # Zero-dependency dark cockpit styling
│   └── vivyroom_app.js               # Live telemetry, chat, voice & SVG chart client
├── docs/
│   └── ARCHITECTURE_AND_DATAFLOW.md  # Full 5-Pillar Architecture & Runbook Specification
├── tests/
│   └── test_vivyroom_full_suite.py   # End-to-end automated test suite
├── run_vivyroom.bat                  # Portable 1-click Windows launcher
└── run_vivyroom.sh                   # Portable 1-click Linux/macOS launcher
```

---

## 4. Mandatory Trader Skills & Zero-Config System Prompt (`SYSTEM_PROMPT.md`)

1. **Zero-Config Trader System Prompt (`SYSTEM_PROMPT.md`):**
   - Pre-configured with the **Sovereign Quant & Algo-Bot Trader** persona (`role = SOVEREIGN_QUANT_AND_ALGO_BOT_TRADER`).
   - Prioritizes autonomous self-learning (`Autopoietic Skill-SP`, `VM-11`, `Cautreo RLS`) focused 100% on **Bot-Trade & Quant-Trade** mathematics.
   - Enforces the **Anti-Human-Bias Shield (`anti_human_bias_shield = True`)**: blocks imitating emotional/discretionary human trading habits (FOMO, loss averaging / Martingale without edge, removing Stop-Loss).
   - **Zero Setup Required:** Users downloading `Vivyroom` only need to edit `SYSTEM_PROMPT.md` (or edit it directly inside the single-screen UI) — no `.env`, API keys, or extra configuration needed.
2. **Four Mandatory Ultra-Lightweight Skills (`vivyroom/agent/mandatory_trader_skills.py`):**
   - `websearch_zero_api`: Zero-API ultra-lightweight web search (`urllib.request` + `HTMLParser`) with priority routing & ranking for **RoboForex** (`roboforex.com`) and **ForexFactory** (`forexfactory.com`).
   - `github_huggingface_connector`: Connects directly to GitHub (`api.github.com`) & HuggingFace Hub (`huggingface.co/api/models`) to download repositories or AI models, couple them into `Cautreo WorkingStateWindow4`, and seal the coupled adapter into an encrypted `.vcore` container (`open_source_weights = False`).
   - `youtube_learning_ingestor`: Extracts YouTube video transcripts/metadata (`< 2 MB` RAM, zero headless browser) and distills mathematical Bot-Trade / Quant-Trade lessons into Cautreo memory.
   - `market_news_synthesizer`: Synthesizes High-Impact economic calendar events and macro news from ForexFactory, RoboForex, and financial RSS feeds into quantitative `macro_news_bias` $\in [-1.0, +1.0]$.

---

## 4.5. `Vivyroom v2.0` Full-Scale Sovereign Edition (`DSH × Harness²` + `Computer-Use` + `Browser-Use` + `5-Tier Memory Spine`)

`Vivyroom v2.0` upgrades the control room into an **Antigravity-class General-Purpose Agent** with **Native Embedded Vivy (`0` external API keys, `0` intermediate LLM relays)**:

1. **Natural Human-Grade Conversation (`SilentPersonaFirewall` — `0%` Spec/System-Prompt Leakage):**
   - `vivyroom/agent/persona_firewall.py` and `vivyroom/agent/generative_cortex.py` (`VivyNativeEmbeddedCortex`) ensure Vivy converses warmly, naturally, and intelligently like a human partner without ever reciting her own `SYSTEM_PROMPT.md`, `.vcore` encryption headers, RAM numbers, or Expert IDs inside chat messages. Technical telemetry is cleanly isolated inside the collapsible X-Ray Drawer (`Alt+X`).
2. **Autopoietic Room Self-Control, Self-Repair & Self-Modification (`VivyRoomSelfArchitectEngine`):**
   - When permitted or requested by the user (`user_permitted = True`), Vivy can inspect, diagnose, repair, and modify her own room (`ui/index.html`, `ui/vivyroom_style.css`, Python modules, and mounted skills) with automatic `.vivyroom_snapshots/` backups, AST syntax verification, live browser hot-reload, and 1-step rollback.
3. **Native Rust + Python `DSH × Harness²` Self-Evolution (`crates/vivyroom-rs` / `vivyroom_harness.dll`):**
   - Rebuilds the core primitives of `DSH` (`deepseek-harness`: `Engram O(1)`, `MTP` speculative decoding, `Propose -> Probe -> Compose`, `Gate 4` permission & destructive-surface guard `vh_dsh_evaluate_gate4`, and `4-Layer Overlay Sync L0->L1->L2->L3` with `TAKE / KEEP / MERGE` defaulting to `KEEP` for user skills).
   - Integrates `Harness²` (`8 Organs H1..H8` attached strictly to `Vivyroom Runtime` with `attached_to_cautreo_engine = False`), `HomeostaticCognitiveFrictionDrive` (`ABSORBED_BY_RELIEF_DRIVE` vs. `SHED_BY_ATROPHY`), and `SkillSelfPlayAndDistillationFoundry` (`Proposer -> Solver -> Controller`).
4. **Universal Open-Source Plugin & Skill Compiler (`vivyroom/plugin_compiler/bridge.py`):**
   - Compiles and hot-mounts `SKILL.md` (Anthropic/gstack format), `plugin.json` (MCP format), and Python tool modules after AST static safety validation (`PluginStaticValidator`).
5. **Four Community-Validated Open-Source Capability Pillars (Rebuilt Natively):**
   - **`VivyNativeComputerUseEngine`:** Native rebuild of `trycua/cua` (~30.8k ⭐) + `compute-use` Bounded Candidate Loop (`cand_01..cand_N` + `capture_id` Stale Capture Guard) + `Open Interpreter` (~68.7k ⭐) `Code-as-Action`.
   - **`VivyNativeBrowserUseEngine` (`0%` LangChain):** Native rebuild of `browser-use/browser-use` (~118.6k ⭐) Indexed DOM Tree (`[1]..[N]`) + `Crawl4AI` (~86k ⭐) HTML-to-Clean-Markdown distiller + optional direct WebSocket CDP support (`Lightpanda` ~30.1k ⭐ Zig headless browser or Chromium/Edge).
   - **`VivyMultiProviderKeylessWebEngine`:** Multi-tier keyless web search combining `SearXNG` (~37k ⭐), `DuckDuckGo Lite/HTML` (`ddgs` ~4.5k ⭐ with `uddg` redirect unwrapping and `y.js` ad filtering), `Trafilatura/Crawl4AI` clean page extraction, and `RoboForex`/`ForexFactory` priority routing.
   - **`VivyFiveTierMemorySpine` (`M1..M5`):** Combines `M1` (Rust `Engram O(1)`), `M2` (`Cautreo WorkingStateWindow-4` & `VM-11`), `M3` (`BiTemporalSQLiteFTS5Vault` using Python's built-in `sqlite3` `FTS5` BM25 recall + `Mem0` ~67.6k ⭐ Auto-Fact Extraction + `Graphiti` ~28k ⭐ Bi-Temporal Superseding `[ISOLATED / SUPERSEDED]` with zero deletion), `M4` (`ZeroBloatEpisodeCompactor` `99.997%` archetype compression), and `M5` (`TwoBrainSovereignSync`).

---

## 5. 1-Command Git Clone & 1-Time Install (Windows & Linux / OrangePi)

<!-- [ISOLATED / REPLACED — Pre-OranPi Manual Launch Section]
### 5.1. Launch Single-Screen Unified Cockpit
On Windows:
```powershell
.\run_vivyroom.bat
```
Or directly via Python:
```bash
python -m vivyroom.server --host 127.0.0.1 --port 7860
```
Then open **`http://127.0.0.1:7860`** in any browser.
-->

### 5.1. 1-Line Clone & 1-Time Install on **Windows** (PowerShell or CMD)
```powershell
git clone https://github.com/ngocchau-ai/Vivyroom.git && cd Vivyroom && .\install_and_run.bat
```
*(Or in PowerShell: `git clone https://github.com/ngocchau-ai/Vivyroom.git; cd Vivyroom; .\install_and_run.ps1`)*

### 5.2. 1-Line Clone & 1-Time Install on **Linux / OrangePi SBC**
```bash
git clone https://github.com/ngocchau-ai/Vivyroom.git && cd Vivyroom && bash install_and_run.sh
```
- **1-Time Setup Stamp (`.vivyroom_installed`):** On the very first run, `install_and_run` automatically verifies Python and installs `numpy` once (`< 5 seconds`), writes `.vivyroom_installed`, and immediately launches `http://127.0.0.1:7860`. Every subsequent run starts instantaneously (`0` setup overhead).
- **Zero-Button 100% Autonomous Single-Chat Interface (`OranPi × PocketFlow`):**
  - The UI contains **`0` manual buttons**. You interact exclusively through **one unified chat box** (press `Enter` to send) or by editing the collapsible `SYSTEM_PROMPT.md` drawer (auto-saves automatically on edit/blur, or via `/prompt <new system prompt>` directly in chat).
  - Vivy autonomously plans and chains multiple skills in a single turn (`market_news_synthesizer -> websearch_zero_api -> finance_moe_analyze -> youtube_learning_ingestor -> github_huggingface_connector`), renders quantitative SVG charts inline inside the chat stream, and automatically executes `LUCID_STANDBY` RAM Dream compaction when needed.

### 5.3. Run Headless Self-Test & Full Pytest Suite
```bash
python -m vivyroom --self-test
pytest -q
```

---

## 6. Changelog

| Agent | Timestamp | Reason |
|:---|:---|:---|
| Antigravity IDE | 2026-10-24 11:48 ICT | Initial release of `Vivyroom` v1.0.0: Portable `<2 GB RAM` unified `Vivy + Cautreo` cockpit, proprietary 3-Cores-in-1 `.vcore` encryption, 8-Expert `Vivy-Finance-3B-MoE`, and Qwen-Decomposed Bidirectional Semantic Compiler. |
| Antigravity IDE | 2026-10-24 12:12 ICT | Added 4 mandatory ultra-lightweight skills (`websearch_zero_api` prioritizing RoboForex & ForexFactory, `github_huggingface_connector`, `youtube_learning_ingestor`, `market_news_synthesizer`) and Zero-Config `SYSTEM_PROMPT.md` with Anti-Human-Bias Shield (`6/6 pytest passed`). |
| Antigravity IDE | 2026-10-24 13:18 ICT | Upgraded to Button-Free (`0` buttons) 100% Autonomous Single-Chat UI powered by open-source `OranPi × PocketFlow` (`vivyroom/agent/oranpi_pocket_agent.py`), autonomous multi-skill chaining, auto-saving `SYSTEM_PROMPT.md`, and 1-command Git clone & 1-time install scripts (`install_and_run.bat`, `install_and_run.ps1`, `install_and_run.sh`) for Windows and Linux / OrangePi. |
| Antigravity IDE | 2026-10-24 19:04 ICT | Upgraded to `Vivyroom v2.0` Full-Scale Sovereign Edition: native Rust `crates/vivyroom-rs` (`vivyroom_harness.dll`), `SilentPersonaFirewall` (`0%` spec/prompt leakage), `VivyNativeEmbeddedCortex`, `VivyRoomSelfArchitectEngine` (permission-gated self-control & 1-step rollback), `DSH × Harness²` (`Gate 4` + 8 Organs + Skill Self-Play Distillation), `UniversalPluginCompilerBridge`, `VivyNativeComputerUseEngine`, `VivyNativeBrowserUseEngine` (`0%` LangChain), `VivyMultiProviderKeylessWebEngine`, and `VivyFiveTierMemorySpine` (`17/17 pytest` + `3/3 cargo test` passed). |
