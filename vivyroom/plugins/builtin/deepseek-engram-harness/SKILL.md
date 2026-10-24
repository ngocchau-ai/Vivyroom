---
name: deepseek-engram-harness
description: Native Rust DeepSeek-style Propose-Probe-Compose orchestration with Multi-Token Prediction (MTP) speculative verification and Engram O(1) memory lookup.
---

# DeepSeek Engram & MTP Harness Skill

1. Query `EngramMemoryTable` in `vivyroom_harness.dll` via 64-bit FNV-1a + SplitMix64 N-gram hashing in O(1).
2. Propose multi-step candidate reasoning trajectories and verify in parallel via `MtpSpeculativeVerifier`.
3. Apply `VM-11` falsification dampening to contradicted branches and compose coherent natural discourse.
