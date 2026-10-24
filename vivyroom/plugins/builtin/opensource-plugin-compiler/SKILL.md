---
name: opensource-plugin-compiler
description: Compiles and hot-mounts external SKILL.md workflows, plugin.json bundles, MCP servers, and public GitHub/HuggingFace tool repositories with AST safety validation.
---

# Open-Source Plugin & Skill Compiler

1. Fetch or locate `SKILL.md`, `plugin.json`, or Python tool modules from local path or GitHub repository.
2. Audit all Python scripts with `PluginStaticValidator` (`ast.parse`) to block unsafe calls.
3. Hot-mount compiled skill descriptors into Vivyroom's live execution registry without restart.
