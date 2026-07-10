# Agent instructions

Read `CLAUDE.md` — it is the canonical guide for AI agents working in this
repo (routing table, compatibility contract, verification harness, hard
rules). Everything there applies to every agent, not just Claude.

Absolute minimum if you read nothing else:
- ALWAYS read/write files as UTF-8 (WITHOUT BOM).
- Never change node mapping keys, display names, or input/output schemas —
  verify with `scripts/dump_node_schema.py --compare` (see CLAUDE.md).
- Never open the user's ComfyUI input/output/workflows or start the ComfyUI
  server (personal content).
