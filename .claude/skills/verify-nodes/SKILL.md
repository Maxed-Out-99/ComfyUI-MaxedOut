---
name: verify-nodes
description: Verify that changes to ComfyUI-MaxedOut didn't break any node registrations, schemas, or the web layer. Run after ANY change to Python node code or web/ JS. This is the repo's only automated correctness gate — there are no unit tests.
---

# Verify ComfyUI-MaxedOut

Run both gates from the repo root. The venv python is required (system python
lacks torch/comfy deps).

## 1. Node schema gate (~60–90s, imports torch + the real ComfyUI core)

```
C:\Users\user\Documents\ComfyUI\.venv\Scripts\python.exe scripts/dump_node_schema.py scripts/node_schema_current.json
C:\Users\user\Documents\ComfyUI\.venv\Scripts\python.exe scripts/dump_node_schema.py --compare scripts/node_schema_baseline.json scripts/node_schema_current.json
```

PASS = `OK: 68 nodes identical`.

- Any `Failed to import` line in the dump output means a module crashed at
  import and its nodes silently vanished — fix that first.
- If you intentionally changed a schema (new node/input), the compare will
  fail by design: regenerate the baseline (`... dump_node_schema.py
  scripts/node_schema_baseline.json`), eyeball the git diff of the baseline
  to confirm ONLY your intended change moved, and commit it together with
  the code change.
- `scripts/node_schema_current.json` is gitignored; don't commit it.

## 2. Web static gate (fast)

```
C:\Users\user\Documents\ComfyUI\.venv\Scripts\python.exe scripts/check_web.py
```

PASS = `OK: web layer clean`. Checks that every relative import in
`web/**/*.js` resolves, every node name referenced by JS exists in the
schema baseline, all injectCss/@import CSS paths resolve, and every
subdirectory module is imported by `web/index.js` (add new files there).

## What this does NOT cover

Runtime behavior in the browser/sampler. You cannot start ComfyUI (owner's
privacy rule) — for UI-visible changes, tell the owner exactly what to click
in a short numbered smoke-test list and let him run it.
