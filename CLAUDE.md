# ComfyUI-MaxedOut — guide for AI agents

ComfyUI custom node pack ("Maxed Out", ~68 nodes) published to the ComfyUI
registry. Users install it as ONE pack; the owner's Patreon workflows depend
on the exact node names. The owner does not read code — you are maintaining
this for him, so keep it navigable and verify with the harness below instead
of asking him to review diffs.

## Read only what you need (routing table)

| Working on… | Read |
|---|---|
| Flux/SDXL/ZIT empty latents, resolution presets | `nodes/latents.py` |
| Image scale-to-megapixels, resolution matchers | `nodes/resolution.py` |
| Prompt/conditioning (Flux guidance, Qwen edit) | `nodes/prompts.py` |
| Mask ops, bbox, crop, image+mask preview | `nodes/masks.py` |
| Image/video loaders & savers, workflow extract | `nodes/media_io.py` + `web/nodes/load_*.js` |
| Image/Video Comparer | `nodes/comparers.py` + `web/nodes/image_comparer.js` |
| Checkpoint load/save | `nodes/checkpoints.py` + `web/nodes/checkpoint_loader.js` |
| FFGO material combining | `nodes/ffgo.py` + `web/nodes/combine_materials_ffgo.js` |
| WAN 2.2 anything | `nodes/wan22/` (buckets, latent_io, i2v, video_ops) |
| LTX anything | `nodes/ltx/` (latents, samplers, preview) + `web/nodes/ltx_sampler.js` |
| Power Lora Loaders | `loraloader_mxd/` (Python) + `web/nodes/power_lora_base.js` and its two subclass files |
| Model info dialogs / CivitAI cache | `loraloader_mxd/server/` + `web/lib/mxd_dialog_info.js`, `web/lib/mxd_model_info_service.js` |
| Character prompts | `CharacterPrompts/` + `web/nodes/character_prompts.js` |
| Smart (GGUF) loaders | `smart_loaders_mxd/` — vendored from city96/ComfyUI-GGUF, see its `UPSTREAM.md` before editing |
| Live video previews during sampling | `system/live_preview.py` + `web/nodes/live_preview_panel.js`; LTX-specific: `nodes/ltx/preview.py` |
| External model folder registration | `system/model_paths.py` + `web/nodes/model_storage_settings.js` |
| Shared Python helpers | `nodes/shared/` (paths, metadata, guarded routes) |
| Shared web layer | `web/lib/` (rgthree-derived; see `web/WEB.md`) |

Each Python module's docstring lists the nodes it registers. `__init__.py` at
the root merges every module's `NODE_CLASS_MAPPINGS` via safe-import (a broken
module prints a warning instead of killing the pack).

## Compatibility contract (breaking any of these breaks user workflows)

Never change without explicit owner approval:
- Node mapping keys and display names (`NODE_CLASS_MAPPINGS` /
  `NODE_DISPLAY_NAME_MAPPINGS`), input names/types/defaults/order,
  `RETURN_TYPES`/`RETURN_NAMES`, `FUNCTION`, `CATEGORY`.
- HTTP routes (`/mxd/...`, `/loraloader-mxd/...`) and websocket events
  (`MXD_live_preview_*`, `loraloader-mxd-refreshed-*`).
- JS widget serialization shapes (lora rows: `{on, lora, strength[, strengthTwo]}`;
  LTX2 strength rows: `{type: "Ltx2StrengthWidget", key, value}`) and widget
  names (`lora_N`, `ltx2_<key>`).
- Settings IDs (`MXD.*`, `mxd.Combo++.Submenu`) and the userdata path
  (`userdata/loraloader_mxd/`, gitignored CivitAI metadata cache).
- `model_storage_config.json` lives at the REPO ROOT (gitignored).

File layout is invisible to users — moving code is always safe IF the schema
harness stays clean.

## Verify every change (the harness is the review)

```
# Node schema snapshot vs baseline — must print "OK: 68 nodes identical"
C:\Users\user\Documents\ComfyUI\.venv\Scripts\python.exe scripts/dump_node_schema.py scripts/node_schema_current.json
C:\Users\user\Documents\ComfyUI\.venv\Scripts\python.exe scripts/dump_node_schema.py --compare scripts/node_schema_baseline.json scripts/node_schema_current.json

# Web static checks (imports resolve, node names exist, CSS paths valid)
C:\Users\user\Documents\ComfyUI\.venv\Scripts\python.exe scripts/check_web.py
```

The dump imports the pack against the real installed core (~60–90s, loads
torch). If you INTENTIONALLY change a node's schema (new input, new node),
re-baseline: run the dump to `scripts/node_schema_baseline.json` and commit it
with the change, saying so in the commit message. Delete
`scripts/node_schema_current.json` afterwards (never commit it).

There are no unit tests; the harness + the owner launching ComfyUI Desktop is
the verification story. You cannot launch ComfyUI yourself (see Hard rules).

## Environment facts

- Installed ComfyUI core (real source): `C:\Users\user\AI\ComfyUI\resources\ComfyUI`
- Venv python (torch 2.9.1+cu128, no pytest): `C:\Users\user\Documents\ComfyUI\.venv\Scripts\python.exe`
- This repo lives inside the user-data dir at
  `C:\Users\user\Documents\ComfyUI\custom_nodes\ComfyUI-MaxedOut`.
- The sibling pack `ComfyUI-MaxedOut-Runpod` registers the same display names
  on this machine — duplicate entries in the node search come from it, not
  from a bug here.
- Import gotcha replicated in `scripts/dump_node_schema.py`: core's `nodes.py`
  prepends `<core>/comfy` to `sys.path`, so import core's `utils` package
  BEFORE `server` or a bare `import utils` resolves to `comfy/utils.py`.

## Hard rules

- Write files as UTF-8 WITHOUT BOM. (`web/nodes/better_combos.js` is the one
  legacy BOM file — leave its BOM alone.)
- NEVER open `ComfyUI/input`, `ComfyUI/output`, or the owner's workflows, and
  never start the ComfyUI server — personal content lives there. Reading any
  code (this repo, other custom_nodes, the installed core) is fine.
- Pushing a commit that touches `pyproject.toml` to main auto-publishes to the
  ComfyUI registry (`.github/workflows/publish.yml`). Don't touch the version
  or push unless the owner asks for a release.
- `userdata/`, `model_storage_config.json`, `testing/`, `local_notes/` are
  gitignored on purpose; never commit them.
- Some nodes require `comfy_api` (guarded by `HAVE_COMFY_API`); keep the
  guard pattern when adding API-dependent nodes so the pack degrades
  gracefully on old cores.

## Import-time side effects (know before you "clean up")

- `system/live_preview.py` monkeypatches `latent_preview.get_previewer` and
  `prepare_callback` (streaming video previews + auto-save to
  `<output>/live_previews`). Registers no nodes.
- `system/model_paths.py` registers the user's external model storage
  folders with `folder_paths` at import (opt-in; X:\ drive on this machine).
- `nodes/checkpoints.py` replaces `comfy.diffusers_convert.cat_tensors`
  (FakeDevice fix — see the comment there).
- `nodes/ltx/preview.py` can DOWNLOAD the taeltx VAE from HuggingFace on
  first use (node-triggered, not import-triggered).
- Route registration goes through `nodes/shared/routes.py::register_get_route`
  so a missing PromptServer never kills node registration.
