# web/ — frontend layer map

ComfyUI serves this whole directory as `WEB_DIRECTORY` and auto-loads every
`.js` it finds; `index.js` ALSO imports everything explicitly (ES modules
dedupe, so double-loading is harmless) — keep new files listed in `index.js`
so loading never depends on the server's glob behavior. `scripts/check_web.py`
enforces this.

## Layers

- **`lib/`** — shared library, originally derived from rgthree-comfy's module
  split and renamed with the `mxd_` prefix. These are complementary layers,
  NOT duplicates — don't merge them:
  - `mxd_runtime.js` — singleton runtime; patches `app.loadApiJson`, canvas
    copy/paste tracking, toast messages.
  - `mxd_utils.js` — LiteGraph graph/topology helpers (connection layout,
    node traversal). Imports from `mxd_shared_utils.js`.
  - `mxd_shared_utils.js` — generic helpers (debounce, resolver, array ops,
    `injectCss`).
  - `mxd_utils_dom.js` / `mxd_utils_canvas.js` / `mxd_utils_widgets.js` /
    `mxd_utils_menu.js` — DOM building, canvas drawing primitives, custom
    widget classes, model/lora chooser menus.
  - `mxd_base_node.js` — `MxdBaseNode` / `MxdBaseServerNode` +
    `registerForOverride` machinery (how MXD classes replace the stock
    LiteGraph node class for a Python node).
  - `mxd_dialog.js` → `mxd_dialog_info.js` — dialog base → model-info dialogs.
    CSS is injected LAZILY: the first opened info dialog calls `injectCss` on
    `mxd_dialog_base.css` (which `@import`s dialog/buttons/menu css) and
    `mxd_dialog_model_info.css`. CSS must stay in `lib/` next to
    `mxd_dialog_info.js` (paths derive from its `import.meta.url`).
  - `mxd_api.js` / `mxd_model_info_service.js` / `mxd_model_row_widget.js` /
    `mxd_smart_search.js` / `mxd_menu.js` / `mxd_svgs.js`.

- **`nodes/`** — one extension file per node/feature. Each registers via
  `app.registerExtension` and targets Python node names in
  `beforeRegisterNodeDef` (names must match `NODE_CLASS_MAPPINGS` keys).
  - `power_lora_base.js` — shared base for the lora loader: node machinery,
    header row, single-strength row widget. `power_lora_loader.js` adds the
    dual model/clip strength mode. Rows are canvas-drawn in both classic and
    Nodes 2.0; `mxd_nodes2.js` supplies the width and repaint shims that make
    that work. Serialization shapes are frozen (see the CLAUDE.md contract).
  - `better_combos.js` — folder-tree/grid combo display for the MXD latent
    loaders (adapted from pysssss; scoped to MXD nodes only; keeps its BOM).
  - `run_folder.js` — wraps `app.queuePrompt` for batch folder runs; uses
    `/mxd/latents/files`.
  - `prompt_spellcheck.js` — right-click spelling suggestions in any prompt
    textarea, using the vendored dictionary in `vendor/spellcheck/`. Also
    ships standalone as the Spell-Check-MXD pack; both copies claim the shared
    `window.__mxdPromptSpellcheckActive` flag so installing both attaches it
    once. If you rename that flag, rename it in BOTH repos.

- **`vendor/zip_loader/`** — vendored drag-drop zip workflow importer +
  bundled `jszip.min.js` (never lint/format the min file).
- **`vendor/spellcheck/`** — vendored `nspell` browser bundle and English
  Hunspell dictionary used by `nodes/prompt_spellcheck.js` for fully offline
  right-click spelling suggestions in prompt textareas.

## Conventions

- Imports of `../../../scripts/app.js` (and api.js/ui.js/widgets.js) are
  ComfyUI-served URLs — the number of `../` depends on the file's depth under
  `web/`. Files in `lib/` and `nodes/` use three `../`.
- UTF-8 without BOM (exception: `nodes/better_combos.js`).
- After ANY web change run:
  `C:\Users\user\Documents\ComfyUI\.venv\Scripts\python.exe scripts/check_web.py`
