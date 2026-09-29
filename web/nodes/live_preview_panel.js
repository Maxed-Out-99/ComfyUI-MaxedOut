import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyButton } from "../../../scripts/ui/components/button.js";

// A single global live-preview panel docked to the right edge of the canvas,
// toggled by a button in the top-right button cluster (next to Share and
// core's own "Toggle properties panel").
//
// Not a per-node DOM widget: ComfyUI tears down and rebuilds node widgets with
// the graph, so anything attached to a specific node disappears the moment you
// switch workflow tabs, or just stops updating while you're looking elsewhere
// and never resumes. This panel lives in document.body instead, so it survives
// switching workflows, leaving and coming back mid-run, and a full page reload
// (via localStorage) once a run has finished.
//
// It was previously a free-floating window you had to drag into place and
// collapse to get out of the way. It's now pinned: it tracks the canvas
// element's own bounding box, so it lines up with the top and bottom of the
// graph area and -- because core's splitter shrinks the canvas when the
// properties panel opens -- it slides over to sit beside that panel instead of
// on top of it. Only the width is yours to set (drag its left edge); that and
// the open/closed state persist.
//
// The toggle button goes through the legacy `app.menu` button groups, whose
// element core still explicitly mounts into the actionbar's
// legacy-topbar-container -- but none of ComfyButton's legacy styling is used.
// `.comfyui-button` renders as a flat grey box next to the themed Vue buttons
// around it, and ComfyButton's `icon` option emits `mdi mdi-<name>` for a font
// the frontend no longer ships. So the button is styled from the same CSS
// custom properties core's own `variant="secondary" size="icon"` buttons
// resolve to (--secondary-background / --base-foreground / --radius-md), which
// makes it follow the active colour palette, and the glyph is
// `icon-[lucide--film]` -- an iconify class that is verified present in the
// shipped CSS because core uses that icon elsewhere. Made-up icon-[...] classes
// render nothing: only icons used at build time get compiled in.
//
// Backend (system/live_preview.py / nodes/ltx/preview.py) streams frames over
// its own private MXD_live_preview_start / MXD_live_preview_frame websocket
// events (plain JSON, base64 JPEG -- not the shared VHS_latentpreview/b_preview
// channel, which turned out to be an unreliable place to listen: whichever of
// us, VHS, or core's own default preview happened to register first could
// swallow the event before we saw it). Once a run finishes the backend saves
// the result to output/live_previews and sends MXD_live_preview_saved with a
// `kind`: a clip becomes an .mp4 shown in a real <video> (native
// play/pause/scrub/fullscreen/speed/PiP), a single image becomes a .png shown
// in an <img> with no playback chrome to pretend otherwise.

const STORAGE_LAST = "MXD.LastLivePreview";
const STORAGE_OPEN = "MXD.LivePreviewOpen";
const STORAGE_WIDTH = "MXD.LivePreviewWidth";

const SETTING_AUTO_OPEN = "MXD.LivePreviewAutoOpen";
const COMMAND_TOGGLE = "MXD.ToggleLivePreviewPanel";

const MIN_WIDTH = 220;
const DEFAULT_WIDTH = 360;
const HANDLE_WIDTH = 6;

let panelEl, titleEl, bodyEl, canvasEl, videoEl, imageEl, statusEl, toggleButton;
let frames = [];
let currentId = null;
let playTimer = null;
let displayIndex = 0;
let liveRate = 8;
let isLive = false;
let isOpen = false;
let width = DEFAULT_WIDTH;

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function readStoredWidth() {
    const stored = parseFloat(localStorage.getItem(STORAGE_WIDTH));
    return Number.isFinite(stored) ? Math.max(MIN_WIDTH, stored) : DEFAULT_WIDTH;
}

function store(key, value) {
    try {
        localStorage.setItem(key, value);
    } catch (e) {
        // storage full/unavailable -- not worth failing over
    }
}

function nodeLabelFor(id) {
    const shortId = String(id ?? "").split(":")[0];
    const node = app.graph?._nodes_by_id?.[shortId] ?? app.graph?.getNodeById?.(shortId);
    return node?.title || node?.type || (id != null ? `Node ${id}` : "Live Preview");
}

function canvasRect() {
    const el = app.canvasEl ?? document.getElementById("graph-canvas");
    const rect = el?.getBoundingClientRect();
    if (!rect || rect.width < 1 || rect.height < 1) return null;
    return rect;
}

// Pin to the canvas rather than the viewport, so the panel lands inside the
// graph area: below whatever menu layout is active, above the bottom panel,
// and to the left of the properties panel when core's splitter opens it.
function syncGeometry() {
    if (!panelEl || !isOpen) return;
    const rect = canvasRect();
    if (!rect) return;
    panelEl.style.top = `${rect.top}px`;
    panelEl.style.height = `${rect.height}px`;
    panelEl.style.right = `${Math.max(0, window.innerWidth - rect.right)}px`;
    panelEl.style.width = `${clamp(width, MIN_WIDTH, Math.max(MIN_WIDTH, rect.width - 40))}px`;
}

function injectStyles() {
    if (document.getElementById("mxd-live-preview-style")) return;
    const style = document.createElement("style");
    style.id = "mxd-live-preview-style";
    style.textContent = `
#mxd-live-preview-panel {
    position: fixed;
    z-index: 900;
    display: none;
    flex-direction: column;
    background: var(--comfy-menu-bg, rgba(24,24,24,0.96));
    border-left: 1px solid var(--border-color, rgba(255,255,255,0.15));
    box-shadow: -4px 0 20px rgba(0,0,0,0.45);
    color: var(--fg-color, #eee);
    font-family: sans-serif;
}
#mxd-live-preview-panel .mxd-lp-header {
    display: flex; align-items: center; gap: 6px;
    padding: 0 4px 0 10px; height: 28px; flex: none;
    background: rgba(0,0,0,0.25);
    font-size: 12px; user-select: none;
}
#mxd-live-preview-panel .mxd-lp-title {
    flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
#mxd-live-preview-panel .mxd-lp-close {
    background: transparent; border: none; color: inherit; cursor: pointer;
    font-size: 15px; line-height: 1; padding: 4px 7px; border-radius: 4px;
}
#mxd-live-preview-panel .mxd-lp-close:hover { background: rgba(255,255,255,0.12); }
#mxd-live-preview-panel .mxd-lp-body {
    flex: 1; min-height: 0; position: relative; overflow: hidden; background: #000;
}
#mxd-live-preview-panel .mxd-lp-body > * {
    position: absolute; inset: 0; width: 100%; height: 100%;
    object-fit: contain; display: none;
}
#mxd-live-preview-panel .mxd-lp-status {
    padding: 3px 10px; font-size: 11px; opacity: 0.7; flex: none;
}
#mxd-live-preview-panel .mxd-lp-grip {
    position: absolute; left: -${HANDLE_WIDTH / 2}px; top: 0; bottom: 0;
    width: ${HANDLE_WIDTH}px; cursor: ew-resize; z-index: 1;
}

/* Mirrors core's <Button variant="secondary" size="icon">: same theme custom
   properties, so it tracks whatever colour palette is selected. */
.mxd-topbar-button {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: calc(var(--spacing, 0.25rem) * 2);
    width: calc(var(--spacing, 0.25rem) * 8);
    height: calc(var(--spacing, 0.25rem) * 8);
    margin: 0;
    padding: 0;
    border: none;
    border-radius: var(--radius-md, 0.375rem);
    background-color: var(--secondary-background, #3a3a3a);
    color: var(--base-foreground, #eee);
    font: inherit;
    line-height: 1;
    white-space: nowrap;
    cursor: pointer;
    touch-action: manipulation;
    transition: color 0.15s, background-color 0.15s;
}
.mxd-topbar-button:hover {
    background-color: var(--secondary-background-hover, #4a4a4a);
}
/* ComfyButton nests our icon inside its own <span>; make that a flex row too
   so the glyph is centred rather than sitting on the text baseline. */
.mxd-topbar-button > span {
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
.mxd-topbar-button.mxd-lp-active {
    background-color: var(--primary-background, #2d7ff9);
    color: var(--base-foreground, #fff);
}
.mxd-topbar-button.mxd-lp-active:hover {
    background-color: var(--primary-background-hover, #1f6fe0);
}
`;
    document.head.appendChild(style);
}

function makeResizable(gripEl) {
    gripEl.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        gripEl.setPointerCapture(event.pointerId);
        const startX = event.clientX;
        const startWidth = parseFloat(panelEl.style.width) || width;

        const onMove = (moveEvent) => {
            const rect = canvasRect();
            const max = rect ? Math.max(MIN_WIDTH, rect.width - 40) : window.innerWidth;
            width = clamp(startWidth - (moveEvent.clientX - startX), MIN_WIDTH, max);
            panelEl.style.width = `${width}px`;
        };
        const onUp = (upEvent) => {
            gripEl.releasePointerCapture(upEvent.pointerId);
            gripEl.removeEventListener("pointermove", onMove);
            gripEl.removeEventListener("pointerup", onUp);
            store(STORAGE_WIDTH, String(width));
        };
        gripEl.addEventListener("pointermove", onMove);
        gripEl.addEventListener("pointerup", onUp);
    });
}

function buildPanel() {
    if (panelEl) return;
    injectStyles();
    width = readStoredWidth();

    panelEl = document.createElement("div");
    panelEl.id = "mxd-live-preview-panel";

    const headerEl = document.createElement("div");
    headerEl.className = "mxd-lp-header";

    titleEl = document.createElement("span");
    titleEl.className = "mxd-lp-title";
    titleEl.textContent = "Live Preview";

    const closeEl = document.createElement("button");
    closeEl.className = "mxd-lp-close";
    closeEl.textContent = "×";
    closeEl.title = "Close live preview";
    closeEl.addEventListener("click", () => setOpen(false));

    headerEl.append(titleEl, closeEl);

    bodyEl = document.createElement("div");
    bodyEl.className = "mxd-lp-body";

    canvasEl = document.createElement("canvas");

    videoEl = document.createElement("video");
    videoEl.controls = true;
    videoEl.loop = true;
    videoEl.muted = true;
    videoEl.autoplay = true;
    videoEl.playsInline = true;

    imageEl = document.createElement("img");

    statusEl = document.createElement("div");
    statusEl.className = "mxd-lp-status";

    const gripEl = document.createElement("div");
    gripEl.className = "mxd-lp-grip";

    bodyEl.append(canvasEl, videoEl, imageEl);
    panelEl.append(headerEl, bodyEl, statusEl, gripEl);
    document.body.appendChild(panelEl);

    makeResizable(gripEl);
    window.addEventListener("resize", syncGeometry);
    const canvas = app.canvasEl ?? document.getElementById("graph-canvas");
    if (canvas && typeof ResizeObserver !== "undefined") {
        new ResizeObserver(syncGeometry).observe(canvas);
    }
}

function setOpen(open) {
    buildPanel();
    isOpen = open;
    panelEl.style.display = open ? "flex" : "none";
    store(STORAGE_OPEN, open ? "1" : "0");
    syncToggleButton();
    // Playback is suspended while hidden; a run that finished (or is still
    // going) behind a closed panel picks up where it left off on reopen.
    if (open) {
        syncGeometry();
        if (isLive && canvasEl.style.display === "block") startCanvasPlayback(liveRate);
        if (videoEl.style.display === "block") videoEl.play().catch(() => {});
    } else {
        stopCanvasPlayback();
        videoEl.pause();
    }
}

function stopCanvasPlayback() {
    if (playTimer) {
        clearInterval(playTimer);
        playTimer = null;
    }
}

function showOnly(element) {
    for (const child of bodyEl.children) {
        child.style.display = child === element ? "block" : "none";
    }
}

function startCanvasPlayback(rate) {
    stopCanvasPlayback();
    displayIndex = 0;
    showOnly(canvasEl);
    videoEl.pause();
    if (!isOpen) return;

    playTimer = setInterval(() => {
        if (frames.length === 0) return;
        const image = frames[displayIndex];
        if (image) {
            if (canvasEl.width !== image.width || canvasEl.height !== image.height) {
                canvasEl.width = image.width;
                canvasEl.height = image.height;
            }
            canvasEl.getContext("2d")?.drawImage(image, 0, 0);
        }
        displayIndex = (displayIndex + 1) % frames.length;
    }, 1000 / Math.max(1, rate || 8));
}

function showSaved(url, label, kind) {
    stopCanvasPlayback();
    isLive = false;
    if (kind === "image") {
        showOnly(imageEl);
        imageEl.src = url;
        videoEl.pause();
        videoEl.removeAttribute("src");
    } else {
        showOnly(videoEl);
        videoEl.src = url;
        if (isOpen) videoEl.play().catch(() => {});
    }
    statusEl.textContent = "Finished";
    titleEl.textContent = label;
    store(STORAGE_LAST, JSON.stringify({ url, label, kind }));
}

function restoreLastPreview() {
    buildPanel();
    let saved = null;
    try {
        saved = JSON.parse(localStorage.getItem(STORAGE_LAST) || "null");
    } catch (e) {
        // ignore corrupt storage
    }
    setOpen(localStorage.getItem(STORAGE_OPEN) === "1");
    if (saved?.url) showSaved(saved.url, saved.label, saved.kind);
}

const TOGGLE_CLASSES = "mxd-topbar-button mxd-live-preview-toggle";

// Set through ComfyButton's own `classList` prop rather than the element's
// classList: ComfyButton rewrites className wholesale whenever its hidden or
// enabled props change, which would silently drop an externally-added class.
function syncToggleButton() {
    if (!toggleButton) return;
    toggleButton.classList = isOpen ? `${TOGGLE_CLASSES} mxd-lp-active` : TOGGLE_CLASSES;
}

function addToggleButton() {
    if (toggleButton || !app.menu?.element) return;
    injectStyles();
    const icon = document.createElement("i");
    icon.className = "icon-[lucide--film] size-4";
    toggleButton = new ComfyButton({
        content: icon,
        tooltip: "Toggle live preview panel (Ctrl+E)",
        classList: TOGGLE_CLASSES,
        action: () => setOpen(!isOpen),
    });
    // Straight into app.menu's own flex row, not into one of its
    // ComfyButtonGroups: a group is `overflow: hidden` with a 4px radius and
    // strips its children's own rounding, which is the segmented-toolbar look,
    // not a standalone button. As a direct child it gets the row's gap and
    // keeps its corners, and nothing already in those groups is touched.
    app.menu.element.appendChild(toggleButton.element);
    syncToggleButton();
}

let listenersAttached = false;
function attachListeners() {
    if (listenersAttached) return;
    listenersAttached = true;

    api.addEventListener("MXD_live_preview_start", (event) => {
        const detail = event.detail;
        if (detail?.id == null) return;

        buildPanel();
        currentId = detail.id;
        frames = new Array(detail.length);
        liveRate = detail.rate;
        isLive = true;
        if (!isOpen && app.ui.settings.getSettingValue(SETTING_AUTO_OPEN) !== false) setOpen(true);
        titleEl.textContent = nodeLabelFor(detail.id);
        statusEl.textContent = "Live";
        startCanvasPlayback(detail.rate);
    });

    // Frames keep arriving into the buffer whether or not the panel is open,
    // so opening it mid-run shows the current state immediately.
    api.addEventListener("MXD_live_preview_frame", (event) => {
        const detail = event.detail;
        if (!detail || detail.id !== currentId || !frames.length) return;
        const img = new Image();
        img.onload = () => {
            frames[detail.index] = img;
        };
        img.src = detail.data;
    });

    api.addEventListener("MXD_live_preview_saved", (event) => {
        const detail = event.detail;
        if (!detail?.filename) return;

        const url = api.apiURL(
            `/view?filename=${encodeURIComponent(detail.filename)}` +
            `&type=${encodeURIComponent(detail.type || "output")}` +
            `&subfolder=${encodeURIComponent(detail.subfolder || "live_previews")}`
        );
        buildPanel();
        showSaved(url, nodeLabelFor(detail.node_id), detail.kind || "video");
    });
}

app.registerExtension({
    name: "ComfyUI-MaxedOut.LivePreviewPanel",

    commands: [
        {
            id: COMMAND_TOGGLE,
            label: "Toggle Live Preview Panel",
            function: () => setOpen(!isOpen),
        },
    ],

    // Ctrl+E: left hand alone, an easy one-handed reach (unlike Alt+V), and
    // free -- core only takes Ctrl+Shift+E (logs terminal). It's also not in
    // core's "reserved by text input" combo list (that list is Ctrl+A/C/V/X/Z/Y/P,
    // Enter, and navigation keys), so it fires even with the prompt textarea
    // focused, not just when the canvas has focus. Registered as a *default*
    // binding, so it shows up in Settings > Keybindings and can be rebound
    // there without being clobbered on the next load.
    keybindings: [
        { commandId: COMMAND_TOGGLE, combo: { key: "e", ctrl: true } },
    ],

    setup() {
        app.ui.settings.addSetting({
            id: SETTING_AUTO_OPEN,
            category: ["MXD", "Sampling", "Live Preview Auto Open"],
            name: "Open the live preview panel automatically when sampling starts",
            tooltip: "Off means the panel only appears when you click its button in the top-right toolbar; previews still stream into it in the background either way.",
            type: "boolean",
            defaultValue: true,
        });

        attachListeners();
        addToggleButton();
        restoreLastPreview();
    },
});

