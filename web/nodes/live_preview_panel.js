import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// A single global, floating live-preview panel -- NOT a per-node DOM widget.
// ComfyUI tears down/rebuilds node widgets with the graph, so anything
// attached to a specific node disappears the moment you switch workflow
// tabs, or (depending on version) just stops updating while you're looking
// elsewhere and never resumes. This panel lives in document.body instead, so
// it survives switching workflows, leaving and coming back mid-run, and even
// a full page reload (via localStorage) once a run has finished. It's also
// freely draggable (grab the header) and resizable (grab any edge/corner),
// with both position/size and collapsed state remembered across reloads.
//
// Backend (video_preview_mxd.py / ltxnodes.py) streams frames over its own
// private MXD_live_preview_start / MXD_live_preview_frame websocket events
// (plain JSON, base64 JPEG -- not the shared VHS_latentpreview/b_preview
// channel, which turned out to be an unreliable place to listen: whichever
// of us, VHS, or core's own default preview happened to register first could
// swallow the event before we saw it). Once a run finishes, the backend also
// saves an .mp4 to output/live_previews and sends MXD_live_preview_saved, at
// which point this panel swaps from the live canvas to a real <video> --
// native play/pause/scrub/fullscreen/speed/PiP, no custom controls needed.

const STORAGE_LAST = "MXD.LastLivePreview";
const STORAGE_MINIMIZED = "MXD.LivePreviewMinimized";
const STORAGE_GEOMETRY = "MXD.LivePreviewGeometry";

const MIN_WIDTH = 200;
const MIN_HEIGHT = 150;
const HEADER_HEIGHT = 26;
const DEFAULT_WIDTH = 300;
const DEFAULT_HEIGHT = 220;
const RESIZE_HANDLE_SIZE = 8;

let panelEl, headerEl, titleEl, minimizeEl, bodyEl, canvasEl, videoEl, statusEl;
let frames = [];
let currentId = null;
let playTimer = null;
let displayIndex = 0;
let minimized = false;
let expandedHeight = DEFAULT_HEIGHT;

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function nodeLabelFor(id) {
    const shortId = String(id ?? "").split(":")[0];
    const node = app.graph?._nodes_by_id?.[shortId] ?? app.graph?.getNodeById?.(shortId);
    return node?.title || node?.type || (id != null ? `Node ${id}` : "Live Preview");
}

function loadGeometry() {
    try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_GEOMETRY) || "null");
        if (saved && Number.isFinite(saved.left) && Number.isFinite(saved.top)
            && Number.isFinite(saved.width) && Number.isFinite(saved.height)) {
            return saved;
        }
    } catch (e) {
        // ignore corrupt storage
    }
    return {
        left: Math.max(8, window.innerWidth - DEFAULT_WIDTH - 16),
        top: Math.max(8, window.innerHeight - DEFAULT_HEIGHT - 16),
        width: DEFAULT_WIDTH,
        height: DEFAULT_HEIGHT,
    };
}

function saveGeometry() {
    try {
        localStorage.setItem(STORAGE_GEOMETRY, JSON.stringify({
            left: parseFloat(panelEl.style.left),
            top: parseFloat(panelEl.style.top),
            width: parseFloat(panelEl.style.width),
            height: expandedHeight,
        }));
    } catch (e) {
        // storage full/unavailable -- not worth failing over
    }
}

function clampToViewport() {
    const left = clamp(parseFloat(panelEl.style.left), 0, Math.max(0, window.innerWidth - 60));
    const top = clamp(parseFloat(panelEl.style.top), 0, Math.max(0, window.innerHeight - HEADER_HEIGHT));
    panelEl.style.left = `${left}px`;
    panelEl.style.top = `${top}px`;
}

function applyMinimized(min) {
    minimized = min;
    bodyEl.style.display = min ? "none" : "block";
    statusEl.style.display = min ? "none" : "block";
    panelEl.style.height = min ? `${HEADER_HEIGHT}px` : `${expandedHeight}px`;
    for (const handle of panelEl.querySelectorAll("[data-mxd-resize]")) {
        handle.style.display = min ? "none" : "block";
    }
    minimizeEl.textContent = min ? "+" : "–";
    minimizeEl.title = min ? "Expand live preview" : "Minimize live preview";
    localStorage.setItem(STORAGE_MINIMIZED, min ? "1" : "0");
}

function makeDraggable() {
    headerEl.style.cursor = "move";
    headerEl.addEventListener("pointerdown", (event) => {
        if (event.target === minimizeEl) return;
        event.preventDefault();
        headerEl.setPointerCapture(event.pointerId);

        const startX = event.clientX;
        const startY = event.clientY;
        const startLeft = parseFloat(panelEl.style.left);
        const startTop = parseFloat(panelEl.style.top);

        const onMove = (moveEvent) => {
            const left = clamp(startLeft + (moveEvent.clientX - startX), 0, window.innerWidth - 60);
            const top = clamp(startTop + (moveEvent.clientY - startY), 0, window.innerHeight - HEADER_HEIGHT);
            panelEl.style.left = `${left}px`;
            panelEl.style.top = `${top}px`;
        };
        const onUp = (upEvent) => {
            headerEl.releasePointerCapture(upEvent.pointerId);
            headerEl.removeEventListener("pointermove", onMove);
            headerEl.removeEventListener("pointerup", onUp);
            saveGeometry();
        };
        headerEl.addEventListener("pointermove", onMove);
        headerEl.addEventListener("pointerup", onUp);
    });
}

const RESIZE_DIRS = ["n", "s", "e", "w", "ne", "nw", "se", "sw"];
const RESIZE_CURSORS = {
    n: "ns-resize", s: "ns-resize",
    e: "ew-resize", w: "ew-resize",
    ne: "nesw-resize", sw: "nesw-resize",
    nw: "nwse-resize", se: "nwse-resize",
};

function makeResizable() {
    for (const dir of RESIZE_DIRS) {
        const handle = document.createElement("div");
        handle.dataset.mxdResize = dir;
        const edge = `${RESIZE_HANDLE_SIZE}px`;
        const style = {
            position: "absolute",
            cursor: RESIZE_CURSORS[dir],
            zIndex: 1,
        };
        if (dir.includes("n")) style.top = `-${RESIZE_HANDLE_SIZE / 2}px`;
        if (dir.includes("s")) style.bottom = `-${RESIZE_HANDLE_SIZE / 2}px`;
        if (dir.includes("w")) style.left = `-${RESIZE_HANDLE_SIZE / 2}px`;
        if (dir.includes("e")) style.right = `-${RESIZE_HANDLE_SIZE / 2}px`;
        if (dir === "n" || dir === "s") {
            style.left = edge; style.right = edge; style.height = edge;
        } else if (dir === "e" || dir === "w") {
            style.top = edge; style.bottom = edge; style.width = edge;
        } else {
            style.width = edge; style.height = edge;
        }
        Object.assign(handle.style, style);

        handle.addEventListener("pointerdown", (event) => {
            event.preventDefault();
            event.stopPropagation();
            handle.setPointerCapture(event.pointerId);

            const startX = event.clientX;
            const startY = event.clientY;
            const startLeft = parseFloat(panelEl.style.left);
            const startTop = parseFloat(panelEl.style.top);
            const startWidth = parseFloat(panelEl.style.width);
            const startHeight = expandedHeight;

            const onMove = (moveEvent) => {
                const dx = moveEvent.clientX - startX;
                const dy = moveEvent.clientY - startY;
                let left = startLeft, top = startTop, width = startWidth, height = startHeight;

                if (dir.includes("e")) width = clamp(startWidth + dx, MIN_WIDTH, window.innerWidth - startLeft);
                if (dir.includes("s")) height = clamp(startHeight + dy, MIN_HEIGHT, window.innerHeight - startTop);
                if (dir.includes("w")) {
                    width = clamp(startWidth - dx, MIN_WIDTH, startLeft + startWidth);
                    left = startLeft + (startWidth - width);
                }
                if (dir.includes("n")) {
                    height = clamp(startHeight - dy, MIN_HEIGHT, startTop + startHeight);
                    top = startTop + (startHeight - height);
                }

                panelEl.style.left = `${left}px`;
                panelEl.style.top = `${top}px`;
                panelEl.style.width = `${width}px`;
                expandedHeight = height;
                if (!minimized) panelEl.style.height = `${height}px`;
            };
            const onUp = (upEvent) => {
                handle.releasePointerCapture(upEvent.pointerId);
                handle.removeEventListener("pointermove", onMove);
                handle.removeEventListener("pointerup", onUp);
                saveGeometry();
            };
            handle.addEventListener("pointermove", onMove);
            handle.addEventListener("pointerup", onUp);
        });

        panelEl.appendChild(handle);
    }
}

function buildPanel() {
    if (panelEl) return;

    const geometry = loadGeometry();
    expandedHeight = geometry.height;

    panelEl = document.createElement("div");
    panelEl.id = "mxd-live-preview-panel";
    Object.assign(panelEl.style, {
        position: "fixed",
        left: `${geometry.left}px`,
        top: `${geometry.top}px`,
        width: `${geometry.width}px`,
        height: `${geometry.height}px`,
        background: "rgba(24,24,24,0.94)",
        border: "1px solid rgba(255,255,255,0.15)",
        borderRadius: "8px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
        zIndex: 9998,
        display: "none",
        flexDirection: "column",
        fontFamily: "sans-serif",
        color: "#eee",
    });

    headerEl = document.createElement("div");
    Object.assign(headerEl.style, {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: "6px",
        padding: "0 4px 0 8px",
        height: `${HEADER_HEIGHT}px`,
        flex: "none",
        background: "rgba(0,0,0,0.35)",
        borderRadius: "8px 8px 0 0",
        fontSize: "12px",
        userSelect: "none",
    });

    titleEl = document.createElement("span");
    titleEl.textContent = "Live Preview";
    Object.assign(titleEl.style, {
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        flex: "1",
    });

    minimizeEl = document.createElement("button");
    Object.assign(minimizeEl.style, {
        background: "transparent",
        border: "none",
        color: "#eee",
        cursor: "pointer",
        fontSize: "14px",
        lineHeight: "1",
        padding: "4px 6px",
    });
    minimizeEl.addEventListener("click", () => applyMinimized(!minimized));

    headerEl.appendChild(titleEl);
    headerEl.appendChild(minimizeEl);

    bodyEl = document.createElement("div");
    Object.assign(bodyEl.style, {
        flex: "1",
        minHeight: "0",
        position: "relative",
        overflow: "hidden",
    });

    canvasEl = document.createElement("canvas");
    Object.assign(canvasEl.style, {
        position: "absolute", inset: "0", width: "100%", height: "100%",
        objectFit: "contain", display: "block",
    });

    videoEl = document.createElement("video");
    Object.assign(videoEl.style, {
        position: "absolute", inset: "0", width: "100%", height: "100%",
        objectFit: "contain", display: "none", background: "#000",
    });
    videoEl.controls = true;
    videoEl.loop = true;
    videoEl.muted = true;
    videoEl.autoplay = true;
    videoEl.playsInline = true;

    statusEl = document.createElement("div");
    Object.assign(statusEl.style, {
        padding: "3px 8px",
        fontSize: "11px",
        opacity: "0.7",
        flex: "none",
    });

    bodyEl.appendChild(canvasEl);
    bodyEl.appendChild(videoEl);
    panelEl.appendChild(headerEl);
    panelEl.appendChild(bodyEl);
    panelEl.appendChild(statusEl);
    document.body.appendChild(panelEl);

    makeDraggable();
    makeResizable();
    window.addEventListener("resize", clampToViewport);

    applyMinimized(localStorage.getItem(STORAGE_MINIMIZED) === "1");
}

function showPanel() {
    buildPanel();
    panelEl.style.display = "flex";
}

function stopCanvasPlayback() {
    if (playTimer) {
        clearInterval(playTimer);
        playTimer = null;
    }
}

function startCanvasPlayback(rate) {
    stopCanvasPlayback();
    displayIndex = 0;
    canvasEl.style.display = "block";
    videoEl.style.display = "none";
    videoEl.pause();

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

function showSavedVideo(url, label) {
    stopCanvasPlayback();
    canvasEl.style.display = "none";
    videoEl.style.display = "block";
    videoEl.src = url;
    videoEl.play().catch(() => {});
    statusEl.textContent = "Finished";
    titleEl.textContent = label;
    try {
        localStorage.setItem(STORAGE_LAST, JSON.stringify({ url, label }));
    } catch (e) {
        // storage full/unavailable -- not worth failing over
    }
}

function restoreLastPreview() {
    try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_LAST) || "null");
        if (saved?.url) {
            showPanel();
            showSavedVideo(saved.url, saved.label);
        }
    } catch (e) {
        // ignore corrupt storage
    }
}

let listenersAttached = false;
function attachListeners() {
    if (listenersAttached) return;
    listenersAttached = true;

    api.addEventListener("MXD_live_preview_start", (event) => {
        const detail = event.detail;
        if (detail?.id == null) return;

        currentId = detail.id;
        frames = new Array(detail.length);
        showPanel();
        if (minimized) applyMinimized(false);
        titleEl.textContent = nodeLabelFor(detail.id);
        statusEl.textContent = "Live";
        startCanvasPlayback(detail.rate);
    });

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
        showPanel();
        showSavedVideo(url, nodeLabelFor(detail.node_id));
    });
}

app.registerExtension({
    name: "ComfyUI-MaxedOut.LivePreviewPanel",

    setup() {
        attachListeners();
        restoreLastPreview();
    },
});
