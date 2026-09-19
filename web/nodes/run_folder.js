import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Node types that get the "run_folder" toggle (declared server-side), mapped
// to the name of their single-file picker widget.
const NODE_TYPES = new Map([
    ["LoadLatent_I2V_MXD", "latent"],
    ["LoadLatent_I2V_Pipe_MXD", "latent"],
    ["LoadImageFromFolderMXD", "image"],
    ["LoadVideoFromFolderMXD", "video"],
]);

// Server routes that return a fresh, unfiltered disk scan for a picker kind
// (flat array of relative paths). Kinds without an entry here just skip the
// refresh_before_run step, since no such node currently exposes that widget.
const REFRESH_ROUTES = new Map([
    ["latent", "/mxd/latents/files"],
]);

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function dirOf(path) {
    if (!path) return "";
    const norm = String(path).replace(/\\/g, "/");
    const idx = norm.lastIndexOf("/");
    return idx === -1 ? "" : norm.slice(0, idx);
}

// Re-scan disk for this node's file kind and swap in the fresh list, so a
// run_folder loop collected right after sees files a still-running workflow
// wrote in the meantime instead of whatever the dropdown had cached.
async function refreshPickerOptions(pickerWidget, kind) {
    const route = REFRESH_ROUTES.get(kind);
    if (!route) return;
    try {
        const resp = await api.fetchApi(route);
        const files = await resp.json();
        if (Array.isArray(files) && files.length) {
            pickerWidget.options = pickerWidget.options || {};
            pickerWidget.options.values = files;
        }
    } catch (err) {
        console.error(`[RunFolderMXD] Failed to refresh ${kind} files:`, err);
    }
}

// LiteGraph node modes. Bypassed nodes still pass data down the chain, so they
// stay traversable; muted ones cut the branch dead.
const MODE_ALWAYS = 0;
const MODE_NEVER = 2;
const MODE_BYPASS = 4;

function isOutputNode(node) {
    return !!node?.constructor?.nodeData?.output_node;
}

// Walk forward from this node's output slots looking for an output node
// (save/preview/etc). A node that reaches none of them is dropped by the
// backend before execution, so looping it would queue runs that do nothing
// but repeat the rest of the graph.
function feedsAnOutputNode(node) {
    const graph = node.graph || app.graph;
    const seen = new Set([node.id]);
    const queue = [node];

    while (queue.length) {
        const current = queue.shift();
        for (const output of current.outputs || []) {
            for (const linkId of output.links || []) {
                const link = graph?.links?.[linkId] ?? graph?.links?.get?.(linkId);
                if (!link) continue;

                const target = graph.getNodeById?.(link.target_id);
                if (!target || seen.has(target.id)) continue;
                if (target.mode === MODE_NEVER) continue;

                if (isOutputNode(target) && target.mode === MODE_ALWAYS) return true;
                seen.add(target.id);
                queue.push(target);
            }
        }
    }
    return false;
}

// A node only drives the loop if it will actually execute: not muted or
// bypassed itself, and wired into something that produces a result.
function participatesInRun(node) {
    if (node.mode === MODE_NEVER || node.mode === MODE_BYPASS) return false;
    return feedsAnOutputNode(node);
}

// Build the per-step driver for one run_folder node: walk its picker widget
// across every file sitting in the same folder as the current selection.
async function runFolderDriver(node, pickerName) {
    const runWidget = getWidget(node, "run_folder");
    const pickerWidget = getWidget(node, pickerName);
    if (!runWidget || !pickerWidget || !runWidget.value) return null;

    const refreshWidget = getWidget(node, "refresh_before_run");
    if (refreshWidget?.value) {
        await refreshPickerOptions(pickerWidget, pickerName);
    }

    const originalValue = pickerWidget.value;
    const dir = dirOf(originalValue);
    const files = (pickerWidget.options?.values || []).filter((v) => dirOf(v) === dir);
    if (files.length <= 1) return null;

    const set = (v) => {
        pickerWidget.value = v;
        pickerWidget.callback?.(v);
    };
    return {
        steps: files.length,
        apply: (i) => set(files[Math.min(i, files.length - 1)]),
        restore: () => set(originalValue),
    };
}

// Collect every node currently asking for a multi-run loop. All of them
// advance together, so a mixed graph runs max(steps) times with each node
// clamping to its own last entry.
async function collectActiveDrivers() {
    const nodes = app.graph?._nodes || [];
    const active = [];

    for (const node of nodes) {
        const pickerName = NODE_TYPES.get(node.comfyClass);
        if (!pickerName || !participatesInRun(node)) continue;

        const driver = await runFolderDriver(node, pickerName);
        if (driver) active.push(driver);
    }

    return active;
}

app.registerExtension({
    name: "ComfyUI-MaxedOut.RunFolder",

    setup() {
        const originalQueuePrompt = app.queuePrompt.bind(app);

        app.queuePrompt = async function (...args) {
            const active = await collectActiveDrivers();
            if (!active.length) {
                return originalQueuePrompt(...args);
            }

            const steps = Math.max(...active.map((a) => a.steps));
            try {
                for (let i = 0; i < steps; i++) {
                    for (const a of active) a.apply(i);
                    app.canvas?.setDirty(true, true);
                    await originalQueuePrompt(...args);
                }
            } finally {
                for (const a of active) a.restore();
                app.canvas?.setDirty(true, true);
            }
        };
    },
});
