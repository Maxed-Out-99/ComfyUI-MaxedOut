import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

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

// Find every node with run_folder=true and expand it to the list of files
// that live in the same folder as its currently selected file.
async function collectActiveRunFolderNodes() {
    const nodes = app.graph?._nodes || [];
    const active = [];

    for (const node of nodes) {
        const pickerName = NODE_TYPES.get(node.comfyClass);
        if (!pickerName) continue;

        const runWidget = getWidget(node, "run_folder");
        const pickerWidget = getWidget(node, pickerName);
        if (!runWidget || !pickerWidget || !runWidget.value) continue;

        const refreshWidget = getWidget(node, "refresh_before_run");
        if (refreshWidget?.value) {
            await refreshPickerOptions(pickerWidget, pickerName);
        }

        const originalValue = pickerWidget.value;
        const dir = dirOf(originalValue);
        const files = (pickerWidget.options?.values || []).filter((v) => dirOf(v) === dir);
        if (files.length <= 1) continue;

        active.push({ node, widget: pickerWidget, files, originalValue });
    }

    return active;
}

app.registerExtension({
    name: "ComfyUI-MaxedOut.RunFolder",

    setup() {
        const originalQueuePrompt = app.queuePrompt.bind(app);

        app.queuePrompt = async function (...args) {
            const active = await collectActiveRunFolderNodes();
            if (!active.length) {
                return originalQueuePrompt(...args);
            }

            const steps = Math.max(...active.map((a) => a.files.length));
            try {
                for (let i = 0; i < steps; i++) {
                    for (const a of active) {
                        const file = a.files[Math.min(i, a.files.length - 1)];
                        a.widget.value = file;
                        a.widget.callback?.(file);
                    }
                    app.canvas?.setDirty(true, true);
                    await originalQueuePrompt(...args);
                }
            } finally {
                for (const a of active) {
                    a.widget.value = a.originalValue;
                    a.widget.callback?.(a.originalValue);
                }
                app.canvas?.setDirty(true, true);
            }
        };
    },
});
