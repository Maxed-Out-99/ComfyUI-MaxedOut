import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// nodeType -> { widget: pickerWidgetName, kind: "image" | "video" }
const NODE_CONFIG = new Map([
    ["LoadImageFromFolderMXD", { widget: "image", kind: "image" }],
    ["LoadVideoFromFolderMXD", { widget: "video", kind: "video" }],
]);

// Cache the file lists per kind so flipping the "source" widget is instant
// and never needs a page refresh.
const _filesPromise = new Map();
function fetchFiles(kind) {
    if (!_filesPromise.has(kind)) {
        _filesPromise.set(
            kind,
            api
                .fetchApi(`/mxd/single_loader/files?kind=${kind}`)
                .then((resp) => resp.json())
                .catch((err) => {
                    console.error(`[LoadSingleFileMXD] Failed to fetch ${kind} files:`, err);
                    _filesPromise.delete(kind);
                    return { outputs: [""], inputs: [""] };
                })
        );
    }
    return _filesPromise.get(kind);
}

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function applySource(node, config, files) {
    const sourceWidget = getWidget(node, "source");
    const fileWidget = getWidget(node, config.widget);
    if (!sourceWidget || !fileWidget) {
        return;
    }

    const source = sourceWidget.value === "inputs" ? "inputs" : "outputs";
    const list = Array.isArray(files?.[source]) && files[source].length ? files[source] : [""];

    fileWidget.options = fileWidget.options || {};
    fileWidget.options.values = list;

    // Keep the current selection if it still exists for this source, otherwise
    // fall back to the first (newest) entry.
    const matched = list.find((v) => v === fileWidget.value);
    if (matched) {
        fileWidget.value = matched;
    } else {
        fileWidget.value = list[0];
        fileWidget.callback?.(fileWidget.value);
    }

    app.canvas?.setDirty(true, true);
}

app.registerExtension({
    name: "ComfyUI-MaxedOut.LoadSingleFileFromFolder",

    beforeRegisterNodeDef(nodeType, nodeData) {
        const config = NODE_CONFIG.get(nodeData.name);
        if (!config) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            const sourceWidget = getWidget(node, "source");

            if (sourceWidget && !sourceWidget._mxdSingleFileCallbackWrapped) {
                const original = sourceWidget.callback;
                sourceWidget.callback = function () {
                    const r = original?.apply(this, arguments);
                    fetchFiles(config.kind).then((files) => applySource(node, config, files));
                    return r;
                };
                sourceWidget._mxdSingleFileCallbackWrapped = true;
            }

            fetchFiles(config.kind).then((files) => applySource(node, config, files));
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            const node = this;
            // Defer so the saved widget values are restored before we filter.
            fetchFiles(config.kind).then((files) =>
                requestAnimationFrame(() => applySource(node, config, files))
            );
            return result;
        };
    },
});
