import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_TYPES = new Set(["Load Video Batch MXD"]);

// Cache the folder lists for both sources so flipping the "source" widget is
// instant and never needs a page refresh. Uses the video-specific route so
// only folders that actually contain a video (recursively) are listed.
let _foldersPromise = null;
function fetchFolders() {
    if (!_foldersPromise) {
        _foldersPromise = api
            .fetchApi("/mxd/video_batch/folders")
            .then((resp) => resp.json())
            .catch((err) => {
                console.error("[LoadVideoBatchMXD] Failed to fetch folders:", err);
                _foldersPromise = null; // allow retry later
                return { outputs: [""], inputs: [""] };
            });
    }
    return _foldersPromise;
}

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function applySource(node, folders) {
    const sourceWidget = getWidget(node, "source");
    const folderWidget = getWidget(node, "folder");
    if (!sourceWidget || !folderWidget) {
        return;
    }

    const source = sourceWidget.value === "inputs" ? "inputs" : "outputs";
    const list = Array.isArray(folders?.[source]) && folders[source].length
        ? folders[source]
        : [""];

    folderWidget.options = folderWidget.options || {};
    folderWidget.options.values = list;

    // Keep the current selection if it still exists for this source (allowing for leading space differences),
    // otherwise fall back to the root ("").
    const cleanWidgetVal = folderWidget.value?.trim().replace(/^\u00a0+/, "") || "";
    const matched = list.find((v) => (v?.trim().replace(/^\u00a0+/, "") || "") === cleanWidgetVal);
    if (matched) {
        folderWidget.value = matched;
    } else {
        folderWidget.value = list[0];
        folderWidget.callback?.(folderWidget.value);
    }

    app.canvas?.setDirty(true, true);
}

app.registerExtension({
    name: "ComfyUI-MaxedOut.LoadVideoBatchMXD",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_TYPES.has(nodeData.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            const sourceWidget = getWidget(node, "source");

            if (sourceWidget && !sourceWidget._mxdBatchCallbackWrapped) {
                const original = sourceWidget.callback;
                sourceWidget.callback = function () {
                    const r = original?.apply(this, arguments);
                    fetchFolders().then((folders) => applySource(node, folders));
                    return r;
                };
                sourceWidget._mxdBatchCallbackWrapped = true;
            }

            fetchFolders().then((folders) => applySource(node, folders));
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            const node = this;
            // Defer so the saved widget values are restored before we filter.
            fetchFolders().then((folders) =>
                requestAnimationFrame(() => applySource(node, folders))
            );
            return result;
        };
    },
});
