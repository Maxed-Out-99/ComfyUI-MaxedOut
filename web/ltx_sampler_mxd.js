import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LTX_SAMPLER_NODE_TYPES = new Set(["LTXKSampler_MXD", "LTXKSampler2_MXD"]);
const CUSTOM_SIGMAS_MODE = "Custom Sigmas";
const ltxPreviewImages = {};
const ltxPreviewTimers = {};
const ltxPreviewPaused = {};
const ltxPreviewAutoPaused = {};
const textDecoder = new TextDecoder();

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function hideWidget(widget) {
    if (!widget._mxdOriginalComputeSize) {
        widget._mxdOriginalComputeSize = widget.computeSize;
    }

    widget.hidden = true;
    widget.disabled = true;
    widget.computeSize = () => [0, -4];
}

function showWidget(widget) {
    widget.hidden = false;
    widget.disabled = false;

    if (widget._mxdOriginalComputeSize) {
        widget.computeSize = widget._mxdOriginalComputeSize;
    }
}

function resizeNodeToWidgets(node) {
    if (!node.computeSize || !node.setSize) {
        return;
    }

    const computed = node.computeSize();
    const currentWidth = node.size?.[0] ?? computed[0];
    node.setSize([Math.max(currentWidth, computed[0]), computed[1]]);
}

function updateCustomSigmasVisibility(node) {
    const modeWidget = getWidget(node, "mode");
    const sigmasWidget = getWidget(node, "custom_sigmas");

    if (!modeWidget || !sigmasWidget) {
        return;
    }

    if (modeWidget.value === CUSTOM_SIGMAS_MODE) {
        showWidget(sigmasWidget);
    } else {
        hideWidget(sigmasWidget);
    }

    resizeNodeToWidgets(node);
    app.canvas?.setDirty(true, true);
}

function getNodeById(id) {
    return app.graph?._nodes_by_id?.[id] ?? app.graph?.getNodeById?.(id);
}

function updatePauseButton(id, buttonEl) {
    if (!buttonEl) {
        return;
    }

    if (ltxPreviewPaused[id]) {
        buttonEl.textContent = "Play";
        buttonEl.title = "Resume latent preview playback";
    } else {
        buttonEl.textContent = "Pause";
        buttonEl.title = "Pause latent preview playback";
    }
}

function setLatentPreviewPaused(id, paused) {
    ltxPreviewPaused[id] = paused;
    const node = getNodeById(id);
    const widget = node ? getWidget(node, "ltxlatentpreview") : null;
    updatePauseButton(id, widget?.pauseEl);
}

function getPreviewContext(id, width, height) {
    const node = getNodeById(id);
    if (!node) {
        return null;
    }

    let widget = getWidget(node, "ltxlatentpreview");
    if (!widget) {
        const previewEl = document.createElement("div");
        previewEl.style.width = "100%";
        previewEl.style.position = "relative";

        const canvasEl = document.createElement("canvas");
        canvasEl.style.width = "100%";
        canvasEl.style.display = "block";
        previewEl.appendChild(canvasEl);

        const pauseEl = document.createElement("button");
        pauseEl.textContent = "Pause";
        pauseEl.style.position = "absolute";
        pauseEl.style.right = "6px";
        pauseEl.style.bottom = "6px";
        pauseEl.style.padding = "1px 6px";
        pauseEl.style.fontSize = "11px";
        pauseEl.style.lineHeight = "1.2";
        pauseEl.style.opacity = "0.85";
        pauseEl.style.cursor = "pointer";
        previewEl.appendChild(pauseEl);

        widget = node.addDOMWidget("ltxlatentpreview", "ltxcanvas", previewEl, {
            serialize: false,
            hideOnZoom: false,
        });
        widget.serialize = false;
        widget.canvasEl = canvasEl;
        widget.pauseEl = pauseEl;
        widget.computeSize = function (availableWidth) {
            if (!this.aspectRatio) {
                return [availableWidth, -4];
            }
            return [availableWidth, (node.size[0] - 20) / this.aspectRatio + 10];
        };

        pauseEl.addEventListener("pointerdown", (event) => {
            event.preventDefault();
            event.stopImmediatePropagation();
            event.stopPropagation();
        }, true);
        pauseEl.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopImmediatePropagation();
            event.stopPropagation();
            setLatentPreviewPaused(id, !ltxPreviewPaused[id]);
        }, true);
    }
    updatePauseButton(id, widget.pauseEl);

    const canvasEl = widget.canvasEl || widget.element;
    if (canvasEl.width !== width || canvasEl.height !== height) {
        widget.aspectRatio = width / height;
        canvasEl.width = width;
        canvasEl.height = height;
        resizeNodeToWidgets(node);
    }
    return canvasEl.getContext("2d");
}

function beginLatentPreview(id, rate) {
    clearInterval(ltxPreviewTimers[id]);
    let displayIndex = 0;
    ltxPreviewAutoPaused[id] = false;
    setLatentPreviewPaused(id, false);
    const startNode = getNodeById(id);
    if (startNode) {
        startNode.progress = 0;
    }

    ltxPreviewTimers[id] = setInterval(() => {
        const node = getNodeById(id);
        if (!node) {
            clearInterval(ltxPreviewTimers[id]);
            delete ltxPreviewTimers[id];
            delete ltxPreviewAutoPaused[id];
            return;
        }
        if (node.progress == null) {
            if (!ltxPreviewAutoPaused[id]) {
                ltxPreviewAutoPaused[id] = true;
                setLatentPreviewPaused(id, true);
            }
        } else {
            ltxPreviewAutoPaused[id] = false;
        }
        if (ltxPreviewPaused[id]) {
            return;
        }

        const images = ltxPreviewImages[id];
        const image = images?.[displayIndex];
        if (!image) {
            return;
        }
        getPreviewContext(id, image.width, image.height)?.drawImage(image, 0, 0);
        displayIndex = (displayIndex + 1) % images.length;
        app.canvas?.setDirty(true, true);
    }, 1000 / Math.max(1, rate || 8));
}

api.addEventListener("VHS_latentpreview", ({ detail }) => {
    if (detail.id == null) {
        return;
    }

    ltxPreviewImages[detail.id] = [];
    ltxPreviewImages[detail.id].length = detail.length;
    const idParts = String(detail.id).split(":");
    for (let i = 1; i <= idParts.length; i++) {
        const id = idParts.slice(0, i).join(":");
        ltxPreviewImages[id] = ltxPreviewImages[detail.id];
        beginLatentPreview(id, detail.rate);
    }
});

api.addEventListener("b_preview", async (event) => {
    if (Object.keys(ltxPreviewTimers).length === 0) {
        return;
    }

    const header = new DataView(await event.detail.slice(0, 24).arrayBuffer());
    const index = header.getUint32(4);
    const idLength = header.getUint8(8);
    const id = textDecoder.decode(header.buffer.slice(9, 9 + idLength));
    const images = ltxPreviewImages[id];
    if (!images) {
        return;
    }

    event.preventDefault();
    event.stopImmediatePropagation();
    event.stopPropagation();
    images[index] = await window.createImageBitmap(event.detail.slice(24));
}, true);

app.registerExtension({
    name: "ComfyUI-MaxedOut.LTXSamplerMXD",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!LTX_SAMPLER_NODE_TYPES.has(nodeData.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            const modeWidget = getWidget(node, "mode");

            if (modeWidget && !modeWidget._mxdLtxCallbackWrapped) {
                const originalCallback = modeWidget.callback;
                modeWidget.callback = function () {
                    const callbackResult = originalCallback?.apply(this, arguments);
                    updateCustomSigmasVisibility(node);
                    return callbackResult;
                };
                modeWidget._mxdLtxCallbackWrapped = true;
            }

            updateCustomSigmasVisibility(node);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            requestAnimationFrame(() => updateCustomSigmasVisibility(this));
            return result;
        };
    },
});
