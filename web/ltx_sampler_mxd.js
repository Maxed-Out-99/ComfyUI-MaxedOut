import { app } from "../../scripts/app.js";

const LTX_SAMPLER_NODE_TYPES = new Set(["LTXKSampler_MXD", "LTXKSampler2_MXD"]);
const CUSTOM_SIGMAS_MODE = "Custom Sigmas";

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
