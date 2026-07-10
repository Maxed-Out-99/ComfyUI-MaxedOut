import { app } from "../../../scripts/app.js";

const NODE_TYPES = new Set(["WAN22_I2V_Video_Prep_MXD"]);

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function hideWidget(widget) {
    if (widget._mxdOriginalType === undefined) {
        widget._mxdOriginalType = widget.type;
        widget._mxdOriginalComputeSize = widget.computeSize;
    }
    widget.type = "hidden";
    widget.hidden = true;
    widget.disabled = true;
    widget.computeSize = () => [0, -4];
}

function showWidget(widget) {
    if (widget._mxdOriginalType !== undefined) {
        widget.type = widget._mxdOriginalType;
    }
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
    try {
        const computed = node.computeSize();
        const currentWidth = node.size?.[0] ?? computed[0];
        node.setSize([Math.max(currentWidth, computed[0]), computed[1]]);
    } catch (error) {
        // Don't let a layout hiccup block the toggle/redraw.
    }
}

function scheduleResizeNodeToWidgets(node) {
    resizeNodeToWidgets(node);
    requestAnimationFrame(() => {
        resizeNodeToWidgets(node);
        requestAnimationFrame(() => resizeNodeToWidgets(node));
    });
}

function updateTargetFpsVisibility(node) {
    const forceWidget = getWidget(node, "force_fps");
    const targetWidget = getWidget(node, "target_fps");
    if (!forceWidget || !targetWidget) {
        return;
    }

    if (forceWidget.value) {
        showWidget(targetWidget);
    } else {
        hideWidget(targetWidget);
    }

    scheduleResizeNodeToWidgets(node);
    app.canvas?.setDirty(true, true);
}

app.registerExtension({
    name: "ComfyUI-MaxedOut.Wan22VideoPrepMXD",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_TYPES.has(nodeData.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            const forceWidget = getWidget(node, "force_fps");

            if (forceWidget && !forceWidget._mxdWan22CallbackWrapped) {
                const originalCallback = forceWidget.callback;
                forceWidget.callback = function () {
                    const callbackResult = originalCallback?.apply(this, arguments);
                    updateTargetFpsVisibility(node);
                    return callbackResult;
                };
                forceWidget._mxdWan22CallbackWrapped = true;
            }

            updateTargetFpsVisibility(node);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            requestAnimationFrame(() => updateTargetFpsVisibility(this));
            return result;
        };
    },
});
