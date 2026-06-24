import { app } from "../../scripts/app.js";

const NODE_TYPES = new Set(["WAN22_I2V_Video_Prep_MXD"]);

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function hideWidget(widget) {
    if (!widget._mxdOriginalComputeSize) {
        widget._mxdOriginalComputeSize = widget.computeSize;
    }
    widget.hidden = true;
    widget.disabled = true;
    widget.computeSize = () => [0, 0];
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
    let requiredHeight = computed[1];
    for (const widget of node.widgets ?? []) {
        if (widget.hidden) {
            continue;
        }

        const widgetY = Number.isFinite(widget.last_y) ? widget.last_y : 0;
        let widgetHeight = 20;
        try {
            const size = widget.computeSize?.(currentWidth);
            if (Array.isArray(size) && Number.isFinite(size[1])) {
                widgetHeight = Math.max(widgetHeight, size[1]);
            }
        } catch (error) {
            // Keep the fallback height.
        }
        requiredHeight = Math.max(requiredHeight, widgetY + widgetHeight + 8);
    }
    node.setSize([Math.max(currentWidth, computed[0]), Math.ceil(requiredHeight)]);
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
