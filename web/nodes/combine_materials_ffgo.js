import { app } from "../../../scripts/app.js";

// Grows/shrinks the foreground_N input sockets as they're connected, the same
// way reference/conditioning-image nodes reveal a new empty slot once the
// current last one is plugged in. Backend declares foreground_1 (required)
// plus foreground_2..foreground_6 (optional) as individual IMAGE sockets (see
// MAX_FOREGROUNDS in nodes/ffgo.py) so arbitrarily-sized RGBA
// cutouts never have to be batched into one uniform IMAGE tensor.
//
// MAX_FOREGROUNDS must stay in sync with the Python-side constant — it's a
// fixed ceiling, not something to infer from the node's current sockets
// (those get pruned down to foreground_1 immediately, so scanning them back
// would collapse the ceiling to 1 and the slots could never grow again).

const NODE_NAME = "CombineForegroundsBackgroundFFGOMXD";
const MAX_FOREGROUNDS = 6;
const FOREGROUND_RE = /^foreground_(\d+)$/;

function syncForegroundInputs(node) {
    if (!node.inputs) return;

    let maxConnected = 0;
    for (const inp of node.inputs) {
        const m = FOREGROUND_RE.exec(inp.name);
        if (m && inp.link != null) {
            maxConnected = Math.max(maxConnected, parseInt(m[1], 10));
        }
    }

    const target = Math.min(maxConnected + 1, MAX_FOREGROUNDS);

    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const inp = node.inputs[i];
        const m = FOREGROUND_RE.exec(inp?.name);
        if (m) {
            const n = parseInt(m[1], 10);
            if (n > target && inp.link == null) {
                node.removeInput(i);
            }
        }
    }

    for (let n = 1; n <= target; n++) {
        const name = `foreground_${n}`;
        if (!node.inputs.some((inp) => inp.name === name)) {
            node.addInput(name, "IMAGE");
        }
    }

    node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "ComfyUI-MaxedOut.CombineForegroundsBackgroundFFGOMXD",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            syncForegroundInputs(this);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo, ioSlot) {
            const result = onConnectionsChange?.apply(this, arguments);
            if (type === LiteGraph.INPUT) {
                syncForegroundInputs(this);
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            const node = this;
            requestAnimationFrame(() => syncForegroundInputs(node));
            return result;
        };
    },
});
