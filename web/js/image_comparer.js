import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// --- Constants ---
const NODE_TYPE_STRING = "Image Comparer + Save MXD";
const NODE_TYPE_STRINGS = new Set([
    NODE_TYPE_STRING,
    "Video Comparer MXD",
    "VideoComparerMXD",
]);
const NODE_OVERRIDE_CLASSES = new Map();

// --- Canvas Utilities (Inline) ---
function measureText(ctx, str) {
    return ctx.measureText(str).width;
}

// --- Widget Utilities (Base Class) ---
class MxdBaseWidget {
    constructor(name) {
        this.type = "custom";
        this.options = {};
        this.y = 0;
        this.last_y = 0;
        this.mouseDowned = null;
        this.isMouseDownedAndOver = false;
        this.hitAreas = {};
        this.downedHitAreasForMove = [];
        this.downedHitAreasForClick = [];
        this.name = name;
    }
    serializeValue(node, index) {
        return this.value;
    }
    clickWasWithinBounds(pos, bounds) {
        let xStart = bounds[0];
        let xEnd = xStart + (bounds.length > 2 ? bounds[2] : bounds[1]);
        const clickedX = pos[0] >= xStart && pos[0] <= xEnd;
        if (bounds.length === 2) {
            return clickedX;
        }
        return clickedX && pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3];
    }
    mouse(event, pos, node) {
        var _a, _b, _c;
        if (event.type == "pointerdown") {
            this.mouseDowned = [...pos];
            this.isMouseDownedAndOver = true;
            this.downedHitAreasForMove.length = 0;
            this.downedHitAreasForClick.length = 0;
            let anyHandled = false;
            for (const part of Object.values(this.hitAreas)) {
                if (this.clickWasWithinBounds(pos, part.bounds)) {
                    if (part.onMove) {
                        this.downedHitAreasForMove.push(part);
                    }
                    if (part.onClick) {
                        this.downedHitAreasForClick.push(part);
                    }
                    if (part.onDown) {
                        const thisHandled = part.onDown.apply(this, [event, pos, node, part]);
                        anyHandled = anyHandled || thisHandled == true;
                    }
                    part.wasMouseClickedAndIsOver = true;
                }
            }
            return (_a = this.onMouseDown(event, pos, node)) !== null && _a !== void 0 ? _a : anyHandled;
        }
        if (event.type == "pointerup") {
            if (!this.mouseDowned)
                return true;
            this.downedHitAreasForMove.length = 0;
            const wasMouseDownedAndOver = this.isMouseDownedAndOver;
            this.cancelMouseDown();
            let anyHandled = false;
            for (const part of Object.values(this.hitAreas)) {
                if (part.onUp && this.clickWasWithinBounds(pos, part.bounds)) {
                    const thisHandled = part.onUp.apply(this, [event, pos, node, part]);
                    anyHandled = anyHandled || thisHandled == true;
                }
                part.wasMouseClickedAndIsOver = false;
            }
            for (const part of this.downedHitAreasForClick) {
                if (this.clickWasWithinBounds(pos, part.bounds)) {
                    const thisHandled = part.onClick.apply(this, [event, pos, node, part]);
                    anyHandled = anyHandled || thisHandled == true;
                }
            }
            this.downedHitAreasForClick.length = 0;
            if (wasMouseDownedAndOver) {
                const thisHandled = this.onMouseClick(event, pos, node);
                anyHandled = anyHandled || thisHandled == true;
            }
            return (_b = this.onMouseUp(event, pos, node)) !== null && _b !== void 0 ? _b : anyHandled;
        }
        if (event.type == "pointermove") {
            this.isMouseDownedAndOver = !!this.mouseDowned;
            if (this.mouseDowned &&
                (pos[0] < 15 ||
                    pos[0] > node.size[0] - 15 ||
                    pos[1] < this.last_y ||
                    pos[1] > this.last_y + LiteGraph.NODE_WIDGET_HEIGHT)) {
                this.isMouseDownedAndOver = false;
            }
            for (const part of Object.values(this.hitAreas)) {
                if (this.downedHitAreasForMove.includes(part)) {
                    part.onMove.apply(this, [event, pos, node, part]);
                }
                if (this.downedHitAreasForClick.includes(part)) {
                    part.wasMouseClickedAndIsOver = this.clickWasWithinBounds(pos, part.bounds);
                }
            }
            return (_c = this.onMouseMove(event, pos, node)) !== null && _c !== void 0 ? _c : true;
        }
        return false;
    }
    cancelMouseDown() {
        this.mouseDowned = null;
        this.isMouseDownedAndOver = false;
        this.downedHitAreasForMove.length = 0;
    }
    onMouseDown(event, pos, node) { return; }
    onMouseUp(event, pos, node) { return; }
    onMouseClick(event, pos, node) { return; }
    onMouseMove(event, pos, node) { return; }
}

// --- Helper Functions ---
function mediaDataToUrl(data, mediaType = "image") {
    const previewParam = mediaType === "image" ? app.getPreviewFormatParam() : "";
    const subfolder = data.subfolder || "";
    return api.apiURL(
        `/view?filename=${encodeURIComponent(data.filename)}&type=${data.type}&subfolder=${encodeURIComponent(subfolder)}${previewParam}${app.getRandParam()}`
    );
}


function addConnectionLayoutSupport(node, app, options = [["Left", "Right"], ["Right", "Left"]]) {
    // Simplified version: just ensures the prototype methods exist to avoid errors if called
    node.prototype.getConnectionPos = function(isInput, slotNumber, out) {
        return LGraphNode.prototype.getConnectionPos.call(this, isInput, slotNumber, out);
    };
}

// --- Simplified Base Server Node ---
class MxdBaseServerNode extends LGraphNode {
    constructor(title) {
        super(title);
        this.serialize_widgets = true;
        this.setupFromServerNodeData();
    }
    getWidgets() {
        return ComfyWidgets;
    }
    async setupFromServerNodeData() {
        var _a, _b, _c, _d, _e;
        const nodeData = this.constructor.nodeData;
        if (!nodeData) return;
        
        this.comfyClass = nodeData.name;
        let inputs = nodeData["input"]["required"];
        if (nodeData["input"]["optional"] != undefined) {
            inputs = Object.assign({}, inputs, nodeData["input"]["optional"]);
        }
        const WIDGETS = this.getWidgets();
        const config = { minWidth: 1, minHeight: 1, widget: null };
        
        for (const inputName in inputs) {
            const inputData = inputs[inputName];
            const type = inputData[0];
            if ((_a = inputData[1]) === null || _a === void 0 ? void 0 : _a.forceInput) {
                this.addInput(inputName, type);
            } else {
                let widgetCreated = true;
                if (Array.isArray(type)) {
                    Object.assign(config, WIDGETS.COMBO(this, inputName, inputData, app) || {});
                } else if (`${type}:${inputName}` in WIDGETS) {
                    Object.assign(config, WIDGETS[`${type}:${inputName}`](this, inputName, inputData, app) || {});
                } else if (type in WIDGETS) {
                    Object.assign(config, WIDGETS[type](this, inputName, inputData, app) || {});
                } else {
                    this.addInput(inputName, type);
                    widgetCreated = false;
                }
            }
        }
        for (const o in nodeData["output"]) {
            let output = nodeData["output"][o];
            if (output instanceof Array) output = "COMBO";
            const outputName = nodeData["output_name"][o] || output;
            const outputShape = nodeData["output_is_list"][o] ? LiteGraph.GRID_SHAPE : LiteGraph.CIRCLE_SHAPE;
            this.addOutput(outputName, output, { shape: outputShape });
        }
        
        const s = this.computeSize();
        s[0] = Math.max((_d = config.minWidth) !== null && _d !== void 0 ? _d : 1, s[0] * 1.5);
        s[1] = Math.max((_e = config.minHeight) !== null && _e !== void 0 ? _e : 1, s[1]);
        this.size = s;
        this.serialize_widgets = true;
    }

    static registerForOverride(comfyClass, nodeData, mxdClass) {
        mxdClass.nodeData = nodeData;
        
        // This hooks into LiteGraph to use our custom class instead of the default
        const oldRegister = LiteGraph.registerNodeType;
        LiteGraph.registerNodeType = function(nodeId, baseClass) {
             if (nodeId === nodeData.name || baseClass.type === nodeData.name) {
                 return oldRegister.call(LiteGraph, nodeId, mxdClass);
             }
             return oldRegister.call(LiteGraph, nodeId, baseClass);
        }
    }
}

// --- Image Comparer Node & Widget ---

class MxdImageComparerWidget extends MxdBaseWidget {
    constructor(name, node) {
        super(name);
        this.type = "custom";
        this.hitAreas = {};
        this.selected = [];
        this._value = { images: [] };
        this.node = node;
        this._autoSize = { width: 0, height: 0 };
    }
    set value(v) {
        let cleanedVal;
        if (Array.isArray(v)) {
            cleanedVal = v.map((d, i) => {
                if (!d || typeof d === "string") {
                    d = { url: d, name: i == 0 ? "A" : "B", selected: true, mediaType: "image" };
                }
                d.mediaType = d.mediaType || "image";
                return d;
            });
        }
        else {
            cleanedVal = (v.images || []).map((d) => ({ mediaType: "image", ...d }));
        }
        if (cleanedVal.length > 2) {
            const hasAAndB = cleanedVal.some((i) => i.name.startsWith("A")) &&
                cleanedVal.some((i) => i.name.startsWith("B"));
            if (!hasAAndB) {
                cleanedVal = [cleanedVal[0], cleanedVal[1]];
            }
        }
        let selected = cleanedVal.filter((d) => d.selected);
        if (!selected.length && cleanedVal.length) {
            cleanedVal[0].selected = true;
        }
        selected = cleanedVal.filter((d) => d.selected);
        if (selected.length === 1 && cleanedVal.length > 1) {
            cleanedVal.find((d) => !d.selected).selected = true;
        }
        this._value.images = cleanedVal;
        selected = cleanedVal.filter((d) => d.selected);
        this.setSelected(selected);
    }
    get value() {
        return this._value;
    }
    createMediaElement(sel) {
        if (sel.mediaType === "video") {
            const video = document.createElement("video");
            video.src = sel.url;
            video.muted = true;
            video.loop = true;
            video.autoplay = true;
            video.playsInline = true;
            video.preload = "auto";
            video.onloadeddata = () => this.node.setDirtyCanvas(true, true);
            video.play().catch(() => { });
            return video;
        }
        const img = new Image();
        img.src = sel.url;
        return img;
    }
    getMediaSize(media) {
        if (!media) {
            return null;
        }
        if (media instanceof HTMLVideoElement) {
            if (!media.videoWidth || !media.videoHeight) {
                return null;
            }
            return { width: media.videoWidth, height: media.videoHeight };
        }
        if (!media.naturalWidth || !media.naturalHeight) {
            return null;
        }
        return { width: media.naturalWidth, height: media.naturalHeight };
    }
    setSelected(selected) {
        this._value.images.forEach((d) => (d.selected = false));
        this.node.imgs.length = 0;
        for (const sel of selected) {
            if (!sel.img) {
                sel.img = this.createMediaElement(sel);
                this.node.imgs.push(sel.img);
            }
            if (sel.mediaType === "video" && sel.img instanceof HTMLVideoElement) {
                sel.img.play().catch(() => { });
            }
            sel.selected = true;
        }
        this.selected = selected;
    }
    draw(ctx, node, width, y) {
        var _a;
        this.hitAreas = {};
        if (this.value.images.length > 2) {
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.font = `14px Arial`;
            const drawData = [];
            const spacing = 5;
            let x = 0;
            for (const img of this.value.images) {
                const width = measureText(ctx, img.name);
                drawData.push({
                    img,
                    text: img.name,
                    x,
                    width: measureText(ctx, img.name),
                });
                x += width + spacing;
            }
            x = (node.size[0] - (x - spacing)) / 2;
            for (const d of drawData) {
                ctx.fillStyle = d.img.selected ? "rgba(180, 180, 180, 1)" : "rgba(180, 180, 180, 0.5)";
                ctx.fillText(d.text, x, y);
                this.hitAreas[d.text] = {
                    bounds: [x, y, d.width, 14],
                    data: d.img,
                    onDown: this.onSelectionDown,
                };
                x += d.width + spacing;
            }
            y += 20;
        }
        const isClickMode = ((_a = node.properties) === null || _a === void 0 ? void 0 : _a["comparer_mode"]) === "Click";
        if (isClickMode) {
            const image = this.selected[this.node.isPointerDown ? 1 : 0];
            this.updateAutoSize(image, y);
            this.drawMedia(ctx, image, y);
        }
        else {
            const image = this.selected[0];
            this.updateAutoSize(image, y);
            this.drawMedia(ctx, image, y);
            if (node.isPointerOver) {
                this.drawMedia(ctx, this.selected[1], y, this.node.pointerOverPos[0]);
            }
        }
        if (this.selected.some((item) => item.mediaType === "video")) {
            this.node.setDirtyCanvas(true, false);
        }
    }
    updateAutoSize(image, y) {
        const initial = this.node._mxdInitialSize;
        if (initial && (this.node.size[0] !== initial[0] || this.node.size[1] !== initial[1])) {
            return;
        }
        const mediaSize = this.getMediaSize(image === null || image === void 0 ? void 0 : image.img);
        if (!mediaSize) {
            return;
        }
        const nodeWidth = this.node.size[0];
        const imageAspect = mediaSize.width / mediaSize.height;
        const desiredImageHeight = Math.round(nodeWidth / imageAspect);
        const desiredHeight = y + desiredImageHeight;
        if (desiredHeight > this.node.size[1]) {
            this.node.setSize([nodeWidth, desiredHeight]);
            this.node.setDirtyCanvas(true, true);
        }
        this._autoSize.width = nodeWidth;
        this._autoSize.height = Math.max(this._autoSize.height, desiredHeight);
    }
    onSelectionDown(event, pos, node, bounds) {
        const selected = [...this.selected];
        if (bounds === null || bounds === void 0 ? void 0 : bounds.data.name.startsWith("A")) {
            selected[0] = bounds.data;
        }
        else if (bounds === null || bounds === void 0 ? void 0 : bounds.data.name.startsWith("B")) {
            selected[1] = bounds.data;
        }
        this.setSelected(selected);
    }
    drawMedia(ctx, image, y, cropX) {
        const mediaSize = this.getMediaSize(image === null || image === void 0 ? void 0 : image.img);
        if (!mediaSize) {
            return;
        }
        let [nodeWidth, nodeHeight] = this.node.size;
        const imageAspect = mediaSize.width / mediaSize.height;
        let height = nodeHeight - y;
        const widgetAspect = nodeWidth / height;
        let targetWidth, targetHeight;
        let offsetX = 0;
        if (imageAspect > widgetAspect) {
            targetWidth = nodeWidth;
            targetHeight = nodeWidth / imageAspect;
        }
        else {
            targetHeight = height;
            targetWidth = height * imageAspect;
            offsetX = (nodeWidth - targetWidth) / 2;
        }
        const widthMultiplier = mediaSize.width / targetWidth;
        const sourceX = 0;
        const sourceY = 0;
        const sourceWidth = cropX != null ? (cropX - offsetX) * widthMultiplier : mediaSize.width;
        const sourceHeight = mediaSize.height;
        const destX = (nodeWidth - targetWidth) / 2;
        const destY = y + (height - targetHeight) / 2;
        const destWidth = cropX != null ? cropX - offsetX : targetWidth;
        const destHeight = targetHeight;
        ctx.save();
        ctx.beginPath();
        let globalCompositeOperation = ctx.globalCompositeOperation;
        if (cropX) {
            ctx.rect(destX, destY, destWidth, destHeight);
            ctx.clip();
        }
        ctx.drawImage(image === null || image === void 0 ? void 0 : image.img, sourceX, sourceY, sourceWidth, sourceHeight, destX, destY, destWidth, destHeight);
        if (cropX != null && cropX >= (nodeWidth - targetWidth) / 2 && cropX <= targetWidth + offsetX) {
            ctx.beginPath();
            ctx.moveTo(cropX, destY);
            ctx.lineTo(cropX, destY + destHeight);
            ctx.globalCompositeOperation = "difference";
            ctx.strokeStyle = "rgba(255,255,255, 1)";
            ctx.stroke();
        }
        ctx.globalCompositeOperation = globalCompositeOperation;
        ctx.restore();
    }
    computeSize(width) {
        return [width, 20];
    }
    serializeValue(node, index) {
        const v = [];
        for (const data of this._value.images) {
            const d = { ...data };
            delete d.img;
            v.push(d);
        }
        return { images: v };
    }
}

class MxdImageComparer extends MxdBaseServerNode {
    constructor(title) {
        super(title ?? this.constructor?.nodeData?.name ?? NODE_TYPE_STRING);
        this.imageIndex = 0;
        this.imgs = [];
        this.serialize_widgets = true;
        this.isPointerDown = false;
        this.isPointerOver = false;
        this.pointerOverPos = [0, 0];
        this.canvasWidget = null;
        this._mxdInitialSize = null;
        this.properties = this.properties || {};
        this.properties["comparer_mode"] = "Slide";
    }
    onExecuted(output) {
        // super.onExecuted?.(output); // Simplified base doesn't have it
        if ("images" in output) {
            this.canvasWidget.value = {
                images: (output.images || []).map((d, i) => {
                    return {
                        name: i === 0 ? "A" : "B",
                        selected: true,
                        mediaType: "image",
                        url: mediaDataToUrl(d, "image"),
                    };
                }),
            };
        }
        else {
            output.a_images = output.a_images || [];
            output.b_images = output.b_images || [];
            output.a_videos = output.a_videos || [];
            output.b_videos = output.b_videos || [];
            const imagesToChoose = [];
            const total = output.a_images.length + output.b_images.length + output.a_videos.length + output.b_videos.length;
            const multiple = total > 2;
            for (const [i, d] of output.a_images.entries()) {
                imagesToChoose.push({
                    name: output.a_images.length > 1 || multiple ? `A${i + 1}` : "A",
                    selected: i === 0,
                    mediaType: "image",
                    url: mediaDataToUrl(d, "image"),
                });
            }
            for (const [i, d] of output.b_images.entries()) {
                imagesToChoose.push({
                    name: output.b_images.length > 1 || multiple ? `B${i + 1}` : "B",
                    selected: i === 0,
                    mediaType: "image",
                    url: mediaDataToUrl(d, "image"),
                });
            }
            for (const [i, d] of output.a_videos.entries()) {
                imagesToChoose.push({
                    name: output.a_videos.length > 1 || multiple ? `A${i + 1}` : "A",
                    selected: i === 0 && output.a_images.length === 0,
                    mediaType: "video",
                    url: mediaDataToUrl(d, "video"),
                });
            }
            for (const [i, d] of output.b_videos.entries()) {
                imagesToChoose.push({
                    name: output.b_videos.length > 1 || multiple ? `B${i + 1}` : "B",
                    selected: i === 0 && output.b_images.length === 0,
                    mediaType: "video",
                    url: mediaDataToUrl(d, "video"),
                });
            }
            this.canvasWidget.value = { images: imagesToChoose };
        }
    }
    onSerialize(serialised) {
        // super.onSerialize && super.onSerialize(serialised);
        for (let [index, widget_value] of (serialised.widgets_values || []).entries()) {
            if (this.widgets[index]?.name === "mxd_comparer") {
                // Do not persist preview image paths into workflow JSON.
                serialised.widgets_values[index] = [];
            }
        }
    }
    onNodeCreated() {
        this.canvasWidget = this.addCustomWidget(new MxdImageComparerWidget("mxd_comparer", this));
        this.setSize(this.computeSize());
        this.setDirtyCanvas(true, true);
        this._mxdInitialSize = [...this.size];
    }
    setIsPointerDown(down = this.isPointerDown) {
        const newIsDown = down && !!app.canvas.pointer_is_down;
        if (this.isPointerDown !== newIsDown) {
            this.isPointerDown = newIsDown;
            this.setDirtyCanvas(true, false);
        }
        this.imageIndex = this.isPointerDown ? 1 : 0;
        if (this.isPointerDown) {
            requestAnimationFrame(() => {
                this.setIsPointerDown();
            });
        }
    }
    onMouseDown(event, pos, canvas) {
        // super.onMouseDown?.(event, pos, canvas);
        this.setIsPointerDown(true);
        return false;
    }
    onMouseEnter(event) {
        // super.onMouseEnter?.(event);
        this.setIsPointerDown(!!app.canvas.pointer_is_down);
        this.isPointerOver = true;
    }
    onMouseLeave(event) {
        // super.onMouseLeave?.(event);
        this.setIsPointerDown(false);
        this.isPointerOver = false;
    }
    onMouseMove(event, pos, canvas) {
        // super.onMouseMove?.(event, pos, canvas);
        this.pointerOverPos = [...pos];
        this.imageIndex = this.pointerOverPos[0] > this.size[0] / 2 ? 1 : 0;
    }
    getHelp() {
        return `
      <p>
        The MXD comparer node overlays two media inputs (image or video) for quick A/B checks.
      </p>
      <ul>
        <li>
          <p>
            <strong>Notes</strong>
          </p>
          <ul>
            <li><p>
              In Slide mode, hover the node to reveal media B. In Click mode, press and hold to toggle A/B.
            </p></li>
          </ul>
        </li>
      </ul>`;
    }
    static setUp(comfyClass, nodeData) {
        let boundClass = NODE_OVERRIDE_CLASSES.get(nodeData.name);
        if (!boundClass) {
            // Each comparer node needs its own class so static nodeData does not get overwritten.
            boundClass = class extends MxdImageComparer {};
            NODE_OVERRIDE_CLASSES.set(nodeData.name, boundClass);
        }
        boundClass.nodeData = nodeData;
        MxdBaseServerNode.registerForOverride(comfyClass, nodeData, boundClass);
    }
    static onRegisteredForOverride(comfyClass) {
        addConnectionLayoutSupport(MxdImageComparer, app, [
            ["Left", "Right"],
            ["Right", "Left"],
        ]);
    }
}

// Define the static property for properties
MxdImageComparer["@comparer_mode"] = {
    type: "combo",
    values: ["Slide", "Click"],
};

// --- Extension Registration ---
app.registerExtension({
    name: "MXD.ImageComparer.Standalone",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (NODE_TYPE_STRINGS.has(nodeData.name)) {
            MxdImageComparer.setUp(nodeType, nodeData);
        }
    },
});
