var _a;

import { app } from "../../scripts/app.js";
import { MxdBaseServerNode } from "./mxd_base_node.js";
import { mxdRuntime } from "./mxd_runtime.js";
import { addConnectionLayoutSupport } from "./mxd_utils.js";
import {
  drawInfoIcon,
  drawNumberWidgetPart,
  drawRoundedRectangle,
  drawTogglePart,
  fitString,
  isLowQuality,
} from "./mxd_utils_canvas.js";
import {
  MxdBaseWidget,
  MxdBetterButtonWidget,
  MxdDividerWidget,
} from "./mxd_utils_widgets.js";
import { mxdApi } from "./mxd_api.js";
import { showLoraChooser } from "./mxd_utils_menu.js";
import { moveArrayItem, removeArrayItem } from "./mxd_shared_utils.js";
import { MxdLoraInfoDialog } from "./mxd_dialog_info.js";
import { LORA_INFO_SERVICE } from "./mxd_model_info_service.js";

const NODE_TYPE = "Lora Loader MXD";
const PROP_LABEL_SHOW_STRENGTHS = "Show Strengths";
const PROP_LABEL_SHOW_STRENGTHS_STATIC = `@${PROP_LABEL_SHOW_STRENGTHS}`;
const PROP_VALUE_SHOW_STRENGTHS_SINGLE = "Single Strength";
const PROP_VALUE_SHOW_STRENGTHS_SEPARATE = "Separate Model & Clip";

class MxdPowerLoraLoader extends MxdBaseServerNode {
  static title = NODE_TYPE;
  static type = NODE_TYPE;
  static comfyClass = NODE_TYPE;

  static [PROP_LABEL_SHOW_STRENGTHS_STATIC] = {
    type: "combo",
    values: [PROP_VALUE_SHOW_STRENGTHS_SINGLE, PROP_VALUE_SHOW_STRENGTHS_SEPARATE],
  };

  constructor(title = NODE_CLASS.title) {
    super(title);
    this.serialize_widgets = true;
    this.logger = mxdRuntime.newLogSession("[Power Lora Loader]");
    this.loraWidgetsCounter = 0;
    this.widgetButtonSpacer = null;
    this.properties[PROP_LABEL_SHOW_STRENGTHS] = PROP_VALUE_SHOW_STRENGTHS_SINGLE;

    mxdApi.getLoras();

    if (mxdRuntime.loadingApiJson) {
      const fullApiJson = mxdRuntime.loadingApiJson;
      setTimeout(() => {
        this.configureFromApiJson(fullApiJson);
      }, 16);
    }
  }

  configureFromApiJson(fullApiJson) {
    if (this.id == null) {
      const [n, v] = this.logger.errorParts("Cannot load from API JSON without node id.");
      console[n]?.(...v);
      return;
    }
    const nodeData =
      fullApiJson[this.id] || fullApiJson[String(this.id)] || fullApiJson[Number(this.id)];
    if (nodeData == null) {
      const [n, v] = this.logger.errorParts(`No node found in API JSON for node id ${this.id}.`);
      console[n]?.(...v);
      return;
    }
    this.configure({
      widgets_values: Object.values(nodeData.inputs).filter((input) => typeof input?.["lora"] === "string"),
    });
  }

  configure(info) {
    while (this.widgets?.length) this.removeWidget(0);
    this.widgetButtonSpacer = null;

    const hasSerializedNodeData =
      info?.id != null ||
      [
        "pos",
        "size",
        "flags",
        "mode",
        "order",
        "properties",
        "color",
        "bgcolor",
        "title",
        "inputs",
        "outputs",
        "type",
      ].some((key) => info?.[key] !== undefined);
    const serializedSize = Array.isArray(info?.size) ? [...info.size] : null;

    if (hasSerializedNodeData) {
      super.configure(info);
    }

    const baseWidth = this.size?.[0] ?? 0;
    const baseHeight = this.size?.[1] ?? 0;

    for (const widgetValue of info.widgets_values || []) {
      if (widgetValue?.lora !== undefined) {
        const widget = this.addNewLoraWidget();
        widget.value = { ...widgetValue };
      }
    }

    this.addNonLoraWidgets();

    this.size = this.size || [0, 0];
    if (serializedSize) {
      this.size[0] = serializedSize[0];
      this.size[1] = serializedSize[1];
    } else {
      const computed = this.computeSize();
      this.size[0] = Math.max(baseWidth, computed[0]);
      this.size[1] = Math.max(baseHeight, computed[1]);
    }

    this.setDirtyCanvas(true, true);
  }

  onNodeCreated() {
    super.onNodeCreated?.();
    if (!this.widgets?.length) {
      this.addNonLoraWidgets();
    }
    if (!this.configuring && !mxdRuntime.canvasCurrentlyPastingFromClipboard) {
      const computed = this.computeSize();
      this.size = this.size || [0, 0];
      this.size[0] = Math.max(this.size[0], computed[0]);
      this.size[1] = Math.max(this.size[1], computed[1]);
    }
    this.setDirtyCanvas(true, true);
  }

  addNewLoraWidget(lora) {
    this.loraWidgetsCounter++;
    const widget = this.addCustomWidget(new PowerLoraLoaderWidget("lora_" + this.loraWidgetsCounter));
    if (lora) widget.setLora(lora);
    if (this.widgetButtonSpacer) {
      moveArrayItem(this.widgets, widget, this.widgets.indexOf(this.widgetButtonSpacer));
    }
    return widget;
  }

  addNonLoraWidgets() {
    moveArrayItem(
      this.widgets,
      this.addCustomWidget(new MxdDividerWidget({ marginTop: 4, marginBottom: 0, thickness: 0 })),
      0,
    );
    moveArrayItem(this.widgets, this.addCustomWidget(new PowerLoraLoaderHeaderWidget()), 1);

    this.widgetButtonSpacer = this.addCustomWidget(
      new MxdDividerWidget({ marginTop: 4, marginBottom: 0, thickness: 0 }),
    );

    this.addCustomWidget(
      new MxdBetterButtonWidget("+ Add Lora", (event, pos, node) => {
        mxdApi.getLoras().then((lorasDetails) => {
          const loras = lorasDetails.map((l) => l.file);
          showLoraChooser(
            event,
            (value) => {
              if (typeof value === "string" && value !== "NONE") {
                this.addNewLoraWidget(value);
                const computed = this.computeSize();
                this.size[1] = Math.max(this.size?.[1] ?? 15, computed[1]);
                this.setDirtyCanvas(true, true);
              }
            },
            null,
            [...loras],
          );
        });
        return true;
      }),
    );
  }

  getSlotInPosition(canvasX, canvasY) {
    const slot = super.getSlotInPosition(canvasX, canvasY);
    if (!slot) {
      let lastWidget = null;
      for (const widget of this.widgets) {
        if (!widget.last_y) return;
        if (canvasY > this.pos[1] + widget.last_y) {
          lastWidget = widget;
          continue;
        }
        break;
      }
      if (lastWidget?.name?.startsWith("lora_")) {
        return { widget: lastWidget, output: { type: "LORA WIDGET" } };
      }
    }
    return slot;
  }

  getSlotMenuOptions(slot) {
    if (slot?.widget?.name?.startsWith("lora_")) {
      const widget = slot.widget;
      const index = this.widgets.indexOf(widget);
      const canMoveUp = !!this.widgets[index - 1]?.name?.startsWith("lora_");
      const canMoveDown = !!this.widgets[index + 1]?.name?.startsWith("lora_");
      const menuItems = [
        {
          content: `Show Info`,
          callback: () => widget.showLoraInfoDialog(),
        },
        null,
        {
          content: `${widget.value.on ? "Disable" : "Enable"}`,
          callback: () => {
            widget.value.on = !widget.value.on;
          },
        },
        {
          content: `Move Up`,
          disabled: !canMoveUp,
          callback: () => {
            moveArrayItem(this.widgets, widget, index - 1);
          },
        },
        {
          content: `Move Down`,
          disabled: !canMoveDown,
          callback: () => {
            moveArrayItem(this.widgets, widget, index + 1);
          },
        },
        {
          content: `Remove`,
          callback: () => {
            removeArrayItem(this.widgets, widget);
          },
        },
      ];
      new LiteGraph.ContextMenu(menuItems, {
        title: "LORA WIDGET",
        event: mxdRuntime.lastCanvasMouseEvent,
      });
      return undefined;
    }
    return this.defaultGetSlotMenuOptions(slot);
  }

  refreshComboInNode(defs) {
    mxdApi.getLoras(true);
  }

  hasLoraWidgets() {
    return !!this.widgets?.find((w) => w.name?.startsWith("lora_"));
  }

  allLorasState() {
    let allOn = true;
    let allOff = true;
    for (const widget of this.widgets) {
      if (widget.name?.startsWith("lora_")) {
        const on = widget.value?.on;
        allOn = allOn && on === true;
        allOff = allOff && on === false;
        if (!allOn && !allOff) return null;
      }
    }
    return allOn && this.widgets?.length ? true : false;
  }

  toggleAllLoras() {
    const allOn = this.allLorasState();
    const toggledTo = !allOn;
    for (const widget of this.widgets) {
      if (widget.name?.startsWith("lora_") && widget.value?.on != null) {
        widget.value.on = toggledTo;
      }
    }
  }

  static setUp(comfyClass, nodeData) {
    MxdBaseServerNode.registerForOverride(comfyClass, nodeData, NODE_CLASS);
  }

  static onRegisteredForOverride(comfyClass, ctxClass) {
    addConnectionLayoutSupport(NODE_CLASS, app, [
      ["Left", "Right"],
      ["Right", "Left"],
    ]);
    setTimeout(() => {
      NODE_CLASS.category = comfyClass.category;
    });
  }
}
_a = PROP_LABEL_SHOW_STRENGTHS_STATIC;
MxdPowerLoraLoader[_a] = {
  type: "combo",
  values: [PROP_VALUE_SHOW_STRENGTHS_SINGLE, PROP_VALUE_SHOW_STRENGTHS_SEPARATE],
};

class PowerLoraLoaderHeaderWidget extends MxdBaseWidget {
  constructor(name = "PowerLoraLoaderHeaderWidget") {
    super(name);
    this.value = { type: "PowerLoraLoaderHeaderWidget" };
    this.type = "custom";
    this.hitAreas = {
      toggle: { bounds: [0, 0], onDown: this.onToggleDown },
    };
    this.showModelAndClip = null;
  }

  draw(ctx, node, w, posY, height) {
    if (!node.hasLoraWidgets()) return;

    this.showModelAndClip = node.properties[PROP_LABEL_SHOW_STRENGTHS] === PROP_VALUE_SHOW_STRENGTHS_SEPARATE;
    const margin = 10;
    const innerMargin = margin * 0.33;
    const lowQuality = isLowQuality();
    const allLoraState = node.allLorasState();

    posY += 2;
    const midY = posY + height * 0.5;
    let posX = 10;
    ctx.save();
    this.hitAreas.toggle.bounds = drawTogglePart(ctx, { posX, posY, height, value: allLoraState });

    if (!lowQuality) {
      posX += this.hitAreas.toggle.bounds[1] + innerMargin;
      ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
      ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText("Toggle All", posX, midY);

      let rposX = node.size[0] - margin - innerMargin - innerMargin;
      ctx.textAlign = "center";
      ctx.fillText(this.showModelAndClip ? "Clip" : "Strength", rposX - drawNumberWidgetPart.WIDTH_TOTAL / 2, midY);
      if (this.showModelAndClip) {
        rposX = rposX - drawNumberWidgetPart.WIDTH_TOTAL - innerMargin * 2;
        ctx.fillText("Model", rposX - drawNumberWidgetPart.WIDTH_TOTAL / 2, midY);
      }
    }
    ctx.restore();
  }

  onToggleDown(event, pos, node) {
    node.toggleAllLoras();
    this.cancelMouseDown();
    return true;
  }
}

const DEFAULT_LORA_WIDGET_DATA = {
  on: true,
  lora: null,
  strength: 1,
  strengthTwo: null,
};

class PowerLoraLoaderWidget extends MxdBaseWidget {
  constructor(name) {
    super(name);
    this.type = "custom";
    this.haveMouseMovedStrength = false;
    this.loraInfoPromise = null;
    this.loraInfo = null;
    this.showModelAndClip = null;
    this.hitAreas = {
      toggle: { bounds: [0, 0], onDown: this.onToggleDown },
      lora: { bounds: [0, 0], onClick: this.onLoraClick },
      info: { bounds: [0, 0], onDown: this.onInfoDown },
      remove: { bounds: [0, 0], onDown: this.onRemoveDown },
      strengthDec: { bounds: [0, 0], onClick: this.onStrengthDecDown },
      strengthVal: { bounds: [0, 0], onClick: this.onStrengthValUp },
      strengthInc: { bounds: [0, 0], onClick: this.onStrengthIncDown },
      strengthAny: { bounds: [0, 0], onMove: this.onStrengthAnyMove },
      strengthTwoDec: { bounds: [0, 0], onClick: this.onStrengthTwoDecDown },
      strengthTwoVal: { bounds: [0, 0], onClick: this.onStrengthTwoValUp },
      strengthTwoInc: { bounds: [0, 0], onClick: this.onStrengthTwoIncDown },
      strengthTwoAny: { bounds: [0, 0], onMove: this.onStrengthTwoAnyMove },
    };
    this._value = {
      on: true,
      lora: null,
      strength: 1,
      strengthTwo: null,
    };
  }

  set value(v) {
    this._value = v;
    if (typeof this._value !== "object") {
      this._value = { ...DEFAULT_LORA_WIDGET_DATA };
      if (this.showModelAndClip) {
        this._value.strengthTwo = this._value.strength;
      }
    }
    this.getLoraInfo();
  }

  get value() {
    return this._value;
  }

  setLora(lora) {
    this._value.lora = lora;
    this.getLoraInfo();
  }

  draw(ctx, node, w, posY, height) {
    let currentShowModelAndClip = node.properties[PROP_LABEL_SHOW_STRENGTHS] === PROP_VALUE_SHOW_STRENGTHS_SEPARATE;
    if (this.showModelAndClip !== currentShowModelAndClip) {
      let oldShowModelAndClip = this.showModelAndClip;
      this.showModelAndClip = currentShowModelAndClip;
      if (this.showModelAndClip) {
        if (oldShowModelAndClip != null) {
          this.value.strengthTwo = this.value.strength ?? 1;
        }
      } else {
        this.value.strengthTwo = null;
        this.hitAreas.strengthTwoDec.bounds = [0, -1];
        this.hitAreas.strengthTwoVal.bounds = [0, -1];
        this.hitAreas.strengthTwoInc.bounds = [0, -1];
        this.hitAreas.strengthTwoAny.bounds = [0, -1];
      }
    }

    ctx.save();
    const margin = 10;
    const innerMargin = margin * 0.33;
    const lowQuality = isLowQuality();
    const midY = posY + height * 0.5;

    let posX = margin;

    drawRoundedRectangle(ctx, { pos: [posX, posY], size: [node.size[0] - margin * 2, height] });
    this.hitAreas.toggle.bounds = drawTogglePart(ctx, { posX, posY, height, value: this.value.on });
    posX += this.hitAreas.toggle.bounds[1] + innerMargin;

    if (lowQuality) {
      ctx.restore();
      return;
    }

    if (!this.value.on) {
      ctx.globalAlpha = app.canvas.editor_alpha * 0.4;
    }

    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;

    let rposX = node.size[0] - margin - innerMargin - innerMargin;

    const strengthValue = this.showModelAndClip ? (this.value.strengthTwo ?? 1) : (this.value.strength ?? 1);

    let textColor = undefined;
    if (this.loraInfo?.strengthMax != null && strengthValue > this.loraInfo?.strengthMax) {
      textColor = "#c66";
    } else if (this.loraInfo?.strengthMin != null && strengthValue < this.loraInfo?.strengthMin) {
      textColor = "#c66";
    }

    const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, {
      posX: node.size[0] - margin - innerMargin - innerMargin,
      posY,
      height,
      value: strengthValue,
      direction: -1,
      textColor,
    });

    this.hitAreas.strengthDec.bounds = leftArrow;
    this.hitAreas.strengthVal.bounds = text;
    this.hitAreas.strengthInc.bounds = rightArrow;
    this.hitAreas.strengthAny.bounds = [leftArrow[0], rightArrow[0] + rightArrow[1] - leftArrow[0]];

    rposX = leftArrow[0] - innerMargin;

    if (this.showModelAndClip) {
      rposX -= innerMargin;
      this.hitAreas.strengthTwoDec.bounds = this.hitAreas.strengthDec.bounds;
      this.hitAreas.strengthTwoVal.bounds = this.hitAreas.strengthVal.bounds;
      this.hitAreas.strengthTwoInc.bounds = this.hitAreas.strengthInc.bounds;
      this.hitAreas.strengthTwoAny.bounds = this.hitAreas.strengthAny.bounds;

      let textColor = undefined;
      if (this.loraInfo?.strengthMax != null && this.value.strength > this.loraInfo?.strengthMax) {
        textColor = "#c66";
      } else if (this.loraInfo?.strengthMin != null && this.value.strength < this.loraInfo?.strengthMin) {
        textColor = "#c66";
      }
      const [leftArrow2, text2, rightArrow2] = drawNumberWidgetPart(ctx, {
        posX: rposX,
        posY,
        height,
        value: this.value.strength ?? 1,
        direction: -1,
        textColor,
      });
      this.hitAreas.strengthDec.bounds = leftArrow2;
      this.hitAreas.strengthVal.bounds = text2;
      this.hitAreas.strengthInc.bounds = rightArrow2;
      this.hitAreas.strengthAny.bounds = [leftArrow2[0], rightArrow2[0] + rightArrow2[1] - leftArrow2[0]];
      rposX = leftArrow2[0] - innerMargin;
    }

    const showInfoIcon = this.value?.lora && this.value?.lora !== "None";
    const infoIconSize = height * 0.66;
    const infoWidth = infoIconSize + innerMargin + innerMargin;
    if (showInfoIcon) {
      rposX -= innerMargin;
      drawInfoIcon(ctx, rposX - infoIconSize, posY + (height - infoIconSize) / 2, infoIconSize);
      this.hitAreas.info.bounds = [rposX - infoIconSize, infoWidth];
      rposX = rposX - infoIconSize - innerMargin;
    } else {
      this.hitAreas.info.bounds = [0, -1];
    }

    const actionIconSize = infoIconSize;
    const actionWidth = actionIconSize + innerMargin;
    const drawAction = (key, label, color = LiteGraph.WIDGET_TEXT_COLOR) => {
      rposX -= actionWidth;
      const x = rposX;
      const y = posY + (height - actionIconSize) / 2;
      drawRoundedRectangle(ctx, {
        pos: [x, y],
        size: [actionIconSize, actionIconSize],
        borderRadius: actionIconSize * 0.15,
        colorBackground: "rgba(0,0,0,0.22)",
        colorStroke: "rgba(255,255,255,0.12)",
      });
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = color;
      ctx.fillText(label, x + actionIconSize * 0.5, y + actionIconSize * 0.52);
      this.hitAreas[key].bounds = [x, actionIconSize];
      rposX -= innerMargin * 0.25;
    };

    drawAction("remove", "x", "#d88");

    const loraWidth = rposX - posX;
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const loraLabel = String(this.value?.lora || "None");
    ctx.fillText(fitString(ctx, loraLabel, loraWidth), posX, midY);

    this.hitAreas.lora.bounds = [posX, loraWidth];
    posX += loraWidth + innerMargin;

    ctx.globalAlpha = app.canvas.editor_alpha;
    ctx.restore();
  }

  serializeValue(node, index) {
    const v = { ...this.value };
    if (!this.showModelAndClip) {
      delete v.strengthTwo;
    } else {
      this.value.strengthTwo = this.value.strengthTwo ?? 1;
      v.strengthTwo = this.value.strengthTwo;
    }
    return v;
  }

  onToggleDown(event, pos, node) {
    this.value.on = !this.value.on;
    this.cancelMouseDown();
    return true;
  }

  onInfoDown(event, pos, node) {
    this.showLoraInfoDialog();
    this.cancelMouseDown();
    return true;
  }

  onRemoveDown(event, pos, node) {
    removeArrayItem(node.widgets, this);
    node.setDirtyCanvas(true, true);
    this.cancelMouseDown();
    return true;
  }

  onLoraClick(event, pos, node) {
    showLoraChooser(event, (value) => {
      if (typeof value === "string") {
        this.value.lora = value;
        this.loraInfo = null;
        this.getLoraInfo();
      }
      node.setDirtyCanvas(true, true);
    });
    this.cancelMouseDown();
  }

  onStrengthDecDown(event, pos, node) {
    this.stepStrength(-1, false);
  }

  onStrengthIncDown(event, pos, node) {
    this.stepStrength(1, false);
  }

  onStrengthTwoDecDown(event, pos, node) {
    this.stepStrength(-1, true);
  }

  onStrengthTwoIncDown(event, pos, node) {
    this.stepStrength(1, true);
  }

  onStrengthAnyMove(event, pos, node) {
    this.doOnStrengthAnyMove(event, false);
  }

  onStrengthTwoAnyMove(event, pos, node) {
    this.doOnStrengthAnyMove(event, true);
  }

  doOnStrengthAnyMove(event, isTwo = false) {
    if (event.deltaX) {
      let prop = isTwo ? "strengthTwo" : "strength";
      this.haveMouseMovedStrength = true;
      this.value[prop] = (this.value[prop] ?? 1) + event.deltaX * 0.05;
    }
  }

  onStrengthValUp(event, pos, node) {
    this.doOnStrengthValUp(event, false);
  }

  onStrengthTwoValUp(event, pos, node) {
    this.doOnStrengthValUp(event, true);
  }

  doOnStrengthValUp(event, isTwo = false) {
    if (this.haveMouseMovedStrength) return;
    let prop = isTwo ? "strengthTwo" : "strength";
    const canvas = app.canvas;
    canvas.prompt("Value", this.value[prop], (v) => (this.value[prop] = Number(v)), event);
  }

  onMouseUp(event, pos, node) {
    super.onMouseUp(event, pos, node);
    this.haveMouseMovedStrength = false;
  }

  showLoraInfoDialog() {
    if (!this.value.lora || this.value.lora === "None") {
      return;
    }
    const infoDialog = new MxdLoraInfoDialog(this.value.lora).show();
    infoDialog.addEventListener("close", (e) => {
      if (e.detail.dirty) {
        this.getLoraInfo(true);
      }
    });
  }

  stepStrength(direction, isTwo = false) {
    let step = 0.05;
    let prop = isTwo ? "strengthTwo" : "strength";
    let strength = (this.value[prop] ?? 1) + step * direction;
    this.value[prop] = Math.round(strength * 100) / 100;
  }

  getLoraInfo(force = false) {
    if (!this.loraInfoPromise || force == true) {
      let promise;
      if (this.value.lora && this.value.lora != "None") {
        promise = LORA_INFO_SERVICE.getInfo(this.value.lora, force, true);
      } else {
        promise = Promise.resolve(null);
      }
      this.loraInfoPromise = promise.then((v) => (this.loraInfo = v));
    }
    return this.loraInfoPromise;
  }
}

const NODE_CLASS = MxdPowerLoraLoader;

app.registerExtension({
  name: "mxd.PowerLoraLoader",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === NODE_CLASS.type) {
      NODE_CLASS.setUp(nodeType, nodeData);
    }
  },
});
