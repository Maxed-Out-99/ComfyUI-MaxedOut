// "LTX2 Lora Loader MXD" — the Power Lora Loader plus per-layer strength rows
// (video / video→audio / audio / audio→video / other) for LTX2's split
// attention blocks. Shared machinery lives in power_lora_base.js; the lora
// rows themselves are the stock single-strength PowerLoraBaseWidget.
import { app } from "../../../scripts/app.js";
import {
  drawNumberWidgetPart,
  drawRoundedRectangle,
  isLowQuality,
} from "../lib/mxd_utils_canvas.js";
import { MxdBaseWidget, MxdDividerWidget, MxdLabelWidget } from "../lib/mxd_utils_widgets.js";
import { moveArrayItem } from "../lib/mxd_shared_utils.js";
import {
  MxdPowerLoraLoaderBase,
  PowerLoraBaseWidget,
  PowerLoraLoaderHeaderWidget,
} from "./power_lora_base.js";

const NODE_TYPE = "LTX2 Lora Loader MXD";

const LTX2_STRENGTH_ROWS = [
  { key: "video", label: "Video" },
  { key: "video_to_audio", label: "Video → Audio" },
  { key: "audio", label: "Audio" },
  { key: "audio_to_video", label: "Audio → Video" },
  { key: "other", label: "Other" },
];

class MxdLtx2PowerLoraLoader extends MxdPowerLoraLoaderBase {
  static title = NODE_TYPE;
  static type = NODE_TYPE;
  static comfyClass = NODE_TYPE;

  constructor(title = NODE_CLASS.title) {
    super(title, "[LTX2 Power Lora Loader]");
    this.advancedWidgets = {};
  }

  apiJsonInputFilter(input) {
    return typeof input?.["lora"] === "string" || input?.["type"] === "Ltx2StrengthWidget";
  }

  collectNonLoraWidgetValue(widgetValue) {
    if (widgetValue?.type === "Ltx2StrengthWidget" && widgetValue?.key) {
      this._pendingNonLoraValues[widgetValue.key] = widgetValue.value;
    }
  }

  applyNonLoraWidgetValues() {
    for (const [key, value] of Object.entries(this._pendingNonLoraValues)) {
      if (this.advancedWidgets[key]) {
        this.advancedWidgets[key].value = { type: "Ltx2StrengthWidget", key, value };
      }
    }
  }

  addHeaderWidgets() {
    let idx = 0;
    moveArrayItem(
      this.widgets,
      this.addCustomWidget(new MxdDividerWidget({ marginTop: 4, marginBottom: 0, thickness: 0 })),
      idx++,
    );
    moveArrayItem(
      this.widgets,
      this.addCustomWidget(new MxdLabelWidget("LTX2 Layer Strengths", { size: 11 })),
      idx++,
    );

    this.advancedWidgets = {};
    for (const { key, label } of LTX2_STRENGTH_ROWS) {
      const widget = this.addCustomWidget(new Ltx2StrengthWidget(key, label));
      this.advancedWidgets[key] = widget;
      moveArrayItem(this.widgets, widget, idx++);
    }

    moveArrayItem(
      this.widgets,
      this.addCustomWidget(new MxdDividerWidget({ marginTop: 4, marginBottom: 4, thickness: 1 })),
      idx++,
    );
    moveArrayItem(
      this.widgets,
      this.addCustomWidget(new MxdDividerWidget({ marginTop: 4, marginBottom: 0, thickness: 0 })),
      idx++,
    );
    moveArrayItem(this.widgets, this.addCustomWidget(new PowerLoraLoaderHeaderWidget()), idx++);
  }
}

class Ltx2StrengthWidget extends MxdBaseWidget {
  constructor(key, label) {
    super("ltx2_" + key);
    this.type = "custom";
    this.key = key;
    this.label = label;
    this.haveMouseMovedStrength = false;
    this.hitAreas = {
      strengthDec: { bounds: [0, 0], onClick: this.onStrengthDecDown },
      strengthVal: { bounds: [0, 0], onClick: this.onStrengthValUp },
      strengthInc: { bounds: [0, 0], onClick: this.onStrengthIncDown },
      strengthAny: { bounds: [0, 0], onMove: this.onStrengthAnyMove },
    };
    this._value = { type: "Ltx2StrengthWidget", key, value: 1 };
  }

  set value(v) {
    this._value = v;
    if (typeof this._value !== "object") {
      this._value = { type: "Ltx2StrengthWidget", key: this.key, value: 1 };
    }
  }

  get value() {
    return this._value;
  }

  get numValue() {
    return this._value?.value ?? 1;
  }

  set numValue(v) {
    this._value.value = v;
  }

  draw(ctx, node, w, posY, height) {
    ctx.save();
    const margin = 10;
    const innerMargin = margin * 0.33;
    const lowQuality = isLowQuality();
    const midY = posY + height * 0.5;
    const posX = margin;

    drawRoundedRectangle(ctx, { pos: [posX, posY], size: [node.size[0] - margin * 2, height] });

    if (lowQuality) {
      ctx.restore();
      return;
    }

    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(this.label, posX + innerMargin * 2, midY);

    const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, {
      posX: node.size[0] - margin - innerMargin - innerMargin,
      posY,
      height,
      value: this.numValue,
      direction: -1,
    });
    this.hitAreas.strengthDec.bounds = leftArrow;
    this.hitAreas.strengthVal.bounds = text;
    this.hitAreas.strengthInc.bounds = rightArrow;
    this.hitAreas.strengthAny.bounds = [leftArrow[0], rightArrow[0] + rightArrow[1] - leftArrow[0]];

    ctx.restore();
  }

  serializeValue(node, index) {
    return { ...this.value };
  }

  onStrengthDecDown() {
    this.stepStrength(-1);
  }

  onStrengthIncDown() {
    this.stepStrength(1);
  }

  onStrengthAnyMove(event) {
    if (event.deltaX) {
      this.haveMouseMovedStrength = true;
      this.numValue = (this.numValue ?? 1) + event.deltaX * 0.05;
    }
  }

  onStrengthValUp(event) {
    if (this.haveMouseMovedStrength) return;
    const canvas = app.canvas;
    canvas.prompt("Value", this.numValue, (v) => (this.numValue = Number(v)), event);
  }

  onMouseUp(event, pos, node) {
    super.onMouseUp(event, pos, node);
    this.haveMouseMovedStrength = false;
  }

  stepStrength(direction) {
    let step = 0.05;
    let strength = (this.numValue ?? 1) + step * direction;
    this.numValue = Math.round(strength * 100) / 100;
  }
}

MxdLtx2PowerLoraLoader.loraWidgetClass = PowerLoraBaseWidget;

const NODE_CLASS = MxdLtx2PowerLoraLoader;

app.registerExtension({
  name: "mxd.Ltx2PowerLoraLoader",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === NODE_CLASS.type) {
      NODE_CLASS.setUp(nodeType, nodeData);
    }
  },
});
