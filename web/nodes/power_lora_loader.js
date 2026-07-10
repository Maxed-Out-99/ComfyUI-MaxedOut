// "Lora Loader MXD" — multi-lora stack with optional separate model/clip
// strengths (the "Show Strengths" node property). Shared machinery lives in
// power_lora_base.js; this file only adds the dual-strength row behavior.
import { app } from "../../../scripts/app.js";
import { drawNumberWidgetPart, isLowQuality } from "../lib/mxd_utils_canvas.js";
import {
  MxdPowerLoraLoaderBase,
  PowerLoraBaseWidget,
  PROP_LABEL_SHOW_STRENGTHS,
  PROP_LABEL_SHOW_STRENGTHS_STATIC,
  PROP_VALUE_SHOW_STRENGTHS_SINGLE,
  PROP_VALUE_SHOW_STRENGTHS_SEPARATE,
} from "./power_lora_base.js";

const NODE_TYPE = "Lora Loader MXD";

class MxdPowerLoraLoader extends MxdPowerLoraLoaderBase {
  static title = NODE_TYPE;
  static type = NODE_TYPE;
  static comfyClass = NODE_TYPE;

  static [PROP_LABEL_SHOW_STRENGTHS_STATIC] = {
    type: "combo",
    values: [PROP_VALUE_SHOW_STRENGTHS_SINGLE, PROP_VALUE_SHOW_STRENGTHS_SEPARATE],
  };

  constructor(title = NODE_CLASS.title) {
    super(title, "[Power Lora Loader]");
    this.properties[PROP_LABEL_SHOW_STRENGTHS] = PROP_VALUE_SHOW_STRENGTHS_SINGLE;
  }
}

class PowerLoraLoaderWidget extends PowerLoraBaseWidget {
  constructor(name) {
    super(name);
    this.showModelAndClip = null;
    Object.assign(this.hitAreas, {
      strengthTwoDec: { bounds: [0, 0], onClick: this.onStrengthTwoDecDown },
      strengthTwoVal: { bounds: [0, 0], onClick: this.onStrengthTwoValUp },
      strengthTwoInc: { bounds: [0, 0], onClick: this.onStrengthTwoIncDown },
      strengthTwoAny: { bounds: [0, 0], onMove: this.onStrengthTwoAnyMove },
    });
    this._value = { ...this.newDefaultValue() };
  }

  newDefaultValue() {
    return { on: true, lora: null, strength: 1, strengthTwo: null };
  }

  set value(v) {
    this._value = v;
    if (typeof this._value !== "object") {
      this._value = { ...this.newDefaultValue() };
      if (this.showModelAndClip) {
        this._value.strengthTwo = this._value.strength;
      }
    }
    this.getLoraInfo();
  }

  get value() {
    return this._value;
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

    const posX = this.drawRowBackgroundAndToggle(ctx, node, posY, height, margin, innerMargin);

    if (lowQuality) {
      ctx.restore();
      return;
    }

    if (!this.value.on) {
      ctx.globalAlpha = app.canvas.editor_alpha * 0.4;
    }

    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;

    // Rightmost number: clip strength when split, otherwise the single strength.
    const strengthValue = this.showModelAndClip ? (this.value.strengthTwo ?? 1) : (this.value.strength ?? 1);

    const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, {
      posX: node.size[0] - margin - innerMargin - innerMargin,
      posY,
      height,
      value: strengthValue,
      direction: -1,
      textColor: this.strengthTextColor(strengthValue),
    });

    this.hitAreas.strengthDec.bounds = leftArrow;
    this.hitAreas.strengthVal.bounds = text;
    this.hitAreas.strengthInc.bounds = rightArrow;
    this.hitAreas.strengthAny.bounds = [leftArrow[0], rightArrow[0] + rightArrow[1] - leftArrow[0]];

    let rposX = leftArrow[0] - innerMargin;

    if (this.showModelAndClip) {
      rposX -= innerMargin;
      // The rightmost number becomes clip (strengthTwo); the model strength
      // gets its own number to the left.
      this.hitAreas.strengthTwoDec.bounds = this.hitAreas.strengthDec.bounds;
      this.hitAreas.strengthTwoVal.bounds = this.hitAreas.strengthVal.bounds;
      this.hitAreas.strengthTwoInc.bounds = this.hitAreas.strengthInc.bounds;
      this.hitAreas.strengthTwoAny.bounds = this.hitAreas.strengthAny.bounds;

      const [leftArrow2, text2, rightArrow2] = drawNumberWidgetPart(ctx, {
        posX: rposX,
        posY,
        height,
        value: this.value.strength ?? 1,
        direction: -1,
        textColor: this.strengthTextColor(this.value.strength),
      });
      this.hitAreas.strengthDec.bounds = leftArrow2;
      this.hitAreas.strengthVal.bounds = text2;
      this.hitAreas.strengthInc.bounds = rightArrow2;
      this.hitAreas.strengthAny.bounds = [leftArrow2[0], rightArrow2[0] + rightArrow2[1] - leftArrow2[0]];
      rposX = leftArrow2[0] - innerMargin;
    }

    this.drawIconsAndName(ctx, node, posX, posY, height, rposX, innerMargin, midY);

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

  stepStrength(direction, isTwo = false) {
    let step = 0.05;
    let prop = isTwo ? "strengthTwo" : "strength";
    let strength = (this.value[prop] ?? 1) + step * direction;
    this.value[prop] = Math.round(strength * 100) / 100;
  }
}

MxdPowerLoraLoader.loraWidgetClass = PowerLoraLoaderWidget;

const NODE_CLASS = MxdPowerLoraLoader;

app.registerExtension({
  name: "mxd.PowerLoraLoader",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === NODE_CLASS.type) {
      NODE_CLASS.setUp(nodeType, nodeData);
    }
  },
});
