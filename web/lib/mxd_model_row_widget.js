import { app } from "../../../scripts/app.js";
import { MxdBaseWidget } from "./mxd_utils_widgets.js";
import { drawRoundedRectangle, drawInfoIcon, fitString, isLowQuality } from "./mxd_utils_canvas.js";
import { showModelChooser } from "./mxd_utils_menu.js";

/** A single-row "pick a model + info icon" widget, giving any plain combo-backed loader
 * node (Smart Model/CLIP/Checkpoint loaders) the same look/behavior as a Lora Loader MXD
 * row: click the name to open a tree/search chooser, click the "i" icon for the info dialog. */
export class MxdModelRowWidget extends MxdBaseWidget {
  constructor(name, config) {
    super(name);
    this.type = "custom";
    this.config = config;
    this._value = config.initialValue ?? "None";
    this.hitAreas = {
      arrowLeft: { bounds: [0, 0], onClick: this.onArrowLeftClick },
      value: { bounds: [0, 0], onClick: this.onValueClick },
      info: { bounds: [0, 0], onDown: this.onInfoDown },
      arrowRight: { bounds: [0, 0], onClick: this.onArrowRightClick },
    };
  }

  set value(v) {
    this._value = v;
    this.config.onChange?.(v);
  }

  get value() {
    return this._value;
  }

  draw(ctx, node, w, posY, height) {
    ctx.save();
    const margin = 10;
    const innerMargin = margin * 0.33;
    const lowQuality = isLowQuality();
    const midY = posY + height * 0.5;
    const posX = margin;
    const width = node.size[0] - margin * 2;

    drawRoundedRectangle(ctx, { pos: [posX, posY], size: [width, height] });

    if (lowQuality) {
      this.hitAreas.arrowLeft.bounds = [0, -1];
      this.hitAreas.arrowRight.bounds = [0, -1];
      this.hitAreas.info.bounds = [0, -1];
      ctx.restore();
      return;
    }

    const arrowWidth = height * 0.32;
    const arrowHeight = height * 0.4;
    const arrowMargin = innerMargin * 1.5;

    // Left arrow (previous), mirroring the stock combo widget's layout.
    ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || LiteGraph.WIDGET_TEXT_COLOR || "#b8b8b8";
    const leftArrowX = posX + arrowMargin;
    ctx.fill(
      new Path2D(
        `M ${leftArrowX + arrowWidth} ${midY - arrowHeight / 2} L ${leftArrowX} ${midY} L ${leftArrowX + arrowWidth} ${midY + arrowHeight / 2} z`,
      ),
    );
    this.hitAreas.arrowLeft.bounds = [posX, arrowWidth + arrowMargin * 2];
    let leftEdge = posX + arrowWidth + arrowMargin * 2;

    // Right arrow (next), pinned to the right edge.
    const rightArrowX = posX + width - arrowMargin - arrowWidth;
    ctx.fill(
      new Path2D(
        `M ${rightArrowX} ${midY - arrowHeight / 2} L ${rightArrowX + arrowWidth} ${midY} L ${rightArrowX} ${midY + arrowHeight / 2} z`,
      ),
    );
    this.hitAreas.arrowRight.bounds = [rightArrowX - arrowMargin, arrowWidth + arrowMargin * 2];
    let rightEdge = rightArrowX - arrowMargin;

    // Info icon sits just inside (to the left of) the right arrow.
    const showInfoIcon = this._value && this._value !== "None";
    const infoIconSize = height * 0.66;
    if (showInfoIcon) {
      rightEdge -= innerMargin;
      drawInfoIcon(ctx, rightEdge - infoIconSize, posY + (height - infoIconSize) / 2, infoIconSize);
      this.hitAreas.info.bounds = [rightEdge - infoIconSize, infoIconSize];
      rightEdge = rightEdge - infoIconSize - innerMargin;
    } else {
      this.hitAreas.info.bounds = [0, -1];
    }

    // Label (the input name, e.g. "ckpt_name") on the left, value on the right —
    // matching the stock combo widget's layout.
    const valueWidth = rightEdge - leftEdge;
    ctx.textBaseline = "middle";
    ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || LiteGraph.WIDGET_TEXT_COLOR || "#b8b8b8";
    ctx.textAlign = "left";
    ctx.fillText(this.name, leftEdge, midY);

    const labelWidth = ctx.measureText(this.name).width;
    const valueX = leftEdge + labelWidth + innerMargin * 2;
    const valueMaxWidth = Math.max(0, rightEdge - valueX);
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "right";
    const valueLabel = String(this._value || "None");
    ctx.fillText(fitString(ctx, valueLabel, valueMaxWidth), rightEdge, midY);
    this.hitAreas.value.bounds = [leftEdge, valueWidth];

    ctx.restore();
  }

  async stepValue(delta) {
    const files = await this.config.getFileList();
    if (!files || !files.length) return;
    const currentIndex = files.indexOf(this._value);
    const nextIndex =
      currentIndex === -1 ? (delta > 0 ? 0 : files.length - 1) : (currentIndex + delta + files.length) % files.length;
    this.value = files[nextIndex];
  }

  onArrowLeftClick(event, pos, node) {
    this.stepValue(-1).then(() => node.setDirtyCanvas(true, true));
    this.cancelMouseDown();
    return true;
  }

  onArrowRightClick(event, pos, node) {
    this.stepValue(1).then(() => node.setDirtyCanvas(true, true));
    this.cancelMouseDown();
    return true;
  }

  async onValueClick(event, pos, node) {
    const files = await this.config.getFileList();
    showModelChooser(
      event,
      (value) => {
        if (typeof value === "string" && value) {
          this.value = value;
          node.setDirtyCanvas(true, true);
        }
      },
      null,
      files,
      this.config.chooserTitle,
    );
    this.cancelMouseDown();
  }

  onInfoDown(event, pos, node) {
    this.showInfoDialog();
    this.cancelMouseDown();
    return true;
  }

  showInfoDialog() {
    if (!this._value || this._value === "None" || !this.config.dialogClass) {
      return;
    }
    new this.config.dialogClass(this._value).show();
  }
}

/** Swaps a node's default combo widget (by input name) for an `MxdModelRowWidget` in place,
 * keeping the same name/position so prompt serialization is unaffected. */
export function replaceWidgetWithModelRow(node, widgetName, config) {
  const index = node.widgets?.findIndex((w) => w.name === widgetName) ?? -1;
  if (index === -1) return null;

  const oldWidget = node.widgets[index];
  const rowWidget = new MxdModelRowWidget(widgetName, {
    ...config,
    initialValue: oldWidget.value,
  });
  rowWidget.last_y = oldWidget.last_y ?? 0;
  node.widgets[index] = rowWidget;
  return rowWidget;
}
