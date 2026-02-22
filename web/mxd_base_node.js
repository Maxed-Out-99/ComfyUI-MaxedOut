import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { defineProperty, moveArrayItem } from "./mxd_shared_utils.js";

export class MxdBaseNode extends LGraphNode {
  static title = "__NEED_CLASS_TITLE__";
  static type = "__NEED_CLASS_TYPE__";
  static category = "mxd";
  static _category = "mxd";
  static exposedActions = [];

  constructor(title = MxdBaseNode.title) {
    super(title);
    this.comfyClass = "__NEED_COMFY_CLASS__";
    this.isVirtualNode = false;
    this.isDropEnabled = false;
    this.removed = false;
    this.configuring = false;
    this._tempWidth = 0;
    this.__constructed__ = false;
    this.widgets = this.widgets || [];
    this.properties = this.properties || {};

    defineProperty(this, "mode", {
      get: () => this._mxd_mode,
      set: (mode) => {
        if (this._mxd_mode !== mode) {
          const oldMode = this._mxd_mode;
          this._mxd_mode = mode;
          this.onModeChange(oldMode, mode);
        }
      },
    });

    setTimeout(() => this.checkAndRunOnConstructed());
  }

  checkAndRunOnConstructed() {
    if (!this.__constructed__) {
      this.onConstructed();
    }
    return this.__constructed__;
  }

  onConstructed() {
    if (this.__constructed__) return false;
    this.type = this.type ?? undefined;
    this.__constructed__ = true;
    return true;
  }

  configure(info) {
    this.configuring = true;
    super.configure(info);
    for (const w of this.widgets || []) {
      w.last_y = w.last_y || 0;
    }
    this.configuring = false;
  }

  clone() {
    const cloned = super.clone();
    if (cloned?.properties && window.structuredClone) {
      cloned.properties = structuredClone(cloned.properties);
    }
    return cloned;
  }

  onModeChange(from, to) {}

  removeWidget(widget) {
    if (typeof widget === "number") {
      widget = this.widgets[widget];
    }
    if (!widget) return;
    const index = this.widgets.indexOf(widget);
    if (index > -1) {
      this.widgets.splice(index, 1);
    }
    widget.onRemove?.();
  }

  replaceWidget(widgetOrSlot, newWidget) {
    let index = null;
    if (widgetOrSlot != null) {
      index = typeof widgetOrSlot === "number" ? widgetOrSlot : this.widgets.indexOf(widgetOrSlot);
      this.removeWidget(this.widgets[index]);
    }
    index = index != null ? index : this.widgets.length - 1;
    if (this.widgets.includes(newWidget)) {
      moveArrayItem(this.widgets, newWidget, index);
    } else {
      this.widgets.splice(index, 0, newWidget);
    }
  }

  defaultGetSlotMenuOptions(slot) {
    const menuInfo = [];
    if (slot?.output?.links?.length) {
      menuInfo.push({ content: "Disconnect Links", slot });
    }
    const inputOrOutput = slot.input || slot.output;
    if (inputOrOutput) {
      if (inputOrOutput.removable) {
        menuInfo.push(inputOrOutput.locked ? { content: "Cannot remove" } : { content: "Remove Slot", slot });
      }
      if (!inputOrOutput.nameLocked) {
        menuInfo.push({ content: "Rename Slot", slot });
      }
    }
    return menuInfo;
  }

  onRemoved() {
    super.onRemoved?.();
    this.removed = true;
  }

  static setUp() {}
}

export class MxdBaseServerNode extends MxdBaseNode {
  static nodeType = null;
  static nodeData = null;
  static __registeredForOverride__ = false;

  constructor(title) {
    super(title);
    this.isDropEnabled = true;
    this.serialize_widgets = true;
    this.setupFromServerNodeData();
    this.onConstructed();
  }

  getWidgets() {
    return ComfyWidgets;
  }

  async setupFromServerNodeData() {
    const nodeData = this.constructor.nodeData;
    if (!nodeData) throw Error("No node data");

    this.comfyClass = nodeData.name;
    let inputs = nodeData.input.required;
    if (nodeData.input.optional != undefined) {
      inputs = Object.assign({}, inputs, nodeData.input.optional);
    }

    const WIDGETS = this.getWidgets();
    const config = { minWidth: 1, minHeight: 1, widget: null };

    for (const inputName in inputs) {
      const inputData = inputs[inputName];
      const type = inputData[0];
      if (inputData[1]?.forceInput) {
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

        if (widgetCreated && inputData[1]?.forceInput && config?.widget) {
          if (!config.widget.options) config.widget.options = {};
          config.widget.options.forceInput = inputData[1].forceInput;
        }
        if (widgetCreated && inputData[1]?.defaultInput && config?.widget) {
          if (!config.widget.options) config.widget.options = {};
          config.widget.options.defaultInput = inputData[1].defaultInput;
        }
      }
    }

    for (const o in nodeData.output) {
      let output = nodeData.output[o];
      if (output instanceof Array) output = "COMBO";
      const outputName = nodeData.output_name[o] || output;
      const outputShape = nodeData.output_is_list[o] ? LiteGraph.GRID_SHAPE : LiteGraph.CIRCLE_SHAPE;
      this.addOutput(outputName, output, { shape: outputShape });
    }

    const s = this.computeSize();
    s[0] = Math.max(config.minWidth ?? 1, s[0] * 1.5);
    s[1] = Math.max(config.minHeight ?? 1, s[1]);
    this.size = s;
    this.serialize_widgets = true;
  }

  static registerForOverride(comfyClass, nodeData, mxdClass) {
    if (OVERRIDDEN_SERVER_NODES.has(comfyClass)) {
      throw Error(`Already have a class to override ${comfyClass.type || comfyClass.name || comfyClass.title}`);
    }
    OVERRIDDEN_SERVER_NODES.set(comfyClass, mxdClass);
    if (!mxdClass.__registeredForOverride__) {
      mxdClass.__registeredForOverride__ = true;
      mxdClass.nodeType = comfyClass;
      mxdClass.nodeData = nodeData;
      mxdClass.onRegisteredForOverride(comfyClass, mxdClass);
    }
  }

  static onRegisteredForOverride(comfyClass, mxdClass) {}
}

const OVERRIDDEN_SERVER_NODES = new Map();
const oldRegisterNodeType = LiteGraph.registerNodeType;
LiteGraph.registerNodeType = async function (nodeId, baseClass) {
  const clazz = OVERRIDDEN_SERVER_NODES.get(baseClass) || baseClass;
  return oldRegisterNodeType.call(LiteGraph, nodeId, clazz);
};
