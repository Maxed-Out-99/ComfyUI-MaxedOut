import { app } from "../../scripts/app.js";

class LogSession {
  constructor(name) {
    this.name = name || "[mxd]";
  }

  logParts(level, message, ...args) {
    const method = level === "error" ? "error" : level === "warn" ? "warn" : "log";
    return [method, [`${this.name} ${message}`, ...args]];
  }

  errorParts(message, ...args) {
    return this.logParts("error", message, ...args);
  }
}

class MxdRuntime {
  constructor() {
    this.loadingApiJson = null;
    this.lastCanvasMouseEvent = null;
    this.canvasCurrentlyCopyingToClipboard = false;
    this.canvasCurrentlyCopyingToClipboardWithMultipleNodes = false;
    this.canvasCurrentlyPastingFromClipboard = false;
    this.initializeHooks();
  }

  initializeHooks() {
    const runtime = this;

    const loadApiJson = app.loadApiJson;
    app.loadApiJson = async function (apiData, fileName) {
      runtime.loadingApiJson = apiData;
      try {
        return await loadApiJson.apply(app, [...arguments]);
      } finally {
        runtime.loadingApiJson = null;
      }
    };

    const adjustMouseEvent = LGraphCanvas.prototype.adjustMouseEvent;
    LGraphCanvas.prototype.adjustMouseEvent = function (e) {
      adjustMouseEvent.apply(this, [...arguments]);
      runtime.lastCanvasMouseEvent = e;
    };

    const copyToClipboard = LGraphCanvas.prototype.copyToClipboard;
    LGraphCanvas.prototype.copyToClipboard = function (items) {
      runtime.canvasCurrentlyCopyingToClipboard = true;
      runtime.canvasCurrentlyCopyingToClipboardWithMultipleNodes =
        Object.values(items || this.selected_nodes || {}).length > 1;
      try {
        return copyToClipboard.apply(this, [...arguments]);
      } finally {
        runtime.canvasCurrentlyCopyingToClipboard = false;
        runtime.canvasCurrentlyCopyingToClipboardWithMultipleNodes = false;
      }
    };

    const pasteFromClipboard = LGraphCanvas.prototype.pasteFromClipboard;
    LGraphCanvas.prototype.pasteFromClipboard = function () {
      runtime.canvasCurrentlyPastingFromClipboard = true;
      try {
        return pasteFromClipboard.apply(this, [...arguments]);
      } finally {
        runtime.canvasCurrentlyPastingFromClipboard = false;
      }
    };
  }

  newLogSession(name) {
    return new LogSession(name);
  }

  showMessage({ id, type, message, timeout }) {
    const msgId = id || `mxd-${Date.now()}`;
    let container = document.querySelector(".mxd-top-messages-container");
    if (!container) {
      container = document.createElement("div");
      container.className = "mxd-top-messages-container";
      document.body.appendChild(container);
    }
    const node = document.createElement("div");
    node.className = `mxd-top-message ${type || "info"}`;
    node.setAttribute("msg-id", msgId);
    node.textContent = message || "";
    container.appendChild(node);
    setTimeout(() => node.remove(), timeout || 3000);
  }

  isDevMode() {
    return false;
  }
}

export const mxdRuntime = new MxdRuntime();
window.mxdRuntime = mxdRuntime;
