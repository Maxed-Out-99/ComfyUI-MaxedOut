// HTML (DOM-widget) rows for the Power Lora Loader nodes.
//
// This file is a PARALLEL implementation of the row widgets that live in
// power_lora_base.js / power_lora_loader.js. Those
// paint themselves onto the LiteGraph canvas; these build real HTML elements and
// register them with `node.addDOMWidget()`. Nothing here modifies the canvas
// widgets — `usePowerLoraDom()` decides which of the two a node is built from,
// so the classic rows remain reachable and unchanged forever.
//
// Why an HTML version exists
// --------------------------
// Under ComfyUI's Nodes 2.0 (DOM node rendering), core hosts a canvas-drawing
// custom widget in its `WidgetLegacy` component, which only repaints that
// widget's private canvas when the widget's `callback` fires. Our canvas rows
// mutate `this.value` directly and rely on `node.setDirtyCanvas()`, which
// repaints the graph canvas but NOT the widget's private canvas — so toggling a
// lora or nudging a strength changed the value with no visible feedback. Real
// DOM elements sidestep the whole bridge: the browser renders them, and both
// classic ComfyUI and Nodes 2.0 mount DOM widgets natively (core uses the same
// API for every multiline prompt box).
//
// Serialization is deliberately IDENTICAL to the canvas rows: widget names stay
// `lora_N` and values stay `{on, lora, strength[, strengthTwo]}`. Saved
// workflows cannot tell which implementation drew them.
import { app } from "../../../scripts/app.js";
import { isNodes2Enabled } from "../lib/mxd_nodes2.js";
import { injectCss } from "../lib/mxd_shared_utils.js";
import { showLoraChooser } from "../lib/mxd_utils_menu.js";
import { MxdLoraInfoDialog } from "../lib/mxd_dialog_info.js";
import { LORA_INFO_SERVICE } from "../lib/mxd_model_info_service.js";

const CSS_HREF = new URL("../lib/mxd_power_lora.css", import.meta.url).pathname;

export const ROW_HEIGHT = 20;
export const HEADER_HEIGHT = 16;
export const BUTTON_HEIGHT = 22;

const STRENGTH_STEP = 0.05;
const SETTING_ROW_STYLE = "MXD.PowerLora.RowStyle";
const STYLE_AUTO = "Auto (follow Nodes 2.0)";
const STYLE_CLASSIC = "Classic canvas";
const STYLE_HTML = "HTML";

// Which row implementation new nodes should be built from.
// Auto keeps classic behavior whenever Nodes 2.0 is off, so turning Nodes 2.0
// off is a complete return to the original rendering.
export function usePowerLoraDom() {
  let style = STYLE_AUTO;
  try {
    style = app?.extensionManager?.setting?.get(SETTING_ROW_STYLE) ?? STYLE_AUTO;
  } catch (e) {
    style = STYLE_AUTO;
  }
  if (style === STYLE_CLASSIC) return false;
  if (style === STYLE_HTML) return true;
  return isNodes2Enabled();
}

app.registerExtension({
  name: "mxd.PowerLoraRowStyle",
  settings: [
    {
      id: SETTING_ROW_STYLE,
      category: ["MaxedOut", "Power Lora Loader", "Row style"],
      name: "Power Lora Loader row style",
      tooltip:
        "How the Lora Loader MXD / LTX2 Lora Loader MXD rows are drawn. " +
        "Auto uses HTML rows when ComfyUI's Nodes 2.0 is on and the original " +
        "canvas rows when it is off. Reload the page after changing this.",
      type: "combo",
      options: [STYLE_AUTO, STYLE_CLASSIC, STYLE_HTML],
      defaultValue: STYLE_AUTO,
    },
  ],
});

// --- small helpers -------------------------------------------------------

function fmt(value) {
  return Number(value ?? 0).toFixed(2);
}

function el(tag, className, parent) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (parent) parent.appendChild(node);
  return node;
}

function button(className, parent, label) {
  const b = el("button", className, parent);
  b.type = "button";
  if (label != null) b.textContent = label;
  // Keep presses from reaching the canvas (node drag) or the Vue node body.
  b.addEventListener("pointerdown", (e) => e.stopPropagation());
  return b;
}

// Every DOM row on a node registers a refresher so that changes made elsewhere
// (toggle-all, the slot context menu, a workflow load) can repaint the row.
function registerRefresher(node, fn) {
  if (!node.__mxdDomRefreshers) node.__mxdDomRefreshers = [];
  node.__mxdDomRefreshers.push(fn);
  return fn;
}

export function refreshPowerLoraDom(node) {
  for (const fn of node?.__mxdDomRefreshers || []) {
    try {
      fn();
    } catch (e) {
      /* a broken row must not stop the rest from repainting */
    }
  }
}

// Grow the node to fit its widgets, matching what the canvas rows do.
function growNodeToFit(node) {
  const computed = node.computeSize();
  node.size = node.size || [0, 0];
  node.size[1] = Math.max(node.size[1] ?? 15, computed[1]);
  node.setDirtyCanvas(true, true);
}

// Registers `element` as a DOM widget and wires the shared plumbing:
// value <-> state, serialization, fixed row height, and element teardown.
//
// Serialization note: core saves workflows from `widget.value` (JSON-cloned)
// but builds prompts from `serializeValue()`. Every widget here keeps the same
// `value` its canvas counterpart had, and no widget opts out of serialization,
// so `widgets_values` comes out byte-identical either way — a workflow saved in
// one row style loads correctly in the other.
function addRowWidget(node, { name, element, height, getState, setState, serializeState }) {
  injectCss(CSS_HREF);

  const widget = node.addDOMWidget(name, "custom", element, {
    hideOnZoom: true,
    getValue: getState,
    setValue: setState,
    getMinHeight: () => height,
    getMaxHeight: () => height,
    getHeight: () => height,
  });

  // Hand back a plain snapshot, never the live proxy.
  widget.serializeValue = serializeState ?? (() => ({ ...getState() }));

  const priorOnRemove = widget.onRemove?.bind(widget);
  widget.onRemove = () => {
    priorOnRemove?.();
    element.remove();
    const list = node.__mxdDomRefreshers;
    if (list && widget.__mxdRefresher) {
      const i = list.indexOf(widget.__mxdRefresher);
      if (i > -1) list.splice(i, 1);
    }
  };

  return widget;
}

// Wraps the row's raw value object so that ANY external mutation — including
// `widget.value.on = false` from toggleAllLoras or the slot context menu —
// repaints the HTML. Without this, code written for the canvas rows would
// silently change state that the DOM never re-rendered.
function reactiveState(raw, onChange) {
  return new Proxy(raw, {
    set(target, prop, value) {
      target[prop] = value;
      onChange();
      return true;
    },
    deleteProperty(target, prop) {
      delete target[prop];
      onChange();
      return true;
    },
  });
}

// A `-  value  +` stepper with click-to-type and horizontal drag-to-scrub,
// mirroring the canvas rows' strength control.
function strengthControl(parent, { get, set, onCommit }) {
  const wrap = el("div", "mxd-lora-strength", parent);
  const dec = button(null, wrap, "‹");
  const input = el("input", null, wrap);
  const inc = button(null, wrap, "›");

  input.type = "text";
  input.spellcheck = false;

  const render = () => {
    if (document.activeElement !== input) {
      input.value = fmt(get());
    }
  };

  const step = (direction) => {
    set(Math.round((Number(get() ?? 1) + STRENGTH_STEP * direction) * 100) / 100);
    render();
    onCommit?.();
  };

  dec.addEventListener("click", (e) => {
    e.stopPropagation();
    step(-1);
  });
  inc.addEventListener("click", (e) => {
    e.stopPropagation();
    step(1);
  });

  // Drag horizontally to scrub; a click that never moved focuses for typing.
  let dragStartX = null;
  let dragStartValue = 0;
  let dragged = false;

  input.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    if (document.activeElement === input) return; // already typing
    e.preventDefault();
    dragStartX = e.clientX;
    dragStartValue = Number(get() ?? 1);
    dragged = false;
    input.setPointerCapture(e.pointerId);
  });

  input.addEventListener("pointermove", (e) => {
    if (dragStartX == null) return;
    const dx = e.clientX - dragStartX;
    if (!dragged && Math.abs(dx) < 3) return;
    dragged = true;
    set(Math.round((dragStartValue + dx * STRENGTH_STEP) * 100) / 100);
    render();
  });

  input.addEventListener("pointerup", (e) => {
    if (dragStartX == null) return;
    input.releasePointerCapture?.(e.pointerId);
    dragStartX = null;
    if (dragged) {
      onCommit?.();
    } else {
      input.focus();
      input.select();
    }
  });

  const commitTyped = () => {
    const parsed = Number.parseFloat(input.value);
    if (Number.isFinite(parsed)) {
      set(parsed);
    }
    render();
    onCommit?.();
  };

  input.addEventListener("change", commitTyped);
  input.addEventListener("blur", commitTyped);
  input.addEventListener("keydown", (e) => {
    e.stopPropagation(); // don't let the canvas' hotkeys steal typing
    if (e.key === "Enter") {
      input.blur();
    } else if (e.key === "Escape") {
      render();
      input.blur();
    }
  });

  return { render, input };
}

// --- lora row ------------------------------------------------------------

// One lora: [on/off] name [i] [x] [strength] (+ a second strength when `dual`).
// `dual` mirrors the "Show Strengths" property of Lora Loader MXD; the row
// re-reads it on every refresh so switching the property updates live.
export function addDomLoraRow(node, name, { dual = false } = {}) {
  const raw = { on: true, lora: null, strength: 1 };
  if (dual) raw.strengthTwo = null;

  let loraInfo = null;
  let loraInfoPromise = null;

  const root = el("div", "mxd-lora-row");
  const toggle = button("mxd-lora-toggle", root);
  toggle.setAttribute("role", "switch");
  const nameBtn = button("mxd-lora-name", root);
  const nameText = el("span", null, nameBtn);
  const infoBtn = button("mxd-lora-icon mxd-lora-info", root, "i");
  const removeBtn = button("mxd-lora-icon mxd-lora-remove", root, "✕");

  let widget = null;
  const state = reactiveState(raw, () => render());

  const isDualActive = () => dual && node.isShowingSeparateStrengths?.() === true;

  const outOfRange = (value) => {
    if (loraInfo?.strengthMax != null && value > loraInfo.strengthMax) return true;
    if (loraInfo?.strengthMin != null && value < loraInfo.strengthMin) return true;
    return false;
  };

  // Left stepper only exists in dual mode: it is the MODEL strength, and the
  // right stepper becomes CLIP (strengthTwo) — same column order as the canvas
  // rows, where the rightmost number is clip.
  const modelStrength = strengthControl(root, {
    get: () => raw.strength ?? 1,
    set: (v) => {
      raw.strength = v;
    },
    onCommit: () => refreshPowerLoraDom(node),
  });

  const clipStrength = strengthControl(root, {
    get: () => (isDualActive() ? (raw.strengthTwo ?? 1) : (raw.strength ?? 1)),
    set: (v) => {
      if (isDualActive()) raw.strengthTwo = v;
      else raw.strength = v;
    },
    onCommit: () => refreshPowerLoraDom(node),
  });

  function render() {
    const dualActive = isDualActive();
    if (dualActive && raw.strengthTwo == null) {
      raw.strengthTwo = raw.strength ?? 1;
    } else if (!dualActive && dual && raw.strengthTwo != null) {
      raw.strengthTwo = null;
    }

    root.classList.toggle("mxd-lora-off", raw.on !== true);
    toggle.setAttribute("aria-checked", raw.on === true ? "true" : "false");

    const label = String(raw.lora || "None");
    nameText.textContent = label;
    nameBtn.title = label;

    const hasLora = !!raw.lora && raw.lora !== "None";
    infoBtn.style.display = hasLora ? "" : "none";

    // In dual mode both steppers show; otherwise only the single (clip slot).
    modelStrength.input.parentElement.style.display = dualActive ? "" : "none";
    modelStrength.render();
    clipStrength.render();

    modelStrength.input.classList.toggle("mxd-lora-out-of-range", outOfRange(raw.strength ?? 1));
    clipStrength.input.classList.toggle(
      "mxd-lora-out-of-range",
      outOfRange(dualActive ? (raw.strengthTwo ?? 1) : (raw.strength ?? 1)),
    );
  }

  function loadLoraInfo(force = false) {
    if (!loraInfoPromise || force) {
      const hasLora = raw.lora && raw.lora !== "None";
      loraInfoPromise = (hasLora
        ? LORA_INFO_SERVICE.getInfo(raw.lora, force, true)
        : Promise.resolve(null)
      ).then((info) => {
        loraInfo = info;
        render();
      });
    }
    return loraInfoPromise;
  }

  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    raw.on = raw.on !== true;
    render();
    refreshPowerLoraDom(node);
  });

  nameBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    showLoraChooser(e, (value) => {
      if (typeof value === "string") {
        raw.lora = value;
        loraInfo = null;
        loadLoraInfo(true);
        render();
      }
    });
  });

  infoBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    showInfoDialog();
  });

  removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    node.removeWidget(widget);
    refreshPowerLoraDom(node);
    node.setDirtyCanvas(true, true);
  });

  function showInfoDialog() {
    if (!raw.lora || raw.lora === "None") return;
    const dialog = new MxdLoraInfoDialog(raw.lora).show();
    dialog.addEventListener("close", (e) => {
      if (e.detail?.dirty) loadLoraInfo(true);
    });
  }

  widget = addRowWidget(node, {
    name,
    element: root,
    height: ROW_HEIGHT,
    getState: () => state,
    // Mirrors PowerLoraLoaderWidget.serializeValue exactly: `strengthTwo` is
    // OMITTED (not null) when strengths aren't split, because the Python side
    // tests `"strengthTwo" in lora` when building its lora list output.
    serializeState: () => {
      const out = { ...raw };
      if (!dual || !isDualActive()) {
        delete out.strengthTwo;
      } else {
        raw.strengthTwo = raw.strengthTwo ?? 1;
        out.strengthTwo = raw.strengthTwo;
      }
      return out;
    },
    setState: (v) => {
      if (v && typeof v === "object") {
        Object.assign(raw, v);
      }
      loraInfo = null;
      loraInfoPromise = null;
      loadLoraInfo();
      render();
    },
  });

  // Keep the API the canvas rows expose, so the node class and its slot menus
  // work against either implementation.
  widget.setLora = (lora) => {
    raw.lora = lora;
    loraInfo = null;
    loadLoraInfo(true);
    render();
  };
  widget.showLoraInfoDialog = showInfoDialog;
  widget.__mxdRefresher = registerRefresher(node, render);

  render();
  loadLoraInfo();
  return widget;
}

// --- header row ----------------------------------------------------------

// "[toggle all] Toggle All        Model   Strength/Clip"
export function addDomHeaderRow(node, { dual = false } = {}) {
  const root = el("div", "mxd-lora-row mxd-lora-header");
  const toggle = button("mxd-lora-toggle", root);
  toggle.setAttribute("role", "switch");
  const label = el("span", "mxd-lora-header-label", root);
  label.textContent = "Toggle All";
  // Invisible stand-ins for each row's info/remove buttons, so the captions
  // below line up with the strength steppers under the same flex math.
  el("span", "mxd-lora-icon mxd-lora-spacer", root);
  el("span", "mxd-lora-icon mxd-lora-spacer", root);
  const modelCol = el("span", "mxd-lora-header-col", root);
  const strengthCol = el("span", "mxd-lora-header-col", root);

  function render() {
    const anyRows = node.hasLoraWidgets?.() === true;
    root.style.display = anyRows ? "" : "none";
    if (!anyRows) return;

    const dualActive = dual && node.isShowingSeparateStrengths?.() === true;
    const all = node.allLorasState?.();
    toggle.setAttribute("aria-checked", all === true ? "true" : all === null ? "mixed" : "false");
    modelCol.style.display = dualActive ? "" : "none";
    modelCol.textContent = "Model";
    strengthCol.textContent = dualActive ? "Clip" : "Strength";
  }

  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    node.toggleAllLoras?.();
    refreshPowerLoraDom(node);
  });

  // Same `value` the canvas PowerLoraLoaderHeaderWidget carries, so it occupies
  // the same slot in widgets_values.
  const widget = addRowWidget(node, {
    name: "PowerLoraLoaderHeaderWidget",
    element: root,
    height: HEADER_HEIGHT,
    getState: () => ({ type: "PowerLoraLoaderHeaderWidget" }),
    setState: () => {},
  });

  widget.__mxdRefresher = registerRefresher(node, render);
  render();
  return widget;
}

// --- "+ Add Lora" --------------------------------------------------------

export function addDomAddLoraButton(node, onPick) {
  const root = el("button", "mxd-lora-add");
  root.type = "button";
  root.textContent = "+ Add Lora";
  root.addEventListener("pointerdown", (e) => e.stopPropagation());
  root.addEventListener("click", (e) => {
    e.stopPropagation();
    showLoraChooser(e, (value) => {
      if (typeof value === "string" && value !== "NONE") {
        onPick(value);
        growNodeToFit(node);
        refreshPowerLoraDom(node);
      }
    });
  });

  // Empty-string value matches the canvas MxdBetterButtonWidget it replaces.
  return addRowWidget(node, {
    name: "mxd_add_lora_button",
    element: root,
    height: BUTTON_HEIGHT,
    getState: () => "",
    setState: () => {},
  });
}
