// Nodes 2.0 ("Modern Node Design") compatibility helpers.
//
// ComfyUI ships a frontend setting, `Comfy.VueNodes.Enabled`, labelled
// "Modern Node Design (Nodes 2.0)". When it is ON, nodes are rendered as
// DOM/Vue components instead of being painted onto the LiteGraph canvas.
//
// Why canvas-drawn custom widgets need help under Nodes 2.0
// --------------------------------------------------------
// Classic mode: LGraphNode.drawWidgets() paints every widget into the single
// node canvas and calls
//     widget.draw(ctx, node, w, posY, height, lowQuality)
// with `w = widget.width || node.size[0]`. Our widgets never set
// `widget.width`, so historically `w === node.size[0]` and it was harmless for
// a widget to measure right-aligned parts from `node.size[0]` directly.
//
// Nodes 2.0: core wraps each legacy custom widget in its `WidgetLegacy` Vue
// component, which gives the widget its OWN <canvas> sized to the DOM row, sets
// `widget.width` to that row's clientWidth, and calls
//     widget.draw(ctx, node, rowWidth, 1, height)
// The drawing surface is now the row, not the node, and `node.size[0]` no
// longer matches it. Anything positioned from the right edge via `node.size[0]`
// (strength number boxes, column headers, row backgrounds) therefore lands in
// the wrong place — the classic symptom of a custom widget "not working right"
// in Nodes 2.0.
//
// nodeDrawWidth() returns the width a widget should actually draw into.
// Deliberately, with Nodes 2.0 OFF it returns `node.size[0]` — identical to the
// pre-Nodes-2.0 behavior — so turning the setting off is a full revert and
// classic rendering can never be affected by this module.
//
// Repainting: see redrawWidgets() below.
import { app } from "../../../scripts/app.js";

const SETTING_ID = "Comfy.VueNodes.Enabled";

// The setting is read from inside canvas draw loops, so memoize it briefly
// rather than hitting the settings store on every widget of every frame. A
// quarter second is imperceptible when the user flips the toggle.
const CACHE_MS = 250;

let cachedValue = false;
let cachedAt = -Infinity;

// True when ComfyUI is rendering nodes as DOM/Vue components.
// Falls back to false (classic) on frontends too old to know the setting.
export function isNodes2Enabled() {
  const now = performance.now();
  if (now - cachedAt < CACHE_MS) {
    return cachedValue;
  }
  cachedAt = now;
  try {
    cachedValue = app?.extensionManager?.setting?.get(SETTING_ID) === true;
  } catch (e) {
    cachedValue = false;
  }
  return cachedValue;
}

// The width a custom widget's draw() should treat as its drawing surface.
// Pass the `w` argument draw() received. Classic mode always yields
// node.size[0], preserving the original behavior exactly.
export function nodeDrawWidth(node, w) {
  if (isNodes2Enabled() && typeof w === "number" && w > 0) {
    return w;
  }
  return node?.size?.[0] ?? 0;
}

// Repaint a canvas-drawn custom widget after code changed its value directly.
//
// Classic mode repaints the whole node canvas, so setDirtyCanvas() is enough.
// Under Nodes 2.0 each legacy widget lives in its OWN <canvas> inside core's
// WidgetLegacy component, and the graph canvas being dirty says nothing about
// that private canvas. Core's fix is `widget.triggerDraw`, which it documents as
// "compatibility method for widgets implementing the draw method when displayed
// in non-canvas renderers ... set by the current renderer implementation".
// Core only wires it into `widget.callback`; our widgets mutate `this.value`
// directly, so we call it ourselves.
//
// It is absent on classic frontends and on older ones that predate it, hence
// the optional call — this is a no-op there, which is exactly right.
export function redrawWidget(widget) {
  widget?.triggerDraw?.();
}

// Repaint every widget on a node. Used when one interaction changes several
// rows at once (toggle-all, reordering, a row being removed).
export function redrawWidgets(node) {
  for (const widget of node?.widgets || []) {
    widget.triggerDraw?.();
  }
  node?.setDirtyCanvas?.(true, true);
}
