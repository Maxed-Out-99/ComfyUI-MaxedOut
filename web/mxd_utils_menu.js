import { app } from "../../scripts/app.js";
import { mxdApi } from "./mxd_api.js";

export async function showLoraChooser(event, callback, parentMenu, loras) {
  const canvas = app.canvas;
  if (!loras) {
    loras = ["None", ...(await mxdApi.getLoras().then((items) => items.map((l) => l.file)) )];
  }
  new LiteGraph.ContextMenu(loras, {
    event,
    parentMenu: parentMenu != null ? parentMenu : undefined,
    title: "Choose a LoRA",
    scale: Math.max(1, canvas.ds?.scale ?? 1),
    className: "dark",
    callback,
  });
}
