import { app } from "../../../scripts/app.js";

const inputOrder = [
  "model", "source_image", "source_image_b", "vae", "target_latent",
  "ref_boost_mask", "ref_boost", "ref_boost_a", "fit_mode",
];

app.registerExtension({
  name: "mxd.Krea2Edit",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Krea2EditModelPatchMXD") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      // Interleave optional sockets without making the second reference required.
      this.inputs.sort((a, b) => inputOrder.indexOf(a.name) - inputOrder.indexOf(b.name));
      return result;
    };
  },
});
