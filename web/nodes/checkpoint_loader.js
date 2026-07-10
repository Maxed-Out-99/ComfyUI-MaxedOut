import { app } from "../../../scripts/app.js";
import { mxdApi } from "../lib/mxd_api.js";
import { replaceWidgetWithModelRow } from "../lib/mxd_model_row_widget.js";
import { MxdCheckpointInfoDialog } from "../lib/mxd_dialog_info.js";

const NODE_TYPE = "LoadCheckpointMXD";

app.registerExtension({
  name: "mxd.LoadCheckpoint",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      replaceWidgetWithModelRow(this, "ckpt_name", {
        getFileList: () => mxdApi.getCheckpointList(),
        chooserTitle: "Choose a Checkpoint",
        dialogClass: MxdCheckpointInfoDialog,
      });
      this.setDirtyCanvas(true, true);
      return r;
    };
  },
});
