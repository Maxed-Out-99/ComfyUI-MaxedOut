import { app } from "../../../scripts/app.js";

// Exposes the model storage auto-register root as a ComfyUI setting, as an
// alternative to MAXEDOUT_MODEL_STORAGE / model_storage_config.json for
// people who'd rather not touch a config file. Purely a convenience UI --
// the actual registration happens at import time in
// model_paths_autoregister_mxd.py, which reads this setting back out of
// user/default/comfy.settings.json, so a change here only takes effect
// after restarting the ComfyUI server.
app.registerExtension({
    name: "ComfyUI-MaxedOut.ModelStorageSettings",

    setup() {
        app.ui.settings.addSetting({
            id: "MXD.ModelStorageRoot",
            category: ["MXD", "Model Storage", "Root Folder"],
            name: "Model storage root folder (requires restart)",
            tooltip: "Every subfolder inside this path is auto-registered as a model folder, the same way ComfyUI/models/<type> works -- e.g. a 'loras' subfolder here behaves like ComfyUI/models/loras. Leave blank to disable. Applying a change requires restarting the ComfyUI server.",
            type: "text",
            defaultValue: "",
        });
    },
});
