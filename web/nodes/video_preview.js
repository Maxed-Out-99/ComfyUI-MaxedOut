import { app } from "../../../scripts/app.js";

// Adds a global "preview any sampler as it renders" toggle, backed by
// system/live_preview.py's general get_previewer hook. Writes its own
// MXD_latentpreview / MXD_latentpreviewrate workflow-extra keys, deliberately
// separate from VHS's own VHS_latentpreview/VHS_latentpreviewrate keys, so
// this toggle and VideoHelperSuite's identically-named setting don't fight
// over the same flag -- each hook only wraps the previewer when ITS OWN flag
// is on. The actual rendering (web/nodes/live_preview_panel.js) is a global
// floating panel, not a per-node widget, so it isn't affected by which
// workflow tab is open.
app.registerExtension({
    name: "ComfyUI-MaxedOut.VideoPreview",

    setup() {
        app.ui.settings.addSetting({
            id: "MXD.VideoPreview",
            category: ["MXD", "Sampling", "Latent Previews"],
            name: "Display animated previews when sampling",
            tooltip: "Opt in to a live, playable preview for any sampler (Wan, LTX, images, etc.). Finished previews are saved to output/live_previews. Leave off to avoid all MXD preview decoding and saving overhead.",
            type: "boolean",
            defaultValue: false,
        });

        app.ui.settings.addSetting({
            id: "MXD.VideoPreviewRate",
            category: ["MXD", "Sampling", "Latent Preview Rate"],
            name: "Playback rate override (0 = auto per model)",
            type: "number",
            attrs: { min: 0, step: 1, max: 60 },
            defaultValue: 0,
        });

        const originalGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function (...args) {
            const res = await originalGraphToPrompt.apply(this, args);
            if (res?.workflow) {
                res.workflow.extra ??= {};
                res.workflow.extra["MXD_latentpreview"] = app.ui.settings.getSettingValue("MXD.VideoPreview") === true;
                res.workflow.extra["MXD_latentpreviewrate"] = app.ui.settings.getSettingValue("MXD.VideoPreviewRate");
            }
            return res;
        };
    },
});

