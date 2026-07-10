import { app } from "../../../scripts/app.js";

// Adds a global "preview any sampler as it renders" toggle, backed by
// video_preview_mxd.py's general get_previewer hook. Writes its own
// MXD_latentpreview / MXD_latentpreviewrate workflow-extra keys, deliberately
// separate from VHS's own VHS_latentpreview/VHS_latentpreviewrate keys, so
// this toggle and VideoHelperSuite's identically-named setting don't fight
// over the same flag -- each hook only wraps the previewer when ITS OWN flag
// is on. The actual rendering (web/live_preview_panel_mxd.js) is a global
// floating panel, not a per-node widget, so it isn't affected by which
// workflow tab is open.
app.registerExtension({
    name: "ComfyUI-MaxedOut.VideoPreview",

    setup() {
        app.ui.settings.addSetting({
            id: "MXD.VideoPreview",
            category: ["MXD", "Sampling", "Latent Previews"],
            name: "Display animated previews when sampling",
            tooltip: "Shows a live, playable video preview on any sampler node (Wan, LTX, etc.) while it runs, with pause/play and a copy saved to output/live_previews when the run finishes. On by default -- turn off here if you'd rather not pay the decode overhead.",
            type: "boolean",
            defaultValue: true,
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
            if (res?.workflow?.extra) {
                res.workflow.extra["MXD_latentpreview"] = app.ui.settings.getSettingValue("MXD.VideoPreview");
                res.workflow.extra["MXD_latentpreviewrate"] = app.ui.settings.getSettingValue("MXD.VideoPreviewRate");
            }
            return res;
        };
    },
});
