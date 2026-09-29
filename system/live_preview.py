"""
General-purpose animated video preview during sampling, for ANY KSampler-family
node (not just LTX). Stock ComfyUI already ships an approximate RGB decode for
several video latent formats (comfy/latent_formats.py: Wan21, Wan22, LTXV,
Mochi, HunyuanVideo, Cosmos...), it just only ever shows a single static
thumbnail. This wraps whatever `latent_preview.get_previewer()` returns so it
streams a cycling batch of frames instead, over its own private
`MXD_live_preview_start` / `MXD_live_preview_frame` websocket events (plain
JSON, base64-encoded JPEG frames) -- deliberately NOT the `VHS_latentpreview` /
`b_preview` wire protocol ComfyUI-VideoHelperSuite (VHS) and ComfyUI core's own
default single-image preview both also listen on. Sharing that channel turned
out to be unreliable: whichever of us/VHS/core happened to register its
listener first could swallow the event before the others saw it, so the
frontend panel (web/nodes/live_preview_panel.js) would sometimes just never
render. A fully private channel has no such collision risk.

Formats core has NO previewer for at all (LTX 2.3 / LTXAV: no
latent_rgb_factors, no taesd decoder name) fall back to the taeltx decoder in
`nodes/ltx/preview.py`, so LTX behaves like every other model here -- animated
previews on any sampler, no extra node to wire in.

Enabled per-run via the `MXD_latentpreview` flag in the workflow's
extra_pnginfo, set by web/nodes/video_preview.js.

Also patches `latent_preview.prepare_callback` to save the final frame batch
of any run with this preview enabled to `<output_dir>/live_previews/` (see
`nodes/shared/live_preview_output.py` -- mp4 for a clip, png for a single
image), so
finished previews can be reviewed later even without a Save/VideoCombine node
in the workflow, and to notify the frontend panel (via
`MXD_live_preview_saved`) so it can swap to the saved file with real, native
playback controls.

Portions of the animated preview logic are adapted from GPL-3.0-licensed
ComfyUI-KJNodes and ComfyUI-VideoHelperSuite and were modified by Maxed Out in
2026. See THIRD_PARTY_LICENSES.md and the repository-root LICENSE.
"""
import time
import base64
from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image

import latent_preview
import server
from comfy.cli_args import args, LatentPreviewMethod

from ..nodes.shared.live_preview_output import save_preview, notify_saved

_serv = server.PromptServer.instance

_RATES = {
    "Mochi": 24 // 6,
    "LTXV": 24 // 8,
    "HunyuanVideo": 24 // 4,
    "Cosmos1CV8x8x8": 24 // 8,
    "Wan21": 16 // 4,
    "Wan22": 24 // 4,
}
_DEFAULT_RATE = 8

_FLAG_ENABLED = "MXD_latentpreview"
_FLAG_RATE = "MXD_latentpreviewrate"


def _ltx_fallback(latent_format, rate):
    """taeltx previewer for LTX formats core can't preview at all, else None."""
    # Core hands back None both when it HAS no decoder for the format and when
    # previews are switched off entirely (ComfyUI's own "Preview method: none",
    # pushed per-prompt via execution.set_preview_method). Only the first case
    # is ours to fill in -- overriding the second would show previews to someone
    # who just turned them off.
    if args.preview_method == LatentPreviewMethod.NoPreviews:
        return None
    try:
        from ..nodes.ltx.preview import make_auto_previewer
        return make_auto_previewer(latent_format, rate)
    except Exception as e:
        print(f"[MXD video preview] LTX preview fallback unavailable: {e}")
        return None


def _ltx_save_final(node_id, previewer, x0, rate):
    """Final-clip save for the LTX fallback (it unpacks the latent itself)."""
    try:
        from ..nodes.ltx.preview import save_auto_preview
        save_auto_preview(node_id, previewer, x0, rate)
    except Exception as e:
        print(f"[MXD video preview] Failed to save LTX live preview: {e}")


def _running_extra():
    try:
        return next(iter(_serv.prompt_queue.currently_running.values()))[3]["extra_pnginfo"]["workflow"]["extra"]
    except Exception:
        return {}


def _decode_frames(previewer, x0):
    """Decode every frame of a (possibly video) latent batch to an RGB tensor sequence in -1..1 range."""
    if x0.ndim == 5:
        # (B, C, T, H, W) -> (B*T, C, H, W), batch-major, matching the layout
        # decode_latent_to_preview_image below flattens to before slicing frames.
        x0 = x0.movedim(2, 1)
        x0 = x0.reshape((-1,) + x0.shape[-3:])
    if hasattr(previewer, "taesd"):
        return previewer.taesd.decode(x0).movedim(1, 3)
    reshape = getattr(previewer, "latent_rgb_factors_reshape", None)
    if reshape is not None:
        x0 = reshape(x0)
    factors = previewer.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
    bias = previewer.latent_rgb_factors_bias
    if bias is not None:
        bias = bias.to(dtype=x0.dtype, device=x0.device)
    return F.linear(x0.movedim(1, -1), factors, bias=bias)


class _MXDAnimatedPreviewer:
    """Wraps a core LatentPreviewer to stream a cycling batch of frames instead of one still."""

    def __init__(self, previewer, rate=_DEFAULT_RATE):
        self.first_preview = True
        self.last_time = 0.0
        self.c_index = 0
        self.rate = rate
        if hasattr(previewer, "taesd"):
            self.taesd = previewer.taesd
        elif hasattr(previewer, "latent_rgb_factors"):
            self.latent_rgb_factors = previewer.latent_rgb_factors
            self.latent_rgb_factors_bias = previewer.latent_rgb_factors_bias
            self.latent_rgb_factors_reshape = getattr(previewer, "latent_rgb_factors_reshape", None)
        else:
            raise ValueError("Unsupported preview type for MXD animated previews")

    def decode_latent_to_preview_image(self, preview_format, x0):
        if x0.ndim == 5:
            x0 = x0.movedim(2, 1)
            x0 = x0.reshape((-1,) + x0.shape[-3:])
        num_images = x0.size(0)
        new_time = time.time()
        num_previews = int((new_time - self.last_time) * self.rate)
        self.last_time += num_previews / self.rate
        if num_previews > num_images:
            num_previews = num_images
        elif num_previews <= 0:
            return None
        if self.first_preview:
            self.first_preview = False
            _serv.send_sync(
                "MXD_live_preview_start",
                {"length": num_images, "rate": self.rate, "id": _serv.last_node_id},
            )
            self.last_time = new_time + 1.0 / self.rate
        if self.c_index + num_previews > num_images:
            frames = x0.roll(-self.c_index, 0)[:num_previews]
        else:
            frames = x0[self.c_index:self.c_index + num_previews]
        self._send_frames(frames, self.c_index, num_images)
        self.c_index = (self.c_index + num_previews) % num_images
        return None

    def _send_frames(self, image_tensor, ind, leng):
        image_tensor = _decode_frames(self, image_tensor)
        max_size = 512
        if image_tensor.size(1) > max_size or image_tensor.size(2) > max_size:
            t = image_tensor.movedim(-1, 0)
            if t.size(2) < t.size(3):
                h = (max_size * t.size(2)) // t.size(3)
                t = F.interpolate(t, (h, max_size), mode="bilinear")
            else:
                w = (max_size * t.size(3)) // t.size(2)
                t = F.interpolate(t, (max_size, w), mode="bilinear")
            image_tensor = t.movedim(0, -1)
        previews = (
            ((image_tensor + 1.0) / 2.0)
            .clamp(0, 1)
            .mul(0xFF)
            .to(device="cpu", dtype=torch.uint8)
        )
        node_id = _serv.last_node_id
        for preview in previews:
            img = Image.fromarray(preview.numpy())
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=90)
            data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            _serv.send_sync("MXD_live_preview_frame", {"id": node_id, "index": ind, "length": leng, "data": data_url})
            ind = (ind + 1) % leng


# Captured before we patch get_previewer ourselves, so the save hook below can
# always obtain a plain (undecorated) previewer for decoding -- regardless of
# what either hook's own flag decides for a given run.
_stock_get_previewer = latent_preview.get_previewer


def _install_hook():
    if getattr(latent_preview.get_previewer, "_mxd_patched", False):
        return

    original_get_previewer = latent_preview.get_previewer

    def _mxd_get_previewer(device, latent_format, *args, **kwargs):
        previewer = original_get_previewer(device, latent_format, *args, **kwargs)

        try:
            extra = _running_extra()
            enabled = bool(extra.get(_FLAG_ENABLED, False))
            rate = extra.get(_FLAG_RATE) or _RATES.get(latent_format.__class__.__name__, _DEFAULT_RATE)
        except Exception:
            enabled = False
            rate = _DEFAULT_RATE

        if not enabled:
            return previewer

        if not hasattr(previewer, "decode_latent_to_preview"):
            # Core gave us nothing to wrap (LTX 2.3 has no preview decoder at
            # all). Fall back to taeltx rather than leaving the panel empty.
            return _ltx_fallback(latent_format, rate) or previewer

        try:
            return _MXDAnimatedPreviewer(previewer, rate)
        except ValueError:
            return previewer

    _mxd_get_previewer._mxd_patched = True
    latent_preview.get_previewer = _mxd_get_previewer
    print("[MXD video preview] Installed general animated-preview hook for all samplers.")


def _save_final_preview(node_id, previewer, x0, rate):
    try:
        frames = _decode_frames(previewer, x0)
        if frames.ndim != 4 or frames.size(0) == 0:
            return
        frames = ((frames + 1.0) / 2.0).clamp(0, 1).to(device="cpu", dtype=torch.float32)
        saved = save_preview(node_id, frames, rate)
        notify_saved(_serv, node_id, saved)
    except Exception as e:
        print(f"[MXD video preview] Failed to save live preview: {e}")


def _install_save_hook():
    if getattr(latent_preview.prepare_callback, "_mxd_patched", False):
        return

    original_prepare_callback = latent_preview.prepare_callback

    def _mxd_prepare_callback(model, steps, x0_output_dict=None):
        callback = original_prepare_callback(model, steps, x0_output_dict)

        try:
            extra = _running_extra()
            enabled = bool(extra.get(_FLAG_ENABLED, False))
            rate = extra.get(_FLAG_RATE) or _RATES.get(
                model.model.latent_format.__class__.__name__, _DEFAULT_RATE
            )
        except Exception:
            enabled = False
            rate = _DEFAULT_RATE

        if not enabled:
            return callback

        latent_format = model.model.latent_format
        raw_previewer = _stock_get_previewer(model.load_device, latent_format)
        ltx_previewer = None
        if raw_previewer is None or not hasattr(raw_previewer, "decode_latent_to_preview"):
            ltx_previewer = _ltx_fallback(latent_format, rate)
            if ltx_previewer is None:
                return callback

        node_id = _serv.last_node_id

        def wrapped(step, x0, x, total_steps):
            result = callback(step, x0, x, total_steps)
            if step + 1 >= total_steps:
                if ltx_previewer is not None:
                    _ltx_save_final(node_id, ltx_previewer, x0, rate)
                else:
                    _save_final_preview(node_id, raw_previewer, x0, rate)
            return result

        return wrapped

    _mxd_prepare_callback._mxd_patched = True
    latent_preview.prepare_callback = _mxd_prepare_callback
    print("[MXD video preview] Installed live-preview save-to-disk hook.")


_install_hook()
_install_save_hook()

