"""Automatic LTX live preview via the tiny taeltx autoencoder.

Core ComfyUI has no preview for the LTXAV format used by LTX 2.3
(latent_rgb_factors is None and there is no taesd_decoder_name), so stock
ComfyUI shows nothing at all while an LTX 2.3 sampler runs. This module
decodes latent frames with the tiny "taeltx" autoencoder instead. The taeltx
model is auto-discovered in the vae / vae_approx model folders; if it isn't
found it is downloaded to the configured vae model folder.

Two ways in, both landing on the same previewer:
  * automatic -- system/live_preview.py's get_previewer hook calls
    make_auto_previewer() whenever core hands back no previewer for an LTX
    format, so LTX 2.3 previews on ANY sampler (SamplerCustomAdvanced,
    KSampler, ...) with no extra node wired, exactly like every other model.
The automatic path needs two things core's plain callback doesn't hand it: LTX
2.3 packs audio+video latents into one flat tensor, and I2V runs append guide
frames that must not be previewed. Both are recovered by
_install_sample_context_hook(), which wraps comfy.samplers.CFGGuider.outer_sample
to record latent_shapes / keyframe count for the duration of the sample.

Sends MXD_live_preview_start / MXD_live_preview_frame / MXD_live_preview_saved
websocket events consumed by web/nodes/live_preview_panel.js. Final clips are
saved to <output>/live_previews.

TAE decode path adapted from GPL-3.0-licensed ComfyUI-KJNodes and
ComfyUI-VideoHelperSuite, then modified by Maxed Out in 2026. See
THIRD_PARTY_LICENSES.md and the repository-root LICENSE.
"""
from __future__ import annotations
import os
import base64
import time
import urllib.error
import urllib.request
from io import BytesIO
from PIL import Image
from threading import Lock, Thread

import torch
import torch.nn.functional as F

import comfy
import comfy.latent_formats
import comfy.model_management
import comfy.samplers
import comfy.utils
import server

from ..shared.live_preview_output import save_preview, notify_saved

_serv = server.PromptServer.instance

_TAELTX_FILENAME = "taeltx2_3.safetensors"
_TAELTX_URL = "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors?download=true"
_TAELTX_DOWNLOAD_LOCK = Lock()

# One taeltx instance per file, shared by every previewer. The automatic path
# asks for it on every sampler run, and it is a few MB of weights -- reloading
# it from disk each time would stall the start of every run for no reason.
_TAELTX_CACHE = {}

def _find_taeltx_path(folder_paths):
    for folder in ("vae", "vae_approx"):
        try:
            names = folder_paths.get_filename_list(folder)
        except Exception:
            continue
        name = next((fn for fn in names if "taeltx" in fn.lower()), None)
        if name is not None:
            path = folder_paths.get_full_path(folder, name)
            if path:
                return path
    return None


def _download_taeltx(folder_paths):
    try:
        vae_dirs = folder_paths.get_folder_paths("vae")
    except Exception as exc:
        print(f"[MXD LTX preview] cannot find ComfyUI vae model folder: {exc}")
        return None

    if not vae_dirs:
        print("[MXD LTX preview] cannot find ComfyUI vae model folder.")
        return None

    target_dir = vae_dirs[0]
    target_path = os.path.join(target_dir, _TAELTX_FILENAME)
    partial_path = f"{target_path}.part"

    with _TAELTX_DOWNLOAD_LOCK:
        if os.path.isfile(target_path):
            return target_path

        try:
            os.makedirs(target_dir, exist_ok=True)
            print(f"[MXD LTX preview] downloading {_TAELTX_FILENAME} to {target_path}")
            request = urllib.request.Request(_TAELTX_URL, headers={"User-Agent": "ComfyUI-MaxedOut"})
            with urllib.request.urlopen(request, timeout=120) as response, open(partial_path, "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if not os.path.isfile(partial_path) or os.path.getsize(partial_path) == 0:
                raise RuntimeError("downloaded file is empty")
            os.replace(partial_path, target_path)
            try:
                folder_paths.get_filename_list("vae")
            except Exception:
                pass
            print(f"[MXD LTX preview] downloaded {_TAELTX_FILENAME}")
            return target_path
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            try:
                if os.path.exists(partial_path):
                    os.remove(partial_path)
            except OSError:
                pass
            print(f"[MXD LTX preview] failed to download {_TAELTX_FILENAME}: {exc}")
            return None


def _load_taeltx():
    """Load the taeltx TAE from the vae / vae_approx model folders. Returns a VAE or None."""
    try:
        import folder_paths
        from comfy.sd import VAE
    except Exception:
        return None

    path = _find_taeltx_path(folder_paths)
    if not path:
        path = _download_taeltx(folder_paths)
    if not path:
        return None

    cached = _TAELTX_CACHE.get(path)
    if cached is not None:
        return cached

    try:
        taeltx = VAE(comfy.utils.load_torch_file(path))
        taeltx.first_stage_model.show_progress_bar = False
    except Exception as exc:
        print(f"[MXD LTX preview] failed to load taeltx ({path}): {exc}")
        return None
    _TAELTX_CACHE[path] = taeltx
    return taeltx


########################################################################################################################
# Sampling context — what core's plain preview callback doesn't get told
#
# The callback latent_preview.prepare_callback() builds only ever receives the
# raw x0 tensor. For LTX 2.3 that isn't enough to preview: audio+video runs
# arrive packed into a flat [B, 1, total] tensor (comfy.utils.pack_latents),
# and I2V runs carry guide frames appended at the end of the video latent.
# CFGGuider.outer_sample() is the one place that has both — latent_shapes as an
# argument and the (not yet re-processed) conds on the guider — so wrap it and
# stash them for the duration of the sample.
_SAMPLE_CTX = {"latent_shapes": None, "num_keyframes": 0}


def _count_keyframes(guider):
    """Number of I2V guide frames appended to the end of the video latent, or 0."""
    try:
        positive = (getattr(guider, "conds", None) or {}).get("positive") or []
        kf = positive[0].get("keyframe_idxs") if positive else None
        if kf is None:
            return 0
        return len(torch.unique(kf[0, 0, :, 0]))
    except Exception:
        return 0


def _install_sample_context_hook():
    if getattr(comfy.samplers.CFGGuider.outer_sample, "_mxd_patched", False):
        return

    original_outer_sample = comfy.samplers.CFGGuider.outer_sample

    def _mxd_outer_sample(self, *args, **kwargs):
        previous = dict(_SAMPLE_CTX)
        try:
            shapes = kwargs.get("latent_shapes")
            if shapes is None and len(args) >= 9:
                shapes = args[8]
            _SAMPLE_CTX["latent_shapes"] = shapes
            _SAMPLE_CTX["num_keyframes"] = _count_keyframes(self)
        except Exception:
            _SAMPLE_CTX.update(previous)
        try:
            return original_outer_sample(self, *args, **kwargs)
        finally:
            _SAMPLE_CTX.update(previous)

    _mxd_outer_sample._mxd_patched = True
    comfy.samplers.CFGGuider.outer_sample = _mxd_outer_sample
    print("[MXD LTX preview] Installed sampling-context hook for automatic LTX previews.")


def _video_latent_from_x0(x0):
    """Pull the previewable 5D video latent out of whatever the sampler handed the callback."""
    if x0 is None:
        return None
    if x0.ndim != 5:
        shapes = _SAMPLE_CTX.get("latent_shapes")
        if not shapes or len(shapes) <= 1:
            return None
        try:
            # Audio+video latents are packed into [B, 1, total]; the video one is the 5D entry.
            x0 = next((p for p in comfy.utils.unpack_latents(x0, shapes) if p.ndim == 5), None)
        except Exception:
            return None
        if x0 is None:
            return None
    num_keyframes = _SAMPLE_CTX.get("num_keyframes") or 0
    if num_keyframes > 0:
        # Strip I2V guide frames appended at the end of the latent before previewing.
        x0 = x0[:, :, :-num_keyframes]
    return x0 if x0.size(2) > 0 else None


class _LTXTAEPreviewer:
    """Cycles through LTX video latent frames during sampling, decoding with taeltx."""

    def __init__(self, taeltx, rate=8):
        self.first_preview = True
        self.last_time = 0.0
        self.c_index = 0
        self.rate = rate
        self.taeltx = taeltx

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
                'MXD_live_preview_start',
                {'length': num_images, 'rate': self.rate, 'id': _serv.last_node_id},
            )
            self.last_time = new_time + 1.0 / self.rate
        if self.c_index + num_previews > num_images:
            frames = x0.roll(-self.c_index, 0)[:num_previews]
        else:
            frames = x0[self.c_index:self.c_index + num_previews]
        Thread(target=self._send_frames, args=(frames, self.c_index, num_images)).run()
        self.c_index = (self.c_index + num_previews) % num_images
        return None

    def _send_frames(self, image_tensor, ind, leng):
        max_size, min_size = 512, 256
        image_tensor = self._decode(image_tensor)
        if image_tensor.size(1) < min_size or image_tensor.size(2) < min_size:
            image_tensor = F.interpolate(
                image_tensor.movedim(-1, 0), scale_factor=4, mode='nearest'
            ).movedim(0, -1)
        if image_tensor.size(1) > max_size or image_tensor.size(2) > max_size:
            t = image_tensor.movedim(-1, 0)
            if t.size(2) < t.size(3):
                h = (max_size * t.size(2)) // t.size(3)
                t = F.interpolate(t, (h, max_size), mode='nearest')
            else:
                w = (max_size * t.size(3)) // t.size(2)
                t = F.interpolate(t, (max_size, w), mode='nearest')
            image_tensor = t.movedim(0, -1)
        previews = image_tensor.clamp(0, 1).mul(0xFF).to(device="cpu", dtype=torch.uint8)
        node_id = _serv.last_node_id
        for preview in previews:
            img = Image.fromarray(preview.numpy())
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=90)
            data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            _serv.send_sync('MXD_live_preview_frame', {'id': node_id, 'index': ind, 'length': leng, 'data': data_url})
            # taeltx expands the 8× temporal compression on decode
            ind = (ind + 1) % ((leng - 1) * 8 + 1)

    def _decode(self, x0):
        dev = comfy.model_management.get_torch_device()
        dtype = self.taeltx.first_stage_model.decoder[1].weight.dtype
        x0 = x0.unsqueeze(0).to(dtype=dtype, device=dev)
        return self.taeltx.first_stage_model.decode(x0)[0].permute(1, 2, 3, 0)


def _save_final_ltx_preview(node_id, previewer, x0_v, rate):
    """Decode the full final clip with taeltx and save it to output/live_previews."""
    try:
        frames = x0_v.movedim(2, 1)
        frames = frames.reshape((-1,) + frames.shape[-3:])
        frames = previewer._decode(frames).clamp(0, 1).to(device="cpu", dtype=torch.float32)
        saved = save_preview(node_id, frames, rate, log_prefix="[MXD LTX preview]")
        notify_saved(_serv, node_id, saved)
    except Exception as e:
        print(f"[MXD LTX preview] Failed to save live preview: {e}")


########################################################################################################################
# Automatic path — used by system/live_preview.py's get_previewer hook
class _LTXAutoPreviewer(_LTXTAEPreviewer):
    """taeltx previewer driven by core's own preview callback.

    Unlike the wrapper, nothing here gets to unpack the latent or move the
    decoder around for us, so both are handled per call.
    """

    def decode_latent_to_preview_image(self, preview_format, x0):
        x0 = _video_latent_from_x0(x0)
        if x0 is None:
            return None
        return super().decode_latent_to_preview_image(preview_format, x0)

    def _decode(self, x0):
        # No wrapper staged the decoder onto the sampling device for us.
        self.taeltx.first_stage_model.to(comfy.model_management.get_torch_device())
        return super()._decode(x0)

    def offload(self):
        try:
            self.taeltx.first_stage_model.to(comfy.model_management.unet_offload_device())
        except Exception:
            pass


def is_ltx_video_format(latent_format):
    """True for the LTX formats core ships no previewer for (LTX 2.3 / LTXAV)."""
    ltxav = getattr(comfy.latent_formats, "LTXAV", None)
    if ltxav is not None and isinstance(latent_format, ltxav):
        return True
    return (
        "LTX" in type(latent_format).__name__
        and getattr(latent_format, "latent_rgb_factors", None) is None
    )


def make_auto_previewer(latent_format, rate=8):
    """Previewer for an LTX latent format core can't preview, or None if this isn't one."""
    try:
        if not is_ltx_video_format(latent_format):
            return None
    except Exception:
        return None
    taeltx = _load_taeltx()
    if taeltx is None:
        print("[MXD LTX preview] taeltx model not found in vae / vae_approx — skipping preview.")
        return None
    _install_sample_context_hook()
    return _LTXAutoPreviewer(taeltx, rate=rate)


def save_auto_preview(node_id, previewer, x0, rate):
    """Save the finished clip to output/live_previews and release the decoder."""
    try:
        x0_v = _video_latent_from_x0(x0)
        if x0_v is not None:
            _save_final_ltx_preview(node_id, previewer, x0_v, rate)
    finally:
        previewer.offload()

