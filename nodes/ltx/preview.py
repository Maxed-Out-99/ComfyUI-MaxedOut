"""LTX live preview via the tiny taeltx autoencoder.

Registered node:
  LTXPreview_MXD  LTX Preview MXD (attach the previewer to ANY sampler's model)

Core ComfyUI has no preview for the LTXAV format used by LTX 2.3, and the
latent2rgb approximation looks awful for video. This installs a previewer that
decodes latent frames with the tiny "taeltx" autoencoder for accurate previews.
The taeltx model is auto-discovered in the vae / vae_approx model folders. If
it isn't found, it is downloaded to the configured vae model folder.

Sends MXD_live_preview_start / MXD_live_preview_frame / MXD_live_preview_saved
websocket events consumed by web/live_preview_panel_mxd.js. Final clips are
saved to <output>/live_previews.

TAE decode path borrowed from kjnodes / VideoHelperSuite.
"""
from __future__ import annotations
import os
import base64
import time
import urllib.error
import urllib.request
from fractions import Fraction
from io import BytesIO
from PIL import Image
from threading import Lock, Thread

import torch
import torch.nn.functional as F

import comfy
import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import server
import folder_paths as _folder_paths
from comfy_api.latest import VideoFromComponents, VideoComponents

_serv = server.PromptServer.instance

_TAELTX_FILENAME = "taeltx2_3.safetensors"
_TAELTX_URL = "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors?download=true"
_TAELTX_DOWNLOAD_LOCK = Lock()


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

    try:
        taeltx = VAE(comfy.utils.load_torch_file(path))
        taeltx.first_stage_model.show_progress_bar = False
    except Exception as exc:
        print(f"[MXD LTX preview] failed to load taeltx ({path}): {exc}")
        return None
    return taeltx


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
    """Decode the full final clip with taeltx and save it as an mp4 to output/live_previews."""
    try:
        frames = x0_v.movedim(2, 1)
        frames = frames.reshape((-1,) + frames.shape[-3:])
        frames = previewer._decode(frames).clamp(0, 1).to(device="cpu", dtype=torch.float32)
        if frames.ndim != 4 or frames.size(0) == 0:
            return
        out_dir = os.path.join(_folder_paths.get_output_directory(), "live_previews")
        os.makedirs(out_dir, exist_ok=True)
        safe_id = str(node_id).replace(":", "_").replace("/", "_")
        filename = f"{safe_id}_{int(time.time())}.mp4"
        path = os.path.join(out_dir, filename)
        video = VideoFromComponents(VideoComponents(images=frames, frame_rate=Fraction(max(1, round(rate)))))
        video.save_to(path)
        print(f"[MXD LTX preview] Saved live preview to {path}")
        _serv.send_sync("MXD_live_preview_saved", {
            "node_id": node_id, "filename": filename, "subfolder": "live_previews", "type": "output",
        })
    except Exception as e:
        print(f"[MXD LTX preview] Failed to save live preview: {e}")


class _LTXPreviewWrapper:
    """OUTER_SAMPLE wrapper that installs the taeltx video previewer during sampling."""

    def __init__(self, taeltx):
        self.taeltx = taeltx

    def __call__(self, executor, noise, latent_image, sampler, sigmas,
                 denoise_mask, callback, disable_pbar, seed, latent_shapes):
        guider = executor.class_obj
        device = comfy.model_management.get_torch_device()
        self.taeltx.first_stage_model.to(device)

        previewer = _LTXTAEPreviewer(self.taeltx, rate=8)
        pbar = comfy.utils.ProgressBar(len(sigmas) - 1)
        node_id = _serv.last_node_id

        # Strip I2V guide frames appended at the end of the latent before previewing.
        num_keyframes = 0
        if 'positive' in guider.conds and guider.conds['positive']:
            kf = guider.conds['positive'][0].get('keyframe_idxs')
            if kf is not None:
                num_keyframes = len(torch.unique(kf[0, 0, :, 0]))

        def ltx_callback(step, x0, x, total_steps):
            x0_v = x0
            if x0_v is not None and len(latent_shapes) > 1:
                # Audio+video latents are packed into [B, 1, total]; unpack and
                # take the video tensor (the 5D one). Audio is a lower-rank entry.
                x0_v = next(
                    (p for p in comfy.utils.unpack_latents(x0, latent_shapes) if p.ndim == 5),
                    None,
                )
            if x0_v is not None and x0_v.ndim == 5 and num_keyframes > 0:
                x0_v = x0_v[:, :, :-num_keyframes]
            preview = (
                previewer.decode_latent_to_preview_image("JPEG", x0_v)
                if x0_v is not None and x0_v.ndim == 5 else None
            )
            pbar.update_absolute(step + 1, total_steps, preview)
            if step + 1 >= total_steps and x0_v is not None and x0_v.ndim == 5:
                _save_final_ltx_preview(node_id, previewer, x0_v, previewer.rate)
            if callback is not None:
                callback(step, x0, x, total_steps)

        try:
            return executor(
                noise, latent_image, sampler, sigmas, denoise_mask,
                ltx_callback, disable_pbar, seed, latent_shapes=latent_shapes,
            )
        finally:
            self.taeltx.first_stage_model.to(comfy.model_management.unet_offload_device())


########################################################################################################################
# LTX Preview — attach the taeltx previewer to any model
class LTXPreviewMXD:
    DESCRIPTION = (
        "Enables taeltx video previews during sampling for ANY sampler node "
        "(SamplerCustomAdvanced, KSampler, etc.), not just the MXD LTX samplers. "
        "LTX 2.3 (LTXAV) ships no built-in preview decoder, so core ComfyUI shows "
        "nothing; this attaches a wrapper to the model that decodes latent frames "
        "with the tiny taeltx autoencoder. Wire it between your model loader and "
        "the sampler's model input. Downloads taeltx to your vae folder if missing."
    )
    TITLE = "LTX Preview MXD"
    CATEGORY = "MXD/Sampling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":   ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Turn taeltx previews on/off without unwiring the node."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION     = "apply"
    OUTPUT_NODE  = False

    def apply(self, model, enabled=True):
        if not enabled:
            return (model,)
        taeltx = _load_taeltx()
        if taeltx is None:
            print("[MXD LTX preview] taeltx model not found in vae / vae_approx — skipping preview.")
            return (model,)
        model = model.clone()
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "ltx_mxd_preview",
            _LTXPreviewWrapper(taeltx),
        )
        return (model,)


NODE_CLASS_MAPPINGS = {
    "LTXPreview_MXD": LTXPreviewMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXPreview_MXD": "LTX Preview MXD",
}
