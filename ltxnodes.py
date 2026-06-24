from __future__ import annotations
import os
import re
import struct
import time
import urllib.error
import urllib.request
from io import BytesIO
from PIL import Image
from threading import Lock, Thread

import torch
import torch.nn.functional as F

import comfy
import comfy.model_management
import comfy.patcher_extension
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
import server

_serv = server.PromptServer.instance


########################################################################################################################
# LTX Video Empty Latent Image
class LTXVideoEmptyLatentMXD:
    DESCRIPTION = "Create an LTX Video empty latent batch from connected width/height and frame count."
    TITLE = "LTX Empty Latent Video MXD"
    CATEGORY = "MXD/Latent"

    # All dimensions must be multiples of 32 (LTX 32× spatial compression).
    # Lengths must be 8n+1 for LTX's 8× temporal compression.
    RESOLUTIONS = {
        "16:9 Landscape": None,
        "16:9  512×288":  (512,  288),
        "16:9  768×448":  (768,  448),
        "16:9  832×480":  (832,  480),
        "16:9 1024×576":  (1024, 576),
        "16:9 1280×736":  (1280, 736),

        "9:16 Portrait": None,
        "9:16  288×512":  (288,  512),
        "9:16  448×768":  (448,  768),
        "9:16  480×832":  (480,  832),
        "9:16  576×1024": (576, 1024),

        "4:3 Standard": None,
        "4:3  512×384":  (512,  384),
        "4:3  768×576":  (768,  576),

        "1:1 Square": None,
        "1:1  512×512":  (512,  512),
        "1:1  768×768":  (768,  768),
    }

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "length": (
                    "INT",
                    {
                        "default": 97,
                        "min": 9,
                        "max": 1025,
                        "step": 8,
                        "tooltip": "Number of frames. Must be 8n+1 (e.g. 25, 49, 73, 97, 121, 201).",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "Number of latent videos in the batch.",
                    },
                ),
            },
            "optional": {
                "width": ("INT", {
                    "default": 640, "min": 32, "max": 8192, "step": 32,
                    "tooltip": "Stage 1 width. Connect the LTX Image Scaler stage1_width output for I2V workflows.",
                }),
                "height": ("INT", {
                    "default": 384, "min": 32, "max": 8192, "step": 32,
                    "tooltip": "Stage 1 height. Connect the LTX Image Scaler stage1_height output for I2V workflows.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "length")
    FUNCTION = "generate"

    def generate(self, length, batch_size=1, width=640, height=384):
        # LTX latent: 128 channels, 32× spatial compression, 8× temporal compression
        width = max(32, int(width) // 32 * 32)
        height = max(32, int(height) // 32 * 32)
        length = max(9, 1 + 8 * round((int(length) - 1) / 8))

        t = ((length - 1) // 8) + 1
        h = height // 32
        w = width  // 32
        latent = torch.zeros([batch_size, 128, t, h, w], device=self.device)
        return ({"samples": latent}, length)


########################################################################################################################
# Shared noise helper — equivalent to ComfyUI RandomNoise
class _LTXNoise:
    def __init__(self, seed: int):
        self.seed = seed

    def generate_noise(self, latent: dict) -> torch.Tensor:
        samples = latent["samples"]
        batch_inds = latent.get("batch_index", None)
        return comfy.sample.prepare_noise(samples, self.seed, batch_inds)


########################################################################################################################
# LTX video preview (taeltx TAE decode)
#
# Core ComfyUI has no preview for the LTXAV format used by LTX 2.3, and the
# latent2rgb approximation looks awful for video. This installs a previewer that
# decodes latent frames with the tiny "taeltx" autoencoder for accurate previews.
# The taeltx model is auto-discovered in the vae / vae_approx model folders. If
# it isn't found, it is downloaded to the configured vae model folder.
#
# TAE decode path borrowed from kjnodes / VideoHelperSuite.

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
                'VHS_latentpreview',
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
        for preview in previews:
            img = Image.fromarray(preview.numpy())
            buf = BytesIO()
            buf.write((1).to_bytes(length=4, byteorder='big') * 2)
            buf.write(ind.to_bytes(length=4, byteorder='big'))
            buf.write(struct.pack('16p', _serv.last_node_id.encode('ascii')))
            img.save(buf, format="JPEG", quality=95, compress_level=1)
            _serv.send_sync(server.BinaryEventTypes.PREVIEW_IMAGE, buf.getvalue(), _serv.client_id)
            # taeltx expands the 8× temporal compression on decode
            ind = (ind + 1) % ((leng - 1) * 8 + 1)

    def _decode(self, x0):
        dev = comfy.model_management.get_torch_device()
        dtype = self.taeltx.first_stage_model.decoder[1].weight.dtype
        x0 = x0.unsqueeze(0).to(dtype=dtype, device=dev)
        return self.taeltx.first_stage_model.decode(x0)[0].permute(1, 2, 3, 0)


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
# LTX KSampler — Stage 1 (T2V / I2V generation at base resolution)
class LTXKSamplerMXD:
    DESCRIPTION = (
        "LTX-Video Stage 1 sampler for the distilled workflow. Use Distilled 8 Step "
        "for the trained schedule, or Custom Sigmas when intentionally testing a "
        "manual schedule."
    )
    TITLE = "LTX Stage 1 Sampler MXD"
    CATEGORY = "MXD/Sampling"

    MODES = ["Distilled 8 Step", "Custom Sigmas"]
    _DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975,
                         0.909375, 0.725, 0.421875, 0.0]
    _CUSTOM_SIGMAS_DEFAULT = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":        ("MODEL",),
                "positive":     ("CONDITIONING",),
                "negative":     ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "mode":         (cls.MODES, {"default": "Distilled 8 Step"}),
                "seed":         ("INT",   {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "cfg":          ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (
                    ["euler_ancestral_cfg_pp", "euler_cfg_pp", "euler"],
                    {"default": "euler_ancestral_cfg_pp"},
                ),
                "custom_sigmas": (
                    "STRING",
                    {
                        "default": cls._CUSTOM_SIGMAS_DEFAULT,
                        "multiline": True,
                        "tooltip": "Only used when mode is Custom Sigmas. Enter comma, space, or newline separated sigma values.",
                    },
                ),
                "ltx_preview": ("BOOLEAN", {"default": True, "tooltip": "Show LTX video previews during sampling. Downloads the taeltx VAE to your vae model folder if it is missing."}),
            },
        }

    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("latent",)
    FUNCTION      = "sample"
    OUTPUT_NODE   = False

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        mode="Distilled 8 Step",
        seed=0,
        cfg=2.0,
        sampler_name="euler_ancestral_cfg_pp",
        custom_sigmas=_CUSTOM_SIGMAS_DEFAULT,
        ltx_preview=True,
    ):
        sigmas = _select_sigmas(
            mode,
            {
                "Distilled 8 Step": self._DISTILLED_SIGMAS,
            },
            custom_sigmas,
            "LTX Stage 1 Sampler MXD",
        )
        return _run_sampling(model, positive, negative, latent_image, seed, cfg, sampler_name, sigmas, ltx_preview)


########################################################################################################################
# LTX KSampler 2 — Stage 2 (refinement at 2× resolution with distilled LoRA)
class LTXKSampler2MXD:
    DESCRIPTION = (
        "LTX-Video Stage 2 refiner for the distilled workflow. Official Refine "
        "matches the Lightricks 2.3 two-stage example (start sigma 0.85). "
        "Custom Sigmas is for manual testing."
    )
    TITLE = "LTX Stage 2 Refiner MXD"
    CATEGORY = "MXD/Sampling"

    # Exact stage-2 refine schedule from the official Lightricks 2.3 two-stage
    # workflow (LTX-2.3_T2V_I2V_Two_Stage_Distilled.json, euler_cfg_pp, cfg 1).
    # Only the starting sigma (denoise strength) is meant to vary; use Custom
    # Sigmas for that.
    MODES = ["Official Refine", "Custom Sigmas"]
    _OFFICIAL_REFINE_SIGMAS = [0.85, 0.725, 0.4219, 0.0]
    _CUSTOM_SIGMAS_DEFAULT = "0.85, 0.725, 0.4219, 0.0"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":        ("MODEL",),
                "positive":     ("CONDITIONING",),
                "negative":     ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "mode":         (cls.MODES, {"default": "Official Refine"}),
                "seed":         ("INT",   {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "cfg":          ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (
                    ["euler_cfg_pp", "euler_ancestral_cfg_pp", "euler"],
                    {"default": "euler_cfg_pp"},
                ),
                "custom_sigmas": (
                    "STRING",
                    {
                        "default": cls._CUSTOM_SIGMAS_DEFAULT,
                        "multiline": True,
                        "tooltip": "Only used when mode is Custom Sigmas. Enter comma, space, or newline separated sigma values.",
                    },
                ),
                "ltx_preview": ("BOOLEAN", {"default": True, "tooltip": "Show LTX video previews during sampling. Downloads the taeltx VAE to your vae model folder if it is missing."}),
            },
        }

    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("latent",)
    FUNCTION      = "sample"
    OUTPUT_NODE   = False

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        mode="Official Refine",
        seed=0,
        cfg=1.0,
        sampler_name="euler_cfg_pp",
        custom_sigmas=_CUSTOM_SIGMAS_DEFAULT,
        ltx_preview=True,
    ):
        sigmas = _select_sigmas(
            mode,
            {
                "Official Refine": self._OFFICIAL_REFINE_SIGMAS,
            },
            custom_sigmas,
            "LTX Stage 2 Refiner MXD",
        )
        return _run_sampling(model, positive, negative, latent_image, seed, cfg, sampler_name, sigmas, ltx_preview)


########################################################################################################################
# Sigma schedule helpers
_SIGMA_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def _select_sigmas(mode, presets, custom_sigmas, node_name):
    if mode == "Custom Sigmas":
        values = _parse_custom_sigmas(custom_sigmas, node_name)
    else:
        try:
            values = presets[mode]
        except KeyError as exc:
            allowed = ", ".join([*presets.keys(), "Custom Sigmas"])
            raise ValueError(f"{node_name}: unknown mode '{mode}'. Expected one of: {allowed}.") from exc

    return torch.tensor(values, dtype=torch.float32)


def _parse_custom_sigmas(custom_sigmas, node_name):
    text = str(custom_sigmas or "")
    values = [float(match.group(0)) for match in _SIGMA_RE.finditer(text)]

    if len(values) < 2:
        raise ValueError(f"{node_name}: Custom Sigmas needs at least two sigma values, ending with 0.0.")

    for index, (left, right) in enumerate(zip(values, values[1:]), start=1):
        if right > left:
            raise ValueError(
                f"{node_name}: Custom Sigmas must be in descending order. "
                f"Value {index + 1} ({right}) is greater than value {index} ({left})."
            )

    if abs(values[-1]) > 1e-8:
        raise ValueError(f"{node_name}: Custom Sigmas must end with 0.0.")

    return values


########################################################################################################################
# Shared sampling logic
def _run_sampling(model, positive, negative, latent_image, seed, cfg, sampler_name, sigmas, ltx_preview=False):
    taeltx = _load_taeltx() if ltx_preview else None
    if ltx_preview and taeltx is None:
        print("[MXD LTX preview] taeltx model not found in vae / vae_approx — skipping preview.")

    if taeltx is not None:
        model = model.clone()
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "ltx_mxd_preview",
            _LTXPreviewWrapper(taeltx),
        )

    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)

    sampler = comfy.samplers.sampler_object(sampler_name)

    latent        = latent_image.copy()
    latent_samples = latent["samples"]

    try:
        latent_samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher, latent_samples,
            latent.get("downscale_ratio_spacial", None),
        )
    except AttributeError:
        pass

    latent["samples"] = latent_samples
    noise_mask = latent.get("noise_mask", None)

    noise = _LTXNoise(seed)

    if taeltx is not None:
        # The preview wrapper owns the progress bar / callback.
        callback = None
    else:
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = guider.sample(
        noise.generate_noise(latent),
        latent_samples,
        sampler,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return (out,)


########################################################################################################################
NODE_CLASS_MAPPINGS = {
    "LTXVideoEmptyLatent_MXD":  LTXVideoEmptyLatentMXD,
    "LTXKSampler_MXD":          LTXKSamplerMXD,
    "LTXKSampler2_MXD":         LTXKSampler2MXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVideoEmptyLatent_MXD":  "LTX Empty Latent Video MXD",
    "LTXKSampler_MXD":          "LTX Stage 1 Sampler MXD",
    "LTXKSampler2_MXD":         "LTX Stage 2 Refiner MXD",
}
