"""LTX two-stage distilled samplers.

Registered nodes:
  LTXKSampler_MXD   LTX Stage 1 Sampler MXD (distilled 8-step schedule)
  LTXKSampler2_MXD  LTX Stage 2 Refiner MXD (official refine, start sigma 0.85)

Sigma schedules come from the official Lightricks LTX-2.3 two-stage distilled
workflow (LTX-2.3_T2V_I2V_Two_Stage_Distilled.json). Custom Sigmas mode accepts
a manual descending schedule ending in 0.0 for experimentation.

These nodes used to carry an `ltx_preview` toggle that attached a taeltx
previewer. Live previews now live in the standalone Live-Preview-MXD pack,
which previews any sampler automatically with nothing to wire or toggle.
"""
from __future__ import annotations
import re

import torch

import comfy
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview


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
    ):
        sigmas = _select_sigmas(
            mode,
            {
                "Distilled 8 Step": self._DISTILLED_SIGMAS,
            },
            custom_sigmas,
            "LTX Stage 1 Sampler MXD",
        )
        return _run_sampling(model, positive, negative, latent_image, seed, cfg, sampler_name, sigmas)


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
    ):
        sigmas = _select_sigmas(
            mode,
            {
                "Official Refine": self._OFFICIAL_REFINE_SIGMAS,
            },
            custom_sigmas,
            "LTX Stage 2 Refiner MXD",
        )
        return _run_sampling(model, positive, negative, latent_image, seed, cfg, sampler_name, sigmas)


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
def _run_sampling(model, positive, negative, latent_image, seed, cfg, sampler_name, sigmas):
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


NODE_CLASS_MAPPINGS = {
    "LTXKSampler_MXD":  LTXKSamplerMXD,
    "LTXKSampler2_MXD": LTXKSampler2MXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXKSampler_MXD":  "LTX Stage 1 Sampler MXD",
    "LTXKSampler2_MXD": "LTX Stage 2 Refiner MXD",
}
