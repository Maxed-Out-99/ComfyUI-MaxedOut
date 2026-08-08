"""Image scaling to model-safe megapixel targets + resolution matchers.

Registered nodes:
  Image Scale To Total Pixels (SDXL Safe)       Scale SDXL Image MXD
  Flux Image Scale To Total Pixels (Flux Safe)  Scale Flux Image MXD
  FluxResolutionMatcher                          Flux Resolution Matcher MXD
  SDXLResolutionMatcher                          SDXL Resolution Matcher MXD
  ResolutionSelectorMXD                          Resolution Selector MXD
"""
from __future__ import annotations
import math, comfy, comfy.utils, torch
from .latents import SdxlEmptyLatentImage, ResolutionSelectorEmptyLatentImage

########################################################################################################################
# Image Scale To Total Pixels (SDXL Safe)
class SDXLImageScaleToTotalPixelsSafe:
    DESCRIPTION = """Scale to a target megapixel count and keep aspect ratio. Skips SDXL-safe sizes."""
    upscale_methods = ["bilinear", "bicubic", "lanczos", "nearest-exact", "area"]

    # SDXL-safe resolutions (width, height) – store one orientation only,
    # the code will check both (w, h) and (h, w)
    SDXL_SAFE_RESOLUTIONS = [
        (1024, 1024),
        (1152, 896),
        (1216, 832),
        (1344, 768),
        (1536, 640),
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (cls.upscale_methods, {"default": "bilinear"}),
                "total_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 128.0,
                        "step": 0.01,
                        "tooltip": "Set the total megapixels (e.g., 1.0 = 1 MP)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "MXD/Upscaling"

    def upscale(self, image, upscale_method, total_megapixels):
        if upscale_method in ["nearest-exact", "area"]:
            raise Exception(
                f"❌ '{upscale_method}' gives poor results.\n\n"
                f"👉 Go to the Scale SDXL Image MXD node and switch to another like 'lanczos'.\n\n"
                f"Node may be hidden behind KSampler."
            )

        b, h, w, c = image.shape

        # Skip scaling if the image already matches an SDXL-safe resolution
        if (w, h) in self.SDXL_SAFE_RESOLUTIONS or (h, w) in self.SDXL_SAFE_RESOLUTIONS:
            return (image,)

        # ComfyUI-native megapixel math
        samples = image.movedim(-1, 1)
        orig_h, orig_w = samples.shape[2], samples.shape[3]

        target_pixels = int(round(total_megapixels * 1024 * 1024))
        scale_by = math.sqrt(target_pixels / (orig_w * orig_h))

        new_w = max(1, round(orig_w * scale_by))
        new_h = max(1, round(orig_h * scale_by))

        scaled = comfy.utils.common_upscale(samples, new_w, new_h, upscale_method, "disabled")
        scaled = scaled.movedim(1, -1)
        return (scaled,)

########################################################################################################################
# Flux Image Scale To Total Pixels (Flux Safe)
class FluxImageScaleToTotalPixelsSafe:
    DESCRIPTION = """Scale to a target megapixel count and keep aspect ratio. Skips Flux-safe sizes."""
    upscale_methods = ["bilinear", "bicubic", "lanczos", "nearest-exact", "area"]

    # Flux-safe resolutions (width, height) – stored in one orientation only
    FLUX_SAFE_RESOLUTIONS = [
        (1408, 1408),
        (1728, 1152),
        (1664, 1216),
        (1920, 1088),
        (2176, 960),
        (1024, 1024),
        (1216, 832),
        (1152, 896),
        (1344, 768),
        (1536, 640),
        (320, 320),
        (384, 256),
        (448, 320),
        (448, 256),
        (576, 256),
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (cls.upscale_methods, {"default": "bilinear"}),
                "total_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 128.0,
                        "step": 0.01,
                        "tooltip": "Set the total megapixels (e.g., 1.0 = 1 MP)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "MXD/Upscaling"

    def upscale(self, image, upscale_method, total_megapixels):
        if upscale_method in ["nearest-exact", "area"]:
            raise Exception(
                f"❌ '{upscale_method}' gives poor results.\n\n"
                f"👉 Go to the Scale Flux Image  MXD node and switch to another like 'lanczos'.\n\n"
                f"Node may be hidden behind KSampler."
            )

        b, h, w, c = image.shape

        # Skip scaling if image matches any Flux-safe resolution
        if (w, h) in self.FLUX_SAFE_RESOLUTIONS or (h, w) in self.FLUX_SAFE_RESOLUTIONS:
            return (image,)

        samples = image.movedim(-1, 1)
        orig_h, orig_w = samples.shape[2], samples.shape[3]

        target_pixels = int(round(total_megapixels * 1024 * 1024))
        scale_by = math.sqrt(target_pixels / (orig_w * orig_h))

        new_w = max(1, round(orig_w * scale_by))
        new_h = max(1, round(orig_h * scale_by))

        scaled = comfy.utils.common_upscale(samples, new_w, new_h, upscale_method, "disabled")
        scaled = scaled.movedim(1, -1)
        return (scaled,)

########################################################################################################################
class FluxResolutionMatcher:
    DESCRIPTION = """Match the closest Flux resolution and orientation for the input image."""
    CATEGORY = "MXD/Latent"
    FUNCTION = "match_resolution"
    RETURN_NAMES = ("resolution", "vertical")

    # Full set kept for compatibility (enum list must match FluxEmptyLatentImage)
    RESOLUTIONS = {
        "— High Resolutions —": None,
        "Square (1:1) 1408x1408": (1408, 1408),
        "Standard (4:3) 1664x1216": (1664, 1216),
        "Landscape (3:2) 1728x1152": (1728, 1152),
        "Widescreen (16:9) 1920x1088": (1920, 1088),
        "Ultrawide (21:9) 2176x960": (2176, 960),

        "— Standard Resolutions —": None,
        "Square (1:1) 1024x1024": (1024, 1024),
        "Standard (4:3) 1152x896": (1152, 896),
        "Landscape (3:2) 1216x832": (1216, 832),
        "Widescreen (16:9) 1344x768": (1344, 768),
        "Ultrawide (21:9) 1536x640": (1536, 640),

        "— Low Resolutions —": None,
        "Square (1:1) 320x320": (320, 320),
        "Standard (4:3) 448x320": (448, 320),
        "Landscape (3:2) 384x256": (384, 256),
        "Widescreen (16:9) 448x256": (448, 256),
        "Ultrawide (21:9) 576x256": (576, 256),
    }

    # Keep same enum type so it connects to FluxEmptyLatentImage
    RETURN_TYPES = (list(RESOLUTIONS.keys()), "BOOLEAN")

    # Precompute aspect ratio groups (only for standard resolutions)
    ASPECT_RATIO_GROUPS = {}
    for res_str, dims in RESOLUTIONS.items():
        if dims is None:
            continue
        # Skip high and low groups for logic
        if "High" in res_str or "Low" in res_str:
            continue
        group_name = " ".join(res_str.split(' ')[:-1])
        if group_name not in ASPECT_RATIO_GROUPS:
            w, h = dims
            ratio = w / h
            ASPECT_RATIO_GROUPS[group_name] = {'ratio': ratio, 'resolutions': []}
        ASPECT_RATIO_GROUPS[group_name]['resolutions'].append(res_str)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    def match_resolution(self, image: torch.Tensor):
        if image.dim() < 4 or image.shape[1] < 1 or image.shape[2] < 1:
            print("Warning: Invalid image tensor received. Falling back to default resolution.")
            return ("Square (1:1) 1024x1024", False)

        _batch, height, width, _channels = image.shape
        is_vertical = height > width
        img_aspect_ratio = (height / width) if is_vertical else (width / height)
        img_area = height * width

        best_ar_group_name = min(
            self.ASPECT_RATIO_GROUPS.keys(),
            key=lambda name: abs(img_aspect_ratio - self.ASPECT_RATIO_GROUPS[name]['ratio'])
        )

        candidate_res_strings = self.ASPECT_RATIO_GROUPS[best_ar_group_name]['resolutions']

        best_res_string = min(
            candidate_res_strings,
            key=lambda res_str: abs(img_area - (self.RESOLUTIONS[res_str][0] * self.RESOLUTIONS[res_str][1]))
        )

        return (best_res_string, is_vertical)
########################################################################################################################

class SDXLResolutionMatcher:
    DESCRIPTION = """Match the closest SDXL resolution and orientation for the input image."""
    CATEGORY = "MXD/Latent"
    FUNCTION = "match_resolution"
    RETURN_NAMES = ("resolution", "vertical")

    # Use the exact same enum list as SdxlEmptyLatentImage
    RESOLUTIONS = SdxlEmptyLatentImage.RESOLUTIONS

    RETURN_TYPES = (list(RESOLUTIONS.keys()), "BOOLEAN")

    ASPECT_RATIO_GROUPS = {}
    for res_str, dims in RESOLUTIONS.items():
        if dims is None:
            continue
        group_name = " ".join(res_str.split(" ")[:-1])
        if group_name not in ASPECT_RATIO_GROUPS:
            w, h = dims
            ratio = w / h
            ASPECT_RATIO_GROUPS[group_name] = {"ratio": ratio, "resolutions": []}
        ASPECT_RATIO_GROUPS[group_name]["resolutions"].append(res_str)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    def match_resolution(self, image: torch.Tensor):
        if image.dim() < 4 or image.shape[1] < 1 or image.shape[2] < 1:
            print("Warning: Invalid image tensor received. Falling back to default resolution.")
            return ("Square (1:1) 1024x1024", False)

        _batch, height, width, _channels = image.shape
        is_vertical = height > width
        img_aspect_ratio = (height / width) if is_vertical else (width / height)
        img_area = height * width

        best_ar_group_name = min(
            self.ASPECT_RATIO_GROUPS.keys(),
            key=lambda name: abs(img_aspect_ratio - self.ASPECT_RATIO_GROUPS[name]["ratio"])
        )

        candidate_res_strings = self.ASPECT_RATIO_GROUPS[best_ar_group_name]["resolutions"]

        best_res_string = min(
            candidate_res_strings,
            key=lambda res_str: abs(img_area - (self.RESOLUTIONS[res_str][0] * self.RESOLUTIONS[res_str][1]))
        )

        return (best_res_string, is_vertical)
########################################################################################################################
class ResolutionSelectorMXD:
    DESCRIPTION = """Calculate width and height from aspect ratio and megapixel target, with a vertical toggle. Same math as core's Resolution Selector node, minus the separate portrait/landscape entries."""
    CATEGORY = "MXD/Latent"
    FUNCTION = "calculate"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    OUTPUT_TOOLTIPS = ("Calculated width in pixels.", "Calculated height in pixels.")

    # Reuse the same ratio set as the empty-latent version so both nodes agree.
    ASPECT_RATIOS = ResolutionSelectorEmptyLatentImage.ASPECT_RATIOS

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "aspect_ratio": (
                    list(cls.ASPECT_RATIOS.keys()),
                    {"default": "Square (1:1)"}
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": "Target total megapixels. 1.0 MP ≈ 1024x1024 for square."
                    }
                ),
                "vertical": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "multiple": (
                    "INT",
                    {
                        "default": 8,
                        "min": 8,
                        "max": 128,
                        "step": 4,
                        "tooltip": "Round the calculated resolution to the nearest multiple of this value.",
                        "advanced": True
                    }
                ),
            }
        }

    def calculate(self, aspect_ratio, megapixels, vertical, multiple=8) -> tuple:
        w_ratio, h_ratio = self.ASPECT_RATIOS[aspect_ratio]
        total_pixels = megapixels * 1024 * 1024
        scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
        width = round(w_ratio * scale / multiple) * multiple
        height = round(h_ratio * scale / multiple) * multiple

        if vertical:
            width, height = height, width

        return (width, height)
########################################################################################################################

NODE_CLASS_MAPPINGS = {
    "Image Scale To Total Pixels (SDXL Safe)": SDXLImageScaleToTotalPixelsSafe,
    "Flux Image Scale To Total Pixels (Flux Safe)": FluxImageScaleToTotalPixelsSafe,
    "FluxResolutionMatcher": FluxResolutionMatcher,
    "SDXLResolutionMatcher": SDXLResolutionMatcher,
    "ResolutionSelectorMXD": ResolutionSelectorMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Scale To Total Pixels (SDXL Safe)": "Scale SDXL Image MXD",
    "Flux Image Scale To Total Pixels (Flux Safe)": "Scale Flux Image MXD",
    "FluxResolutionMatcher": "Flux Resolution Matcher MXD",
    "SDXLResolutionMatcher": "SDXL Resolution Matcher MXD",
    "ResolutionSelectorMXD": "Resolution Selector MXD",
}
