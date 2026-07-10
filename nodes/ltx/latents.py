"""LTX Video latent sizing: empty latent generator + two-stage image scaler.

Registered nodes:
  LTXVideoEmptyLatent_MXD  LTX Empty Latent Video MXD
  LTX_Image_Scaler_MXD     LTX Video Image Scaler MXD

Official LTX-2.3 rules (Lightricks model card + example workflows):
  - Width & height must be divisible by 32; frame count must be 8n+1.
  - The distilled two-stage workflow generates Stage 1 low-res, then the
    ltx-2.3-spatial-upscaler-x2 doubles it (exactly 2x) for Stage 2.
  - The one published two-stage resolution is Stage 1 960x544 -> 1920x1088.

Tiers below are FINAL (Stage 2) sizes; Stage 1 is exactly half. Finals are
kept /64 so Stage 1 stays /32 (the latent constraint). Only the 1080p 16:9
row is officially published by Lightricks; the portrait/square rows and the
720p/576p tiers are /32-aligned siblings at the same pixel budget.

Buckets (FINAL size, all /64) -> Stage 1 (half, all /32):
  1080p: 1920x1088 / 1088x1920 / 1408x1408  (Stage 1: 960x544 / 544x960 / 704x704)
  720p:  1280x704  / 704x1280  / 960x960     (Stage 1: 640x352 / 352x640 / 480x480)
  576p:  1024x576  / 576x1024  / 768x768     (Stage 1: 512x288 / 288x512 / 384x384)

Fit (no pad):  proportional resize <= target, /64 aligned.
Crop (no pad): resize-to-cover then center-crop to exact bucket.
Square images map to each tier's square bucket.
"""
from __future__ import annotations

import torch

import comfy.utils
import comfy.model_management
import nodes

from ..wan22.buckets import _is_squareish, _validate_image_batch_4d


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


_LTX_BUCKETS = {
    "1080p": {"landscape": (1920, 1088), "portrait": (1088, 1920), "square": (1408, 1408)},
    "720p":  {"landscape": (1280, 704),  "portrait": (704, 1280),  "square": (960, 960)},
    "576p":  {"landscape": (1024, 576),  "portrait": (576, 1024),  "square": (768, 768)},
}


def _ceil32(x):
    x = (int(x) + 31) // 32 * 32
    return max(32, x)


def _floor32(x):
    x = int(x) // 32 * 32
    return max(32, x)


def _floor64(x):
    x = int(x) // 64 * 64
    return max(64, x)


def _ltx_stage1_dims(final_w, final_h):
    """Return Stage 1 dimensions that upscale exactly to the final size."""
    return max(32, int(final_w) // 2), max(32, int(final_h) // 2)


def _ltx_resize_fit_inside(img, out_w, out_h):
    """Resize to fit inside (out_w, out_h), output /64 aligned on both sides."""
    _, ih, iw, _ = img.shape
    s = min(out_w / iw, out_h / ih)
    tw = _floor64(iw * s)
    th = _floor64(ih * s)
    tw = max(32, min(tw, nodes.MAX_RESOLUTION))
    th = max(32, min(th, nodes.MAX_RESOLUTION))
    resized = comfy.utils.common_upscale(img.movedim(-1, 1), tw, th, "bilinear", "center").movedim(1, -1)
    return resized, tw, th


def _ltx_resize_then_center_crop(img, out_w, out_h):
    """Resize to cover (out_w, out_h) then center-crop to exact /32 target."""
    _, ih, iw, _ = img.shape
    s = max(out_w / iw, out_h / ih)
    tw = _ceil32(iw * s)
    th = _ceil32(ih * s)
    tmp = comfy.utils.common_upscale(img.movedim(-1, 1), tw, th, "bilinear", "center").movedim(1, -1)
    y0 = max(0, (th - out_h) // 2)
    x0 = max(0, (tw - out_w) // 2)
    return tmp[:, y0:y0+out_h, x0:x0+out_w, :]


def _ltx_pick_bucket(iw, ih, tier):
    """Pick the landscape / portrait / square bucket for the given tier."""
    tier_map = _LTX_BUCKETS[tier]
    if _is_squareish(iw, ih):
        return tier_map["square"]
    return tier_map["landscape"] if iw >= ih else tier_map["portrait"]


def _ltx_scale_image_core(image, tier="1080p", crop_to_fit=True):
    """
    Core LTX scaler. Returns (scaled_image, final_w, final_h, stage1_w, stage1_h).
    'tier' is the FINAL (Stage 2) size budget; Stage 1 is exactly half.
    """
    _, ih, iw, _ = image.shape

    bw, bh = _ltx_pick_bucket(iw, ih, tier)

    if _is_squareish(iw, ih):
        crop_to_fit = False

    if crop_to_fit:
        out = _ltx_resize_then_center_crop(image, bw, bh)
    else:
        out, bw, bh = _ltx_resize_fit_inside(image, bw, bh)

    final_w = int(out.shape[2])
    final_h = int(out.shape[1])
    stage1_w, stage1_h = _ltx_stage1_dims(final_w, final_h)
    return out, final_w, final_h, stage1_w, stage1_h


class LTX_Image_Scaler_MXD:
    """
    MXD Image Scaler for LTX Video (distilled two-stage workflow).

    'tier' is the FINAL (Stage 2) size; Stage 1 is exactly half. Finals are /64
    so Stage 1 stays /32 (the LTX latent constraint). Wire stage1_width /
    stage1_height into the empty latent for the low-res pass; the spatial
    upscaler-x2 then doubles it back to the final size.

    Tiers (final / Stage 1):
      1080p  1920x1088 (official 16:9) / 1088x1920 / 1408x1408  ->  half
      720p   1280x704 / 704x1280 / 960x960                      ->  half
      576p   1024x576 / 576x1024 / 768x768                      ->  half

    Modes:
      Perfect Fit (Crops Edges)  resize-to-cover + center-crop to exact bucket.
      Closest Fit (No Crop)      proportional resize, /64-aligned; may be smaller.

    Square images (within +-3% of 1:1) map to the tier's square bucket.
    Outputs the scaled image at final size plus the Stage 1 dimensions.
    """

    TITLE = "LTX Video Image Scaler MXD"
    CATEGORY = "image/processing"
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "scale"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tier": (["1080p", "720p", "576p"], {"default": "1080p"}),
                "crop_to_fit": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Crop Edges",
                    "label_off": "Closest Fit (No Crop)",
                }),
            }
        }

    def scale(self, image, tier="1080p", crop_to_fit=True):
        image = _validate_image_batch_4d(image, "LTX_Image_Scaler_MXD", "image")
        out, _final_w, _final_h, stage1_w, stage1_h = _ltx_scale_image_core(
            image, tier=tier, crop_to_fit=crop_to_fit
        )
        return (out, stage1_w, stage1_h)


NODE_CLASS_MAPPINGS = {
    "LTXVideoEmptyLatent_MXD": LTXVideoEmptyLatentMXD,
    "LTX_Image_Scaler_MXD": LTX_Image_Scaler_MXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVideoEmptyLatent_MXD": "LTX Empty Latent Video MXD",
    "LTX_Image_Scaler_MXD": "LTX Video Image Scaler MXD",
}
