"""WAN 2.2 resolution buckets: empty latents, image scaler, resolution matcher, outpaint pad.

Registered nodes:
  Wan2_2EmptyLatentImageMXD        Wan 2.2 Empty Latent Image MXD
  wan22EmptyHunyuanLatentVideoMXD  WAN2.2 Empty Latent Video MXD
  WAN22_I2V_Image_Scaler_MXD       Image Scaler Wan 2.2 I2V MXD
  WAN22_I2V_Match_Resolution_MXD   Match Resolution Wan 2.2 I2V MXD
  PadImageForOutpaintingMXD        Pad Image for Outpainting MXD

Canonical WAN 2.2 buckets: 480p tier 832x480 / 480x832 / 624x624, 720p tier
1280x720 / 720x1280 / 1024x1024. All scaling keeps dimensions 16-aligned.
"""
from __future__ import annotations
from typing import Tuple

import torch

import comfy.utils
import comfy.model_management
import nodes


# ---- Canonical WAN 2.2 buckets ----
SQUARE_TOL  = 0.03  # exact-ish square passthrough tolerance
AUTO_SQUARE_MAX_AR = 1.25  # Auto may crop to square when the source is within 25% of 1:1.

def _ar(w, h):
    return w / max(1, h)

def _safe_hw(w, h):
    w = max(16, min(w, nodes.MAX_RESOLUTION))
    h = max(16, min(h, nodes.MAX_RESOLUTION))
    return w, h

def _floor16(x):
    x = int(x) // 16 * 16
    return max(16, x)

def _ceil16(x):
    x = (int(x) + 15) // 16 * 16
    return max(16, x)

def _is_squareish(w, h, tol=SQUARE_TOL):
    r = _ar(w, h)
    return abs(r - 1.0) <= tol

def _is_auto_square_candidate(w, h):
    r = _ar(w, h)
    return max(r, 1.0 / max(r, 1e-9)) <= AUTO_SQUARE_MAX_AR

def _wan22_tier_from_area(iw, ih):
    area = iw * ih
    area_480 = 832 * 480
    area_720 = 1280 * 720
    return "480p" if abs(area - area_480) / area_480 <= abs(area - area_720) / area_720 else "720p"

def _wan22_square_bucket(tier, iw=None, ih=None):
    if tier == "720p":
        return (1024, 1024)
    if tier == "480p":
        return (624, 624)
    return (1024, 1024) if _wan22_tier_from_area(iw, ih) == "720p" else (624, 624)

def _wan22_oriented_bucket(tier, orientation, iw=None, ih=None):
    if tier == "Auto":
        tier = _wan22_tier_from_area(iw, ih)
    if orientation == "Tall":
        return (480, 832) if tier == "480p" else (720, 1280)
    if orientation == "Wide":
        return (832, 480) if tier == "480p" else (1280, 720)
    return _wan22_square_bucket(tier, iw, ih)

def _closest_bucket(img_w, img_h, bucket_list, cover=False):
    """
    Pick the best (bw,bh) from bucket_list for this image.
    Uses scale closeness + AR diff to rank.
    """
    in_ar = _ar(img_w, img_h)
    best, best_key = None, (float("inf"), 0.0)
    for bw, bh in bucket_list:
        s = max(bw/img_w, bh/img_h) if cover else min(bw/img_w, bh/img_h)
        ar_diff = abs(_ar(bw, bh) - in_ar)
        key = (abs(1.0 - s), ar_diff)
        if key < best_key:
            best_key, best = key, (bw, bh)
    return best

def _resize_then_center_crop(img, out_w, out_h):
    """
    Resize to cover target (ensures >= target on both sides after ceil16),
    then center-crop. No padding.
    """
    t, ih, iw, c = img.shape
    s = max(out_w / iw, out_h / ih)
    tw = _ceil16(iw * s)
    th = _ceil16(ih * s)
    tmp = comfy.utils.common_upscale(img.movedim(-1, 1), tw, th, "bilinear", "center").movedim(1, -1)
    y0 = max(0, (th - out_h) // 2)
    x0 = max(0, (tw - out_w) // 2)
    return tmp[:, y0:y0+out_h, x0:x0+out_w, :]

def _resize_fit_inside(img, out_w, out_h):
    """
    Resize to fit inside target (ensures <= target on both sides via floor16),
    and return the resized tensor only. No padding.
    """
    t, ih, iw, c = img.shape
    s = min(out_w / iw, out_h / ih)
    tw = _floor16(iw * s)
    th = _floor16(ih * s)
    tw, th = _safe_hw(tw, th)
    resized = comfy.utils.common_upscale(img.movedim(-1, 1), tw, th, "bilinear", "center").movedim(1, -1)
    return resized, tw, th

def _validate_image_batch_4d(image, node_name, input_name):
    if image is None:
        raise ValueError(f"[{node_name}] '{input_name}' is required.")
    if not torch.is_tensor(image):
        raise TypeError(f"[{node_name}] '{input_name}' must be an IMAGE torch tensor, got {type(image).__name__}.")
    if image.ndim != 4:
        raise ValueError(f"[{node_name}] '{input_name}' must have shape [T,H,W,C], got {tuple(image.shape)}.")
    if image.shape[0] <= 0:
        raise ValueError(f"[{node_name}] '{input_name}' contains zero images/frames.")
    if image.shape[1] <= 0 or image.shape[2] <= 0 or image.shape[3] <= 0:
        raise ValueError(f"[{node_name}] '{input_name}' has invalid dimensions {tuple(image.shape)}.")
    return image

def _resize_to_explicit_resolution(img, out_w, out_h, match_mode="crop_to_match"):
    """
    Resize IMAGE batch to an explicit resolution.
    - crop_to_match: cover + center crop (exact output)
    - fit_inside_only: preserve AR, no crop (may be smaller)
    - stretch_exact: force exact output (distorts AR)
    """
    out_w = int(out_w)
    out_h = int(out_h)
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"Invalid target resolution {out_w}x{out_h}.")

    if match_mode == "crop_to_match":
        return _resize_then_center_crop(img, out_w, out_h)

    if match_mode == "fit_inside_only":
        _, ih, iw, _ = img.shape
        s = min(out_w / max(1, iw), out_h / max(1, ih))
        tw = max(1, min(out_w, int(iw * s)))
        th = max(1, min(out_h, int(ih * s)))
        return comfy.utils.common_upscale(img.movedim(-1, 1), tw, th, "bilinear", "center").movedim(1, -1)

    if match_mode == "stretch_exact":
        return comfy.utils.common_upscale(img.movedim(-1, 1), out_w, out_h, "bilinear", "center").movedim(1, -1)

    raise ValueError(
        f"Invalid match_mode '{match_mode}'. Expected one of: crop_to_match, fit_inside_only, stretch_exact."
    )

_WAN22_VALID_RES = {
    (832, 480), (480, 832),
    (1280, 720), (720, 1280),
    (624, 624), (1024, 1024),
}

def _wan22_is_valid_dim(w, h):
    return (w, h) in _WAN22_VALID_RES


def _wan22_pick_bucket(iw, ih, tier, crop_to_fit, aspect_mode="Auto"):
    if tier == "Safe Auto":
        tier = "Auto"

    if aspect_mode in ("Tall", "Wide", "Square"):
        return _wan22_oriented_bucket(tier, aspect_mode, iw, ih)

    is_squareish = _is_squareish(iw, ih)
    is_landscape = iw >= ih

    # --- Square handling ---
    if is_squareish or (crop_to_fit and _is_auto_square_candidate(iw, ih)):
        return _wan22_square_bucket(tier, iw, ih)

    # --- Explicit tiers ---
    if tier == "480p":
        return _closest_bucket(iw, ih, [(832, 480)] if is_landscape else [(480, 832)], cover=crop_to_fit)
    if tier == "720p":
        return _closest_bucket(iw, ih, [(1280, 720)] if is_landscape else [(720, 1280)], cover=crop_to_fit)

    # --- Auto tier logic ---
    buckets_480 = [(832, 480)] if is_landscape else [(480, 832)]
    buckets_720 = [(1280, 720)] if is_landscape else [(720, 1280)]
    iw_ih = iw * ih
    area_480, area_720 = 832 * 480, 1280 * 720
    scale_to_480 = abs(iw_ih - area_480) / area_480
    scale_to_720 = abs(iw_ih - area_720) / area_720

    # prefer minimal scaling
    if iw <= 832 and ih <= 480:
        return _closest_bucket(iw, ih, buckets_480, cover=crop_to_fit)
    return _closest_bucket(iw, ih, buckets_480 if scale_to_480 <= scale_to_720 else buckets_720, cover=crop_to_fit)


def _wan22_scale_image_core(image, tier="Auto", crop_to_fit=False, aspect_mode="Auto"):
    """
    Shared WAN 2.2 scaler core.
    Returns (scaled_image, out_w, out_h, did_passthrough).
    """
    _, ih, iw, _ = image.shape

    # --- Safe Auto logic ---
    if tier == "Safe Auto":
        # passthrough if already WAN-safe
        if _wan22_is_valid_dim(iw, ih):
            return image, iw, ih, True

        area = iw * ih
        area_480, area_720 = 832 * 480, 1280 * 720
        min_area, max_area = int(area_480 * 0.5), int(area_720 * 1.8)

        if area < min_area or area > max_area:
            size_label = "small" if area < min_area else "large"
            raise ValueError(
                f"[WAN22_I2V_Image_Scaler_MXD] Input resolution {iw}x{ih} is too {size_label} for WAN 2.2 video buckets.\n"
                "WAN 2.2 works best around:\n"
                "  - 480p tier ~= 832x480 (or 480x832)\n"
                "  - 720p tier ~= 1280x720 (or 720x1280)\n"
                "  - Squares: 624x624 or 1024x1024\n\n"
                "Please use a source closer to 480p/720p, or first process it "
                "through your WAN 2.2 workflow. This ensures extend runs without mismatch."
            )
        # fallback to Auto scaling
        tier = "Auto"

    # --- Normal path (Auto / 480p / 720p) ---
    bw, bh = _wan22_pick_bucket(iw, ih, tier, crop_to_fit, aspect_mode=aspect_mode)
    if crop_to_fit:
        bw, bh = _safe_hw(_ceil16(bw), _ceil16(bh))
        out = _resize_then_center_crop(image, bw, bh)
    else:
        bw, bh = _safe_hw(_floor16(bw), _floor16(bh))
        out, _, _ = _resize_fit_inside(image, bw, bh)

    return out, int(out.shape[2]), int(out.shape[1]), False


# ---------- Empty latent image generator (for video nodes) ----------
class Wan2_2EmptyLatentImageMXD:
    """
    Utility node for WAN 2.2 workflows.
    Generates an empty latent tensor at common video-friendly resolutions.
    """

    DESCRIPTION = """Create an empty WAN 2.2 latent at a preset resolution."""
    TITLE = "WAN2.2 Empty Latent Image"
    CATEGORY = "WAN2.2/Latent"

    RESOLUTIONS = {
        "— 720p —": None,
        "Widescreen (16:9) 1280×720": (1280, 720),

        "— 480p —": None,
        "Widescreen (16:9) 832×480": (832, 480),
        "Square (1:1) 624×624": (624, 624),
    }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        options = list(cls.RESOLUTIONS.keys())
        return {
            "required": {
                "resolution": (
                    options,
                    {"default": "Square (1:1) 960×960", "tooltip": "Select target resolution preset."}
                ),
                "vertical": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Vertical", "label_off": "Landscape",
                     "tooltip": "Swap width/height for vertical orientation."}
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4096, "tooltip": "Number of latents to generate."}
                ),
            }
        }

    def generate(self, resolution, vertical, batch_size):
        size = self.RESOLUTIONS.get(resolution)
        if size is None:
            raise ValueError(f"'{resolution}' is a header or invalid option.")

        w, h = size
        if vertical:
            w, h = h, w

        # Safety: ensure divisible by 8
        if (w % 8) or (h % 8):
            raise ValueError(f"Resolution must be divisible by 8. Got {w}x{h}.")

        # WAN video length always t=1
        t = 1

        latent = torch.zeros(
            [batch_size, 16, t, h // 8, w // 8],
            device=comfy.model_management.intermediate_device()
        )
        return ({"samples": latent},)

# ---------- Empty latent video generator with presets (for video nodes) ----------
class wan22EmptyHunyuanLatentVideoMXD:
    """
    Exactly like core EmptyHunyuanLatentVideo, but width/height are replaced
    with valid WAN 2.2 resolution presets and a vertical toggle.
    """

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent/video"

    RESOLUTIONS = {
        "— 720p —": None,
        "Widescreen (16:9) 1280×720": (1280, 720),
        "Square (1:1) 1024×1024": (1024, 1024),

        "— 480p —": None,
        "Widescreen (16:9) 832×480": (832, 480),
        "Square (1:1) 624×624": (624, 624),
    }

    @classmethod
    def INPUT_TYPES(cls):
        options = list(cls.RESOLUTIONS.keys())
        return {
            "required": {
                "resolution": (
                    options,
                    {"default": "Widescreen (16:9) 832×480"}
                ),
                "vertical": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Vertical", "label_off": "Landscape"}
                ),
                "length": (
                    "INT",
                    {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4096}
                ),
            }
        }

    def generate(self, resolution, vertical, length, batch_size):
        size = self.RESOLUTIONS.get(resolution)
        if size is None:
            raise ValueError(f"'{resolution}' is not a selectable resolution.")
        w, h = size
        if vertical:
            w, h = h, w

        # identical to core behavior:
        t = ((length - 1) // 4) + 1
        latent = torch.zeros(
            [batch_size, 16, t, h // 8, w // 8],
            device=comfy.model_management.intermediate_device()
        )
        return ({"samples": latent},)


class WAN22_I2V_Image_Scaler_MXD:
    """
    MXD Image Scaler for WAN 2.2 (NO PADDING)
    - Modes: Auto / 480p / 720p (legacy "Safe Auto" still accepted)
    - Fit (no pad): proportional resize ≤ target; returns resized dims.
    - Crop (no pad): resize-to-cover then center-crop to exact target.
    - Square handling:
        * Auto & 480p: ~square → 624×624
        * 720p: ~square -> 1024x1024
    - “Safe Auto”:
        * If input is already a valid WAN 2.2 bucket, passthrough.
        * If input is far outside 480p–720p range, error early.
        * Otherwise, same logic as Auto.
        * Perfect for video-extend workflows.
    """

    TITLE = "Image Bucket Scaler MXD (No Pad)"
    CATEGORY = "image/processing"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tier": (["Auto", "480p", "720p"], {"default": "Auto"}),
                "crop_to_fit": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Perfect Fit (Crops Edges)",
                    "label_off": "Closest Fit (No Crop)"
                }),
                "aspect_mode": (["Auto", "Tall", "Wide", "Square"], {
                    "default": "Auto",
                    "tooltip": "Auto picks wide/tall/square from the source. Use Square/Tall/Wide to force the target bucket shape."
                }),
            }
        }

    def scale(self, image, tier="Auto", crop_to_fit=False, aspect_mode="Auto"):
        # Keep legacy "Safe Auto" values from old workflows working, but expose only one Auto in UI.
        internal_tier = "Safe Auto" if tier == "Auto" else tier
        out, _, _, _ = _wan22_scale_image_core(
            image,
            tier=internal_tier,
            crop_to_fit=crop_to_fit,
            aspect_mode=aspect_mode,
        )
        return (out,)

class WAN22_I2V_Match_Resolution_MXD:
    """
    Match a second image (or image batch) to a reference image resolution for WAN 2.2
    first/last-frame workflows.
    """
    TITLE = "WAN 2.2 I2V Match Resolution"
    CATEGORY = "image/processing"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("matched_image",)
    FUNCTION = "match_resolution"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE", {
                    "tooltip": "Reference size source (usually the first image after WAN bucket scaling)."
                }),
                "image_to_match": ("IMAGE", {
                    "tooltip": "Image or batch to resize using the reference image resolution."
                }),
                "match_mode": (["crop_to_match", "fit_inside_only", "stretch_exact"], {
                    "default": "crop_to_match",
                    "tooltip": "crop_to_match = exact size via cover+center crop; fit_inside_only = no crop, may be smaller; stretch_exact = exact size with distortion."
                }),
                "enforce_wan_bucket": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Validate WAN Bucket",
                    "label_off": "No WAN Validation",
                    "tooltip": "If enabled, reference_image must already be a WAN 2.2 bucket size."
                }),
            }
        }

    def match_resolution(self, reference_image, image_to_match, match_mode="crop_to_match", enforce_wan_bucket=False):
        node_name = "WAN22_I2V_Match_Resolution_MXD"
        reference_image = _validate_image_batch_4d(reference_image, node_name, "reference_image")
        image_to_match = _validate_image_batch_4d(image_to_match, node_name, "image_to_match")

        _, ref_h, ref_w, _ = reference_image.shape

        if enforce_wan_bucket and not _wan22_is_valid_dim(ref_w, ref_h):
            raise ValueError(
                f"[{node_name}] Reference image resolution {ref_w}x{ref_h} is not a valid WAN 2.2 bucket.\n"
                "Valid WAN 2.2 buckets are:\n"
                "  - 832x480 / 480x832\n"
                "  - 1280x720 / 720x1280\n"
                "  - 624x624 / 1024x1024\n\n"
                "Recommended workflow:\n"
                "  1. Scale the first image with 'Image Scaler Wan 2.2 I2V MXD'\n"
                "  2. Use this node to match the second image to the scaled first image"
            )

        matched = _resize_to_explicit_resolution(
            image_to_match,
            out_w=ref_w,
            out_h=ref_h,
            match_mode=match_mode,
        )
        return (matched,)


class PadImageForOutpaintingMXD:
    SEARCH_ALIASES = ["extend canvas", "expand image", "outpaint pad"]

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"
    CATEGORY = "image/transform"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 2}),
                "top": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 2}),
                "right": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 2}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 2}),
                "round_to": (["None", "2", "8", "16", "32", "64"], {"default": "16"}),
            }
        }

    @staticmethod
    def _nearest_multiple(value: int, multiple: int, padded: bool) -> int:
        if multiple <= 1 or value % multiple == 0:
            return value
        lower = (value // multiple) * multiple
        upper = lower + multiple
        if lower <= 0:
            return upper
        if not padded:
            return lower
        return lower if value - lower <= upper - value else upper

    @staticmethod
    def _axis_plan(size: int, before: int, after: int, multiple: int) -> Tuple[int, int, int, int, int]:
        target = size + before + after
        if multiple > 1:
            target = PadImageForOutpaintingMXD._nearest_multiple(target, multiple, before + after > 0)

        delta = target - (size + before + after)
        if delta < 0:
            remove = -delta
            from_after = min(after, remove)
            after -= from_after
            remove -= from_after
            from_before = min(before, remove)
            before -= from_before
            remove -= from_before
            crop_before = remove // 2
            crop_after = remove - crop_before
        else:
            crop_before = 0
            crop_after = 0
            if before > 0 and after > 0:
                add_before = delta // 2
                before += add_before
                after += delta - add_before
            elif before > 0:
                before += delta
            else:
                after += delta

        final_size = size - crop_before - crop_after + before + after
        if final_size <= 0:
            raise ValueError("[PadImageForOutpaintingMXD] Rounding removed the full image on one axis.")
        return before, after, crop_before, crop_after, final_size

    def expand_image(self, image, left, top, right, bottom, round_to="16"):
        image = _validate_image_batch_4d(image, "PadImageForOutpaintingMXD", "image")
        batch, height, width, channels = image.size()
        multiple = 1 if round_to == "None" else int(round_to)

        left, right, crop_left, crop_right, final_width = self._axis_plan(width, left, right, multiple)
        top, bottom, crop_top, crop_bottom, final_height = self._axis_plan(height, top, bottom, multiple)

        cropped = image[:, crop_top:height - crop_bottom, crop_left:width - crop_right, :]
        crop_height = cropped.shape[1]
        crop_width = cropped.shape[2]

        new_image = torch.full(
            (batch, final_height, final_width, channels),
            0.5,
            dtype=image.dtype,
            device=image.device,
        )
        new_image[:, top:top + crop_height, left:left + crop_width, :] = cropped

        mask = torch.ones(
            (final_height, final_width),
            dtype=torch.float32,
            device=image.device,
        )
        mask[top:top + crop_height, left:left + crop_width] = 0.0

        return (new_image, mask.unsqueeze(0))


NODE_CLASS_MAPPINGS = {
    "Wan2_2EmptyLatentImageMXD": Wan2_2EmptyLatentImageMXD,
    "wan22EmptyHunyuanLatentVideoMXD": wan22EmptyHunyuanLatentVideoMXD,
    "WAN22_I2V_Image_Scaler_MXD": WAN22_I2V_Image_Scaler_MXD,
    "WAN22_I2V_Match_Resolution_MXD": WAN22_I2V_Match_Resolution_MXD,
    "PadImageForOutpaintingMXD": PadImageForOutpaintingMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan2_2EmptyLatentImageMXD": "Wan 2.2 Empty Latent Image MXD",
    "wan22EmptyHunyuanLatentVideoMXD": "WAN2.2 Empty Latent Video MXD",
    "WAN22_I2V_Image_Scaler_MXD": "Image Scaler Wan 2.2 I2V MXD",
    "WAN22_I2V_Match_Resolution_MXD": "Match Resolution Wan 2.2 I2V MXD",
    "PadImageForOutpaintingMXD": "Pad Image for Outpainting MXD",
}
