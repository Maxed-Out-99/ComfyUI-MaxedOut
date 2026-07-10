"""Combine Materials FFGO MXD — ComfyUI node mirroring combine_materials.py."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _tensor_to_pil(tensor: torch.Tensor) -> list[Image.Image]:
    if tensor is None:
        return []
    if tensor.dim() == 4:
        images = []
        for i in range(tensor.shape[0]):
            img_np = 255.0 * tensor[i].cpu().numpy()
            images.append(Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8)))
        return images
    if tensor.dim() == 3:
        img_np = 255.0 * tensor.cpu().numpy()
        return [Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))]
    raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")


def _pil_to_tensor(pil_images: Image.Image | list[Image.Image]) -> torch.Tensor:
    if not isinstance(pil_images, list):
        pil_images = [pil_images]
    tensors = []
    for img in pil_images:
        img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img_np).unsqueeze(0))
    return torch.cat(tensors, dim=0)


def _rgba_to_rgb_white(img: Image.Image) -> Image.Image:
    """Convert an RGBA image to RGB with a white background."""
    rgba = img.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.split()[3])
    return background


def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    return _rgba_to_rgb_white(img)


def combine_to_fixed_canvas(
    img1: Image.Image,
    img2: Image.Image,
    canvas_size: tuple[int, int] = (1024, 512),
    direction: str = "horizontal",
) -> Image.Image:
    """Put two images into a fixed canvas, each occupying half with proportional scaling."""
    canvas_w, canvas_h = canvas_size
    img1 = _ensure_rgb(img1)
    img2 = _ensure_rgb(img2)

    if direction == "horizontal":
        target_w, target_h = canvas_w // 2, canvas_h
    else:
        target_w, target_h = canvas_w, canvas_h // 2

    img1.thumbnail((target_w, target_h), Image.LANCZOS)
    img2.thumbnail((target_w, target_h), Image.LANCZOS)

    combined = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    if direction == "horizontal":
        offset_x1 = (target_w - img1.width) // 2
        offset_y1 = (target_h - img1.height) // 2
        combined.paste(img1, (offset_x1, offset_y1))

        offset_x2 = target_w + (target_w - img2.width) // 2
        offset_y2 = (target_h - img2.height) // 2
        combined.paste(img2, (offset_x2, offset_y2))
    else:
        offset_x1 = (target_w - img1.width) // 2
        offset_y1 = (target_h - img1.height) // 2
        combined.paste(img1, (offset_x1, offset_y1))

        offset_x2 = (target_w - img2.width) // 2
        offset_y2 = target_h + (target_h - img2.height) // 2
        combined.paste(img2, (offset_x2, offset_y2))

    return combined


def combine_foregrounds_background(
    foregrounds: list[Image.Image],
    background: Image.Image,
    canvas_size: tuple[int, int] = (1024, 512),
) -> Image.Image:
    """Stack foregrounds on the left half; place background on the right half."""
    if not foregrounds:
        raise ValueError("[Combine Materials FFGO MXD] At least one foreground image is required.")

    canvas_w, canvas_h = canvas_size
    left_w, right_w = canvas_w // 2, canvas_w // 2

    bg = _ensure_rgb(background)
    bg = bg.resize((1280, 720))
    bg.thumbnail((right_w, canvas_h), Image.LANCZOS)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    offset_x = left_w + (right_w - bg.width) // 2
    offset_y = (canvas_h - bg.height) // 2
    canvas.paste(bg, (offset_x, offset_y))

    n = len(foregrounds)
    target_h = canvas_h // n
    for i, fg in enumerate(foregrounds):
        fg = _ensure_rgb(fg)
        fg.thumbnail((left_w, target_h), Image.LANCZOS)

        slot_x = (left_w - fg.width) // 2
        slot_y = i * target_h + (target_h - fg.height) // 2
        canvas.paste(fg, (slot_x, slot_y))

    return canvas


MAX_FOREGROUNDS = 6


class CombineTwoImagesFFGOMXD:
    """Mirrors combine_to_fixed_canvas() — two materials centered side-by-side
    on a split canvas (e.g. a main_entity + a single object cutout)."""

    SEARCH_ALIASES = ["combine materials", "ffgo", "two images", "material combine"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "canvas_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "canvas_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "direction": (["horizontal", "vertical"], {"default": "horizontal"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "combine"
    CATEGORY = "MXD/Image"

    def combine(self, image_1, image_2, canvas_width, canvas_height, direction="horizontal"):
        pil1 = _tensor_to_pil(image_1)[0]
        pil2 = _tensor_to_pil(image_2)[0]
        result = combine_to_fixed_canvas(
            pil1, pil2, canvas_size=(canvas_width, canvas_height), direction=direction
        )
        return (_pil_to_tensor(result),)


class CombineForegroundsBackgroundFFGOMXD:
    """Mirrors combine_foregrounds_background() — one or more foreground
    cutouts stacked on the left half, a background scene on the right. Each
    foreground_N is its own single-image socket (not a batched IMAGE tensor)
    since RGBA cutouts are rarely the same resolution and ComfyUI's IMAGE
    batch dim requires uniform shape; sockets are revealed one at a time in
    the UI as you connect them, like a reference/conditioning image input."""

    SEARCH_ALIASES = ["combine materials", "ffgo", "foreground background", "material combine"]

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(2, MAX_FOREGROUNDS + 1):
            optional[f"foreground_{i}"] = ("IMAGE",)

        return {
            "required": {
                "background": ("IMAGE",),
                "foreground_1": ("IMAGE",),
                "canvas_width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 8}),
                "canvas_height": ("INT", {"default": 720, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "combine"
    CATEGORY = "MXD/Image"

    def combine(self, background, foreground_1, canvas_width, canvas_height, **foreground_kwargs):
        fg_pils = [_tensor_to_pil(foreground_1)[0]]
        for i in range(2, MAX_FOREGROUNDS + 1):
            fg_tensor = foreground_kwargs.get(f"foreground_{i}")
            if fg_tensor is not None:
                fg_pils.append(_tensor_to_pil(fg_tensor)[0])

        bg_pil = _tensor_to_pil(background)[0]
        result = combine_foregrounds_background(
            fg_pils, bg_pil, canvas_size=(canvas_width, canvas_height)
        )
        return (_pil_to_tensor(result),)


NODE_CLASS_MAPPINGS = {
    "CombineTwoImagesFFGOMXD": CombineTwoImagesFFGOMXD,
    "CombineForegroundsBackgroundFFGOMXD": CombineForegroundsBackgroundFFGOMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineTwoImagesFFGOMXD": "Combine 2 Images FFGO MXD",
    "CombineForegroundsBackgroundFFGOMXD": "Combine Fore + Back FFGO MXD",
}
