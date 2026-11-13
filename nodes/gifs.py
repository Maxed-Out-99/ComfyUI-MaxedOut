"""Timed GIF loading, spatial stitching, and saving.

Registered nodes: Load GIF MXD, GIF Stitch MXD, Save GIF MXD.
MXD_GIF carries RGB IMAGE frames and their individual durations in milliseconds.
"""
from __future__ import annotations

import bisect
import hashlib
import os

import numpy as np
import torch
from PIL import Image, ImageSequence

import comfy.utils
import folder_paths


def _animation(frames, durations):
    if frames.ndim != 4 or frames.shape[0] == 0 or frames.shape[-1] != 3:
        raise ValueError("GIF frames must be a nonempty RGB image batch.")
    durations = [int(d) for d in durations]
    if len(durations) != frames.shape[0] or any(d < 10 or d % 10 for d in durations):
        raise ValueError("Each GIF frame needs a positive duration in multiples of 10 ms.")
    return {"frames": frames, "durations": durations}


def _timeline(animation, total):
    ends = np.cumsum(animation["durations"]).tolist()
    period = ends[-1]
    boundaries = {0, total}
    for offset in range(0, total, period):
        boundaries.update(offset + end for end in ends if offset + end < total)
    return ends, period, boundaries


def _stitch_frames(image1, image2, direction, match_image_size, spacing_width, spacing_color):
    """Use the same layout rules as core Image Stitch, on synchronized batches."""
    horizontal = direction in ("left", "right")
    if match_image_size:
        h1, w1 = image1.shape[1:3]
        h2, w2 = image2.shape[1:3]
        width, height = (max(1, int(h1 * w2 / h2)), h1) if horizontal else (w1, max(1, int(w1 * h2 / w2)))
        image2 = comfy.utils.common_upscale(
            image2.movedim(-1, 1), width, height, "lanczos", "disabled"
        ).movedim(1, -1)

    colors = {"white": (1, 1, 1), "black": (0, 0, 0), "red": (1, 0, 0),
              "green": (0, 1, 0), "blue": (0, 0, 1)}
    color = colors[spacing_color]
    # Core pads unmatched sizes with white/black, or black for RGB spacing colors.
    pad_value = 1.0 if spacing_color == "white" else 0.0
    dim = 1 if horizontal else 2
    target = max(image1.shape[dim], image2.shape[dim])

    def pad(image):
        delta = target - image.shape[dim]
        before, after = delta // 2, delta - delta // 2
        padding = (0, 0, 0, 0, before, after) if horizontal else (0, 0, before, after)
        return torch.nn.functional.pad(image, padding, value=pad_value) if delta else image

    image1, image2 = pad(image1), pad(image2)
    pieces = [image2, image1] if direction in ("left", "up") else [image1, image2]
    if spacing_width:
        spacing_width += spacing_width % 2
        shape = list(image1.shape)
        shape[2 if horizontal else 1] = spacing_width
        spacing = image1.new_empty(shape)
        spacing[:] = image1.new_tensor(color)
        pieces.insert(1, spacing)
    return torch.cat(pieces, dim=2 if horizontal else 1)


class LoadGifMXD:
    @classmethod
    def INPUT_TYPES(cls):
        root = folder_paths.get_input_directory()
        files = sorted(f for f in os.listdir(root) if f.lower().endswith(".gif")
                       and os.path.isfile(os.path.join(root, f)))
        return {"required": {"gif": (files, {"image_upload": True, "animated_image_upload": True})}}

    RETURN_TYPES = ("MXD_GIF", "IMAGE")
    RETURN_NAMES = ("gif", "images")
    FUNCTION = "load"
    CATEGORY = "Maxed Out/GIF"
    DESCRIPTION = "Load all GIF frames with their original timing. Transparency is composited onto black."

    def load(self, gif):
        path = folder_paths.get_annotated_filepath(gif)
        frames, durations = [], []
        with Image.open(path) as source:
            if source.format != "GIF":
                raise ValueError("Load GIF MXD requires a GIF file.")
            for frame in ImageSequence.Iterator(source):
                rgba = frame.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
                rgb = Image.alpha_composite(background, rgba).convert("RGB")
                frames.append(torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0))
                duration = int(frame.info.get("duration", 100))
                durations.append(max(10, duration) if duration > 0 else 100)
        frames = torch.stack(frames)
        return (_animation(frames, durations), frames)

    @classmethod
    def IS_CHANGED(cls, gif):
        digest = hashlib.sha256()
        with open(folder_paths.get_annotated_filepath(gif), "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, gif):
        if not folder_paths.exists_annotated_filepath(gif):
            return f"Invalid GIF file: {gif}"
        return True


class GifStitchMXD:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "gif1": ("MXD_GIF",),
            "direction": (["right", "down", "left", "up"], {"default": "right"}),
            "match_image_size": ("BOOLEAN", {"default": True}),
            "spacing_width": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 2}),
            "spacing_color": (["white", "black", "red", "green", "blue"], {"default": "white"}),
        }, "optional": {"gif2": ("MXD_GIF",)}}

    RETURN_TYPES = ("MXD_GIF", "IMAGE")
    RETURN_NAMES = ("gif", "images")
    FUNCTION = "stitch"
    CATEGORY = "Maxed Out/GIF"
    DESCRIPTION = "Stitch GIFs spatially with original timing. The shorter GIF loops to the longer GIF's duration."

    def stitch(self, gif1, direction, match_image_size, spacing_width, spacing_color, gif2=None):
        gif1 = _animation(gif1["frames"], gif1["durations"])
        if gif2 is None:
            return (gif1, gif1["frames"])
        gif2 = _animation(gif2["frames"], gif2["durations"])
        total = max(sum(gif1["durations"]), sum(gif2["durations"]))
        ends1, period1, bounds1 = _timeline(gif1, total)
        ends2, period2, bounds2 = _timeline(gif2, total)
        bounds = sorted(bounds1 | bounds2)
        # Merge consecutive identical frame pairs to avoid needless image copies.
        pairs, durations = [], []
        for start, end in zip(bounds, bounds[1:]):
            pair = (bisect.bisect_right(ends1, start % period1),
                    bisect.bisect_right(ends2, start % period2))
            if pairs and pairs[-1] == pair:
                durations[-1] += end - start
            else:
                pairs.append(pair)
                durations.append(end - start)
        image1 = gif1["frames"][[p[0] for p in pairs]]
        image2 = gif2["frames"][[p[1] for p in pairs]].to(image1)
        frames = _stitch_frames(image1, image2, direction, match_image_size, spacing_width, spacing_color)
        return (_animation(frames, durations), frames)


class SaveGifMXD:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"gif": ("MXD_GIF",),
                             "filename_prefix": ("STRING", {"default": "MaxedOut/GIF"})}}

    RETURN_TYPES = ("MXD_GIF",)
    RETURN_NAMES = ("gif",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Maxed Out/GIF"
    DESCRIPTION = "Save an infinitely looping GIF with its frame timing and an animated preview."

    def save(self, gif, filename_prefix="MaxedOut/GIF"):
        gif = _animation(gif["frames"], gif["durations"])
        images = gif["frames"]
        root, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images.shape[2], images.shape[1])
        filename = filename.replace("%batch_num%", "0")
        filename = f"{filename}_{counter:05}_.gif"
        frames = [Image.fromarray((frame.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
                  for frame in images]
        frames[0].save(os.path.join(root, filename), save_all=True, append_images=frames[1:],
                       duration=gif["durations"], loop=0, disposal=2, optimize=False)
        return {"ui": {"images": [{"filename": filename, "subfolder": subfolder, "type": "output"}],
                       "animated": (True,)}, "result": (gif,)}


NODE_CLASS_MAPPINGS = {"Load GIF MXD": LoadGifMXD, "GIF Stitch MXD": GifStitchMXD, "Save GIF MXD": SaveGifMXD}
NODE_DISPLAY_NAME_MAPPINGS = {key: key for key in NODE_CLASS_MAPPINGS}
