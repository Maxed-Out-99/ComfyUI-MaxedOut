"""Writes a finished live preview to output/live_previews and tells the panel.

    Both preview paths (system/live_preview.py's general animated previewer and
    nodes/ltx/preview.py's taeltx fallback) end a run the same way: they hold a
decoded (N, H, W, 3) float tensor in 0..1 and want it on disk so the panel can
show it with real playback controls after sampling stops.

A single-frame batch is saved as a PNG, not a one-frame mp4. Every non-video
model goes through the general previewer too, so a plain Flux/SDXL run used to
end up as a 1-frame video that the panel rendered in a <video> element,
complete with a scrubber for a clip with nothing to scrub. The `kind` field on
MXD_live_preview_saved tells the frontend which element to use.
"""
import os
import re
import time
from fractions import Fraction

from PIL import Image

import folder_paths

SUBFOLDER = "live_previews"


def _out_dir():
    path = os.path.join(folder_paths.get_output_directory(), SUBFOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def save_preview(node_id, frames, rate, log_prefix="[MXD video preview]"):
    """Save decoded frames (N, H, W, 3) in 0..1 to output/live_previews.

    Returns (filename, kind) for notify_saved(), or None if there was nothing
    to save.
    """
    if frames.ndim != 4 or frames.size(0) == 0:
        return None

    # Prompt node IDs are normally small integers, but the API accepts arbitrary
    # string keys. Keep them as a filename label without allowing path
    # separators, dot traversal, Windows device names, or overlong components.
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(node_id)).strip("_")[:80]
    safe_id = f"node_{label or 'unknown'}"
    stamp = time.time_ns()
    out_dir = _out_dir()

    if frames.size(0) == 1:
        kind = "image"
        filename = f"{safe_id}_{stamp}.png"
        array = frames[0].mul(0xFF).clamp(0, 0xFF).byte().cpu().numpy()
        Image.fromarray(array).save(os.path.join(out_dir, filename))
    else:
        # Imported lazily: comfy_api is optional on older cores, and only the
        # video branch needs it -- image previews should still work without it.
        from comfy_api.latest import VideoFromComponents, VideoComponents

        kind = "video"
        filename = f"{safe_id}_{stamp}.mp4"
        video = VideoFromComponents(
            VideoComponents(images=frames, frame_rate=Fraction(max(1, round(rate))))
        )
        video.save_to(os.path.join(out_dir, filename))

    print(f"{log_prefix} Saved live preview to {os.path.join(out_dir, filename)}")
    return filename, kind


def notify_saved(serv, node_id, saved):
    """Send MXD_live_preview_saved for a save_preview() result (no-op if None)."""
    if not saved:
        return
    filename, kind = saved
    serv.send_sync("MXD_live_preview_saved", {
        "node_id": node_id,
        "filename": filename,
        "subfolder": SUBFOLDER,
        "type": "output",
        "kind": kind,
    })

