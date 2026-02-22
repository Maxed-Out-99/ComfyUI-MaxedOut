"""
Standalone Image Comparer Node extracted from the original project.
"""
import os
import time

import folder_paths
from nodes import PreviewImage, SaveImage

# --- Constants ---
NAMESPACE = 'MXD'

def get_name(name):
    return '{} ({})'.format(name, NAMESPACE)

def get_category(sub_dirs=None):
    if sub_dirs is None:
        return NAMESPACE
    else:
        return "{}/utils".format(NAMESPACE)

# --- Node Definition ---
class MxdImageComparerSave(PreviewImage):
    """Compares original/new images and optionally saves outputs."""

    NAME = "Image Comparer + Save MXD"
    CATEGORY = get_category()
    FUNCTION = "compare_and_save"
    OUTPUT_NODE = True
    DESCRIPTION = "Compares images in-node; new image displays first and is the only image saved."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ([
                    "Save (New Img) + Preview",
                    "Save (New Img) Only",
                    "Preview Only"
                ], {
                    "default": "Save (New Img) + Preview",
                    "tooltip": "Choose whether to save the new image, preview both, or preview only."
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Base prefix for saved files. A suffix is added when both images are present."
                }),
            },
            "optional": {
                "original_image": ("IMAGE", {"tooltip": "Original image for comparison. Shown on slide."}),
                "new_image (displayed first)": ("IMAGE", {"tooltip": "New image to compare/save. Displayed first."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    def _build_prefix(self, base_prefix, suffix, other_present):
        if other_present:
            return "{}_{}".format(base_prefix, suffix)
        return base_prefix

    def _process_images(self, images, prefix, save_this, preview, prompt, extra_pnginfo):
        if images is None or len(images) == 0:
            return []
        if save_this:
            result = SaveImage().save_images(images, prefix, prompt, extra_pnginfo)
        elif preview:
            result = PreviewImage().save_images(images, prefix, prompt, extra_pnginfo)
        else:
            return []
        if not preview:
            return []
        if isinstance(result, dict):
            return result.get("ui", {}).get("images", [])
        return []

    def compare_and_save(self,
                         filename_prefix="ComfyUI",
                         mode="Save (New Img) + Preview",
                         prompt=None,
                         extra_pnginfo=None,
                         **kwargs):

        new_key = "new_image (displayed first)"
        original_image = kwargs.get("original_image", kwargs.get("original_image (displayed on slide)"))
        new_image = kwargs.get(new_key, kwargs.get("new_image"))

        preview = mode != "Save (New Img) Only"
        save_enabled = mode != "Preview Only"
        save_original = False
        save_new = save_enabled

        original_has = original_image is not None and len(original_image) > 0
        new_has = new_image is not None and len(new_image) > 0

        original_prefix = self._build_prefix(filename_prefix, "original", new_has)
        new_prefix = self._build_prefix(filename_prefix, "new", original_has)

        result_ui = {"a_images": [], "b_images": []}
        result_ui["a_images"] = self._process_images(
            new_image,
            new_prefix,
            save_new,
            preview,
            prompt,
            extra_pnginfo
        )
        result_ui["b_images"] = self._process_images(
            original_image,
            original_prefix,
            save_original,
            preview,
            prompt,
            extra_pnginfo
        )

        if preview:
            # Also publish standard ui.images so the sidebar shows previews.
            result_ui["images"] = result_ui["a_images"] + result_ui["b_images"]
            return {"ui": result_ui}
        return {}


class MxdVideoComparer:
    """Compare two videos in-node and pass through the new video."""

    NAME = "Video Comparer MXD"
    CATEGORY = "MXD/video"
    FUNCTION = "compare_videos"
    OUTPUT_NODE = True
    DESCRIPTION = "Compares original/new videos in-node; new video displays first and is passed through."
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("new_video",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "new_video": ("VIDEO", {"tooltip": "New video to compare. This output is passed through."}),
            },
            "optional": {
                "original_video": ("VIDEO", {"tooltip": "Original video for comparison."}),
            },
        }

    def _save_preview_video(self, video_obj, filename):
        if video_obj is None:
            return None

        previews_dir = os.path.join(folder_paths.get_output_directory(), "previews")
        os.makedirs(previews_dir, exist_ok=True)

        preview_path = os.path.join(previews_dir, filename)
        video_obj.save_to(preview_path, format="mp4", codec="h264")
        return {
            "filename": filename,
            "subfolder": "previews",
            "type": "output",
        }

    def compare_videos(self, new_video, original_video=None):
        stamp = int(time.time() * 1000)

        new_preview = self._save_preview_video(new_video, f"mxd_video_compare_new_{stamp}.mp4")
        original_preview = self._save_preview_video(
            original_video,
            f"mxd_video_compare_original_{stamp}.mp4",
        )

        result_ui = {
            "a_videos": [new_preview] if new_preview else [],
            "b_videos": [original_preview] if original_preview else [],
            "videos": [],
        }
        result_ui["videos"] = result_ui["a_videos"] + result_ui["b_videos"]

        return {
            "ui": result_ui,
            "result": (new_video,),
        }

# --- Registration ---
NODE_CLASS_MAPPINGS = {
    MxdImageComparerSave.NAME: MxdImageComparerSave,
    MxdVideoComparer.NAME: MxdVideoComparer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    MxdImageComparerSave.NAME: "Image Comparer + Save MXD",
    MxdVideoComparer.NAME: "Video Comparer MXD",
}

WEB_DIRECTORY = "."
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
