from __future__ import annotations
import torch, os, folder_paths, node_helpers, json, hashlib
import numpy as np
from PIL import Image, ImageOps, ImageSequence
from nodes import PreviewImage, SaveImage
try:
    from comfy_api.input_impl import VideoFromFile
    HAVE_COMFY_API_VIDEO = True
except Exception as _e:
    VideoFromFile = None
    HAVE_COMFY_API_VIDEO = False
    print(f"[ComfyUI-MaxedOut] comfy_api video I/O not available in media_io: {_e}")

from .shared.metadata import _safe_json_loads
from .shared.paths import _sort_paths_newest_first
from .shared.routes import register_get_route


def _extract_params_from_prompt_json(prompt_json: dict):
    """
    Returns (positive, negative) from saved Comfy prompt graph.
    """
    pos = ""
    neg = ""
    if not isinstance(prompt_json, dict):
        return pos, neg

    # unwrap if saved as {"prompt": {...}}
    graph = prompt_json.get("prompt", prompt_json)
    if not isinstance(graph, dict):
        return pos, neg

    # try to find KSampler/KSamplerAdvanced node
    ks = None
    for _, v in graph.items():
        if "KSampler" in v.get("class_type", ""):
            ks = v
            break
    if not ks:
        return pos, neg

    kin = ks.get("inputs", {})

    def _as_node_id(x):
        return str(x[0]) if isinstance(x, (list, tuple)) and x else None

    def _text_from_clip(node_id):
        n = graph.get(str(node_id), {})
        if n.get("class_type") == "CLIPTextEncode":
            return str(n.get("inputs", {}).get("text", "")).strip()
        return ""

    pos = _text_from_clip(_as_node_id(kin.get("positive")))
    neg = _text_from_clip(_as_node_id(kin.get("negative")))

    return pos, neg

def _indent_paths(paths):
    indented = []
    for path in paths:
        if not path:
            indented.append("")
            continue
        clean_path = path.lstrip("  ")
        depth = clean_path.count("/")
        indent = " " * (depth * 4)
        indented.append(indent + clean_path)
    return indented


def _scan_subdir_mtimes(root: str, subdirs: set, branch_latest: dict, exts: tuple = None):
    """
    Walk `root`, adding every subfolder's relative path to `subdirs` and
    bubbling the mtime of its most recently modified file up to every
    ancestor branch (including "" for the root) in `branch_latest`.

    When `exts` is given, a folder (and its ancestors) is only added if it
    directly or recursively contains at least one file matching `exts` --
    so folders with no relevant content don't show up as pickable at all.
    """
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            # Exclude hidden folders (e.g. .git, .github) and __pycache__
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
            rel_path = os.path.relpath(dirpath, root)
            rel_path = "" if rel_path == "." else rel_path.replace(os.path.sep, "/")

            latest = 0.0
            has_match = exts is None
            for f in filenames:
                if exts and not f.lower().endswith(exts):
                    continue
                has_match = True
                try:
                    m = os.path.getmtime(os.path.join(dirpath, f))
                except OSError:
                    continue
                if m > latest:
                    latest = m

            if not has_match:
                continue

            if rel_path:
                subdirs.add(rel_path)

            parts = [p for p in rel_path.split("/") if p]
            for i in range(len(parts) + 1):
                branch = "/".join(parts[:i])
                if latest > branch_latest.get(branch, -1.0):
                    branch_latest[branch] = latest
                if i > 0:
                    subdirs.add(branch)
    except OSError:
        pass


def _list_image_batch_subdirs(root: str, exts: tuple = None):
    """
    Recursive subfolders under `root`, newest first. Each folder is ordered by
    the mtime of the most recently modified file anywhere inside it (so a
    folder that just received a new file jumps back to the top). '' = the
    root itself, always first. If `exts` is given, only folders that
    directly or recursively contain a matching file are included.
    """
    subdirs = set()
    branch_latest = {}
    _scan_subdir_mtimes(root, subdirs, branch_latest, exts)
    ordered = sorted(subdirs, key=lambda d: (-branch_latest.get(d, -1.0), d.lower()))
    return [""] + ordered


def _list_image_batch_subdirs_union(output_root: str, input_root: str, exts: tuple = None):
    """
    Union of recursive subfolders from both roots, newest first. A folder
    present under both roots is ranked by whichever side has the more
    recent file, so it doesn't matter which source the user has selected.
    """
    subdirs = set()
    branch_latest = {}
    _scan_subdir_mtimes(output_root, subdirs, branch_latest, exts)
    _scan_subdir_mtimes(input_root, subdirs, branch_latest, exts)
    ordered = sorted(subdirs, key=lambda d: (-branch_latest.get(d, -1.0), d.lower()))
    return [""] + ordered


IMAGE_BATCH_EXTS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_BATCH_EXTS = (".mp4",)


def _list_files_recursive(root: str, exts: tuple):
    """Recursively list files under `root` matching `exts`, newest first, as relpaths."""
    try:
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
            for f in filenames:
                if f.lower().endswith(exts):
                    files.append(os.path.join(dirpath, f))
        files = _sort_paths_newest_first(files)
        return [os.path.relpath(f, root).replace(os.sep, "/") for f in files]
    except OSError:
        return []


def _list_files_recursive_union(output_root: str, input_root: str, exts: tuple):
    """
    Union of recursive files from both roots, newest first. A relative path
    present under both roots is ranked by whichever side's file is more
    recent, so it doesn't matter which source the user has selected.
    """
    mtimes = {}

    def scan(root):
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
                for f in filenames:
                    if not f.lower().endswith(exts):
                        continue
                    full = os.path.join(dirpath, f)
                    rel = os.path.relpath(full, root).replace(os.sep, "/")
                    try:
                        m = os.path.getmtime(full)
                    except OSError:
                        m = 0.0
                    if m > mtimes.get(rel, -1.0):
                        mtimes[rel] = m
        except OSError:
            pass

    scan(output_root)
    scan(input_root)
    return sorted(mtimes, key=lambda p: (-mtimes[p], p.lower())) or [""]


# Server routes so the frontend can swap folder/file dropdowns between
# inputs/outputs without reloading the page.
from aiohttp import web as _mxd_web


async def _mxd_list_image_batch_folders(request):
    return _mxd_web.json_response({
        "outputs": _indent_paths(_list_image_batch_subdirs(folder_paths.get_output_directory())),
        "inputs": _indent_paths(_list_image_batch_subdirs(folder_paths.get_input_directory())),
    })


async def _mxd_list_video_batch_folders(request):
    return _mxd_web.json_response({
        "outputs": _indent_paths(_list_image_batch_subdirs(folder_paths.get_output_directory(), VIDEO_BATCH_EXTS)),
        "inputs": _indent_paths(_list_image_batch_subdirs(folder_paths.get_input_directory(), VIDEO_BATCH_EXTS)),
    })


async def _mxd_list_single_loader_files(request):
    kind = request.query.get("kind", "image")
    exts = VIDEO_BATCH_EXTS if kind == "video" else IMAGE_BATCH_EXTS
    return _mxd_web.json_response({
        "outputs": _list_files_recursive(folder_paths.get_output_directory(), exts),
        "inputs": _list_files_recursive(folder_paths.get_input_directory(), exts),
    })


register_get_route("/mxd/image_batch/folders", _mxd_list_image_batch_folders)
register_get_route("/mxd/video_batch/folders", _mxd_list_video_batch_folders)
register_get_route("/mxd/single_loader/files", _mxd_list_single_loader_files)


class LoadImageBatchMXD:
    DESCRIPTION = """Load images from an inputs or outputs folder, make masks from alpha, and read prompts."""
    TITLE = "Load Image Batch (Inputs/Outputs + Prompts)"
    CATEGORY = "MXD/Image"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "positive", "negative")
    OUTPUT_IS_LIST = (True, True, True, True)
    FUNCTION = "load_batch"

    @classmethod
    def INPUT_TYPES(cls):
        # Provide the union of inputs + outputs subfolders so any saved value
        # validates regardless of which source it belongs to. The frontend
        # filters the visible list down to the selected source on the fly.
        union = _indent_paths(_list_image_batch_subdirs_union(
            folder_paths.get_output_directory(), folder_paths.get_input_directory()
        ))
        return {
            "required": {
                "source": (("outputs", "inputs"), {"default": "outputs"}),
                "folder": (tuple(union), {"default": ""}),
            }
        }

    def _extract_prompts(self, image: Image.Image):
        pos, neg = "", ""
        try:
            raw = image.info.get("prompt")
            if raw:
                prompt_json = _safe_json_loads(raw)
                if prompt_json:
                    pos, neg = _extract_params_from_prompt_json(prompt_json)
                else:
                    pos = raw
        except Exception as e:
            print(f"[LoadImageBatchMXD] Prompt parse failed: {e}")
        return pos, neg

    def load_batch(self, folder: str, source: str = "outputs"):
        folder = folder.lstrip("  ")
        root = (
            folder_paths.get_input_directory()
            if source == "inputs"
            else folder_paths.get_output_directory()
        )
        folder_path = os.path.normpath(os.path.join(root, folder)) if folder else root

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"No such folder: {folder_path}")

        valid_exts = IMAGE_BATCH_EXTS

        # Recursively find all matching files
        files = []
        for dirpath, dirnames, filenames in os.walk(folder_path):
            dirnames.sort()
            for f in sorted(filenames):
                if f.lower().endswith(valid_exts):
                    files.append(os.path.join(dirpath, f))

        if not files:
            raise FileNotFoundError(f"No valid images found in folder '{folder_path}' (including subfolders)")

        images, masks, positives, negatives, prefixes = [], [], [], [], []

        for path in files:
            i = Image.open(path)
            i = ImageOps.exif_transpose(i)

            pos, neg = self._extract_prompts(i)
            positives.append(pos)
            negatives.append(neg)

            rgb = i.convert("RGB")
            arr = np.array(rgb).astype(np.float32) / 255.0
            img_t = torch.from_numpy(arr)[None, ...]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask_t = 1.0 - torch.from_numpy(mask).unsqueeze(0)
            else:
                h, w = arr.shape[:2]
                mask_t = torch.zeros((1, h, w), dtype=torch.float32)

            images.append(img_t)
            masks.append(mask_t)

        return (images, masks, positives, negatives)


class LoadVideoBatchMXD:
    DESCRIPTION = """Load videos from an inputs or outputs folder as a batch."""
    TITLE = "Load Video Batch (Inputs/Outputs)"
    CATEGORY = "MXD/Video"

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("VIDEO",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_batch"

    @classmethod
    def INPUT_TYPES(cls):
        # Same union-of-sources pattern as LoadImageBatchMXD; reuses that
        # node's folder listing helper, filtered to folders that actually
        # contain a video so empty/irrelevant folders don't show up.
        union = _indent_paths(_list_image_batch_subdirs_union(
            folder_paths.get_output_directory(), folder_paths.get_input_directory(), VIDEO_BATCH_EXTS
        ))
        return {
            "required": {
                "source": (("outputs", "inputs"), {"default": "outputs"}),
                "folder": (tuple(union), {"default": ""}),
            }
        }

    def load_batch(self, folder: str, source: str = "outputs"):
        if not HAVE_COMFY_API_VIDEO:
            raise RuntimeError(
                "[LoadVideoBatchMXD] Video output requires a newer ComfyUI core with "
                "comfy_api.latest / comfy_api.input_impl support. Please update ComfyUI."
            )

        folder = folder.lstrip("  ")
        root = (
            folder_paths.get_input_directory()
            if source == "inputs"
            else folder_paths.get_output_directory()
        )
        folder_path = os.path.normpath(os.path.join(root, folder)) if folder else root

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"No such folder: {folder_path}")

        valid_exts = VIDEO_BATCH_EXTS

        # Recursively find all matching files
        files = []
        for dirpath, dirnames, filenames in os.walk(folder_path):
            dirnames.sort()
            for f in sorted(filenames):
                if f.lower().endswith(valid_exts):
                    files.append(os.path.join(dirpath, f))

        if not files:
            raise FileNotFoundError(f"No valid videos found in folder '{folder_path}' (including subfolders)")

        videos = [VideoFromFile(path) for path in files]

        return (videos,)


class LoadImageFromFolderMXD:
    DESCRIPTION = (
        "Load a single image from any inputs/outputs subfolder. Turn on run_folder "
        "to auto-queue every image in that same folder, one after another."
    )
    TITLE = "Load Image (From Folder) MXD"
    CATEGORY = "MXD/Image"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "positive", "negative", "filename")
    FUNCTION = "load_image"

    @classmethod
    def INPUT_TYPES(cls):
        # Union of both sources so any saved value validates regardless of which
        # source it belongs to; the frontend narrows the visible list to the
        # selected source on the fly (mirrors LoadImageBatchMXD's folder picker).
        union = _list_files_recursive_union(
            folder_paths.get_output_directory(), folder_paths.get_input_directory(), IMAGE_BATCH_EXTS
        )
        return {
            "required": {
                "source": (("outputs", "inputs"), {"default": "outputs"}),
                "image": (tuple(union), ),
                "run_folder": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, hitting Queue Prompt auto-queues every image in this file's folder, one after another, instead of just the selected file.",
                }),
            }
        }

    def _extract_prompts(self, image: Image.Image):
        pos, neg = "", ""
        try:
            raw = image.info.get("prompt")
            if raw:
                prompt_json = _safe_json_loads(raw)
                if prompt_json:
                    pos, neg = _extract_params_from_prompt_json(prompt_json)
                else:
                    pos = raw
        except Exception as e:
            print(f"[LoadImageFromFolderMXD] Prompt parse failed: {e}")
        return pos, neg

    def load_image(self, image: str, source: str = "outputs", run_folder: bool = False):
        image = image.lstrip("  ")
        root = (
            folder_paths.get_input_directory()
            if source == "inputs"
            else folder_paths.get_output_directory()
        )
        path = os.path.normpath(os.path.join(root, image)) if image else None

        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"No such image: {path}")

        i = Image.open(path)
        i = ImageOps.exif_transpose(i)

        pos, neg = self._extract_prompts(i)

        rgb = i.convert("RGB")
        arr = np.array(rgb).astype(np.float32) / 255.0
        img_t = torch.from_numpy(arr)[None, ...]

        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask_t = 1.0 - torch.from_numpy(mask).unsqueeze(0)
        else:
            h, w = arr.shape[:2]
            mask_t = torch.zeros((1, h, w), dtype=torch.float32)

        return (img_t, mask_t, pos, neg, os.path.basename(path))

    @classmethod
    def IS_CHANGED(cls, image, source="outputs", run_folder=False):
        root = (
            folder_paths.get_input_directory()
            if source == "inputs"
            else folder_paths.get_output_directory()
        )
        path = os.path.normpath(os.path.join(root, image.lstrip("  "))) if image else None
        if not path or not os.path.isfile(path):
            return ""
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()


class LoadVideoFromFolderMXD:
    DESCRIPTION = (
        "Load a single video from any inputs/outputs subfolder. Turn on run_folder "
        "to auto-queue every video in that same folder, one after another."
    )
    TITLE = "Load Video (From Folder) MXD"
    CATEGORY = "MXD/Video"

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("VIDEO", "filename")
    FUNCTION = "load_video"

    @classmethod
    def INPUT_TYPES(cls):
        union = _list_files_recursive_union(
            folder_paths.get_output_directory(), folder_paths.get_input_directory(), VIDEO_BATCH_EXTS
        )
        return {
            "required": {
                "source": (("outputs", "inputs"), {"default": "outputs"}),
                "video": (tuple(union), ),
                "run_folder": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, hitting Queue Prompt auto-queues every video in this file's folder, one after another, instead of just the selected file.",
                }),
            }
        }

    def load_video(self, video: str, source: str = "outputs", run_folder: bool = False):
        if not HAVE_COMFY_API_VIDEO:
            raise RuntimeError(
                "[LoadVideoFromFolderMXD] Video output requires a newer ComfyUI core with "
                "comfy_api.latest / comfy_api.input_impl support. Please update ComfyUI."
            )

        video = video.lstrip("  ")
        root = (
            folder_paths.get_input_directory()
            if source == "inputs"
            else folder_paths.get_output_directory()
        )
        path = os.path.normpath(os.path.join(root, video)) if video else None

        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"No such video: {path}")

        return (VideoFromFile(path), os.path.basename(path))

    @classmethod
    def IS_CHANGED(cls, video, source="outputs", run_folder=False):
        root = (
            folder_paths.get_input_directory()
            if source == "inputs"
            else folder_paths.get_output_directory()
        )
        path = os.path.normpath(os.path.join(root, video.lstrip("  "))) if video else None
        if not path or not os.path.isfile(path):
            return ""
        try:
            return str(os.path.getmtime(path))
        except OSError:
            return ""


class LoadImageWithPromptsMXD:
    DESCRIPTION = """Load one input image, create a mask from alpha, and read prompts if present."""
    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "positive", "negative")
    FUNCTION = "load_image"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        files = _sort_paths_newest_first([os.path.join(input_dir, f) for f in files])
        files = [os.path.basename(f) for f in files]
        return {"required": {"image": (files, {"image_upload": True})}}

    def _extract_prompts(self, img: Image.Image):
        pos, neg = "", ""
        raw = img.info.get("prompt")
        if raw:
            prompt_json = _safe_json_loads(raw)
            if prompt_json:
                pos, neg = _extract_params_from_prompt_json(prompt_json)
            else:
                pos = raw
        return pos, neg

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images, output_masks = [], []
        pos, neg = "", ""
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            frame = i.convert("RGB")

            if len(output_images) == 0:
                w, h = frame.size
                # extract prompts only once (from first frame)
                pos, neg = self._extract_prompts(i)

            if frame.size != (w, h):
                continue

            arr = np.array(frame).astype(np.float32) / 255.0
            tensor_img = torch.from_numpy(arr)[None, ...]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")

            output_images.append(tensor_img)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, pos, neg)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True

########################################################################################################################

class SaveImage_MXD:
    TITLE = "Save Image MXD"
    CATEGORY = "MXD/Image"
    OUTPUT_NODE = True
    FUNCTION = "save"

    DESCRIPTION = """Save images to the output folder or preview them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to preview and/or save."}),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "File name prefix. Tip: you can use a subfolder like 'tests/my_run'."
                }),
                "mode": ([
                    "Save + Preview",
                    "Save Only",
                    "Preview only"
                ], {
                    "default": "Save + Preview",
                    "tooltip": "Choose whether to write files to disk, only preview, or save quietly."
                }),
            },
            "optional": {
                "embed_workflow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Embed workflow metadata when saving PNG previews/files."
                }),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    OUTPUT_TOOLTIPS = ("Saves and/or previews the images.",)

    @staticmethod
    def _filtered_extra_pnginfo(extra_pnginfo, embed_workflow):
        if embed_workflow or not isinstance(extra_pnginfo, dict):
            return extra_pnginfo
        filtered = {k: v for k, v in extra_pnginfo.items() if str(k).lower() != "workflow"}
        return filtered or None

    def save(self, images, filename_prefix, mode, embed_workflow=True, prompt=None, extra_pnginfo=None):
        if embed_workflow:
            save_prompt = prompt
            save_extra_pnginfo = self._filtered_extra_pnginfo(extra_pnginfo, True)
        else:
            # Core SaveImage embeds the hidden `prompt` graph too.
            # Drop both to truly disable workflow reconstruction from saved files.
            save_prompt = None
            save_extra_pnginfo = None

        if mode.startswith("Preview"):
            return PreviewImage().save_images(images, filename_prefix, save_prompt, save_extra_pnginfo)
        result = SaveImage().save_images(images, filename_prefix, save_prompt, save_extra_pnginfo)
        if mode == "Save Only" and isinstance(result, dict):
            # Strip UI previews so nothing shows up in the ComfyUI viewer.
            return {k: v for k, v in result.items() if k != "ui"}
        return result

########################################################################################################################

class ExtractWorkflowFromImageMXD:
    TITLE = "Extract Workflow From Image MXD"
    CATEGORY = "MXD/Image"
    OUTPUT_NODE = True
    FUNCTION = "extract_and_save"

    DESCRIPTION = """Save workflow metadata to a JSON file from a wired image execution context."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Any connected image. Used to trigger extraction/save."}),
                "filename_prefix": ("STRING", {
                    "default": "workflow/ComfyUI",
                    "tooltip": "Output JSON prefix. You can include subfolders, e.g. 'workflow/my_run'.",
                }),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_path",)
    OUTPUT_TOOLTIPS = ("Relative path to the saved JSON file in outputs.",)

    @staticmethod
    def _decode_json_candidate(value):
        if value is None:
            return None

        if isinstance(value, (dict, list)):
            return value

        if isinstance(value, bytes):
            for enc in ("utf-8", "utf-16", "latin-1"):
                try:
                    value = value.decode(enc)
                    break
                except Exception:
                    continue
            if isinstance(value, bytes):
                value = value.decode("utf-8", "ignore")

        if not isinstance(value, str):
            return None

        raw = value.strip()
        if not raw:
            return None

        if raw.lower().startswith("workflow:"):
            raw = raw.split(":", 1)[1].strip()

        parsed = _safe_json_loads(raw)
        if isinstance(parsed, (dict, list)):
            return parsed
        return None

    def _extract_workflow_from_context(self, prompt=None, extra_pnginfo=None):
        if isinstance(extra_pnginfo, dict):
            for key in ("workflow", "Workflow"):
                parsed = self._decode_json_candidate(extra_pnginfo.get(key))
                if parsed is not None:
                    return parsed

        parsed_extra = self._decode_json_candidate(extra_pnginfo)
        if isinstance(parsed_extra, dict):
            for key in ("workflow", "Workflow"):
                parsed = self._decode_json_candidate(parsed_extra.get(key))
                if parsed is not None:
                    return parsed

        if prompt is not None:
            parsed_prompt = self._decode_json_candidate(prompt)
            if parsed_prompt is not None:
                return {"prompt": parsed_prompt}
            if isinstance(prompt, dict):
                return {"prompt": prompt}

        return None

    def extract_and_save(self, image, filename_prefix="workflow/ComfyUI", prompt=None, extra_pnginfo=None):
        workflow = self._extract_workflow_from_context(prompt, extra_pnginfo)
        if workflow is None:
            raise ValueError(
                "No workflow metadata is available in this execution context. "
                "Connect generated images from the current run, or ensure workflow metadata is present."
            )

        filename_prefix += self.prefix_append
        height = image[0].shape[0]
        width = image[0].shape[1]
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, width, height
        )
        os.makedirs(full_output_folder, exist_ok=True)

        file = f"{filename}_{counter:05}_.json"
        save_path = os.path.join(full_output_folder, file)

        with open(save_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(workflow, f, ensure_ascii=False, indent=2)

        rel = os.path.join(subfolder, file) if subfolder else file
        rel = rel.replace("\\", "/")
        return {
            "ui": {"text": [f"Saved workflow JSON: {rel}"]},
            "result": (rel,),
        }

########################################################################################################################

NODE_CLASS_MAPPINGS = {
    "Load Image Batch MXD": LoadImageBatchMXD,
    "Load Video Batch MXD": LoadVideoBatchMXD,
    "LoadImageFromFolderMXD": LoadImageFromFolderMXD,
    "LoadVideoFromFolderMXD": LoadVideoFromFolderMXD,
    "LoadImageWithPromptsMXD": LoadImageWithPromptsMXD,
    "Save Image MXD": SaveImage_MXD,
    "Extract Workflow From Image MXD": ExtractWorkflowFromImageMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load Image Batch MXD": "Load Image Batch (Inputs/Outputs) MXD",
    "Load Video Batch MXD": "Load Video Batch (Inputs/Outputs) MXD",
    "LoadImageFromFolderMXD": "Load Image (From Folder) MXD",
    "LoadVideoFromFolderMXD": "Load Video (From Folder) MXD",
    "LoadImageWithPromptsMXD": "Load Image MXD",
    "Save Image MXD": "Save Image MXD",
    "Extract Workflow From Image MXD": "Extract Workflow From Image MXD",
}
