from __future__ import annotations
import os, re, glob, json, hashlib
from collections import deque
from typing import Any, Dict, Tuple, Optional, List, Union

import torch
from safetensors import safe_open

import folder_paths
import comfy.utils
import comfy.model_management
from comfy.cli_args import args
from nodes import KSamplerAdvanced
import node_helpers, nodes

# Comfy API
try:
    from comfy_api.latest import io, ui
    from comfy_api.input import VideoInput
    from comfy_api.input_impl import VideoFromFile, VideoFromComponents
    from comfy_api.util import VideoComponents, VideoContainer, VideoCodec
    HAVE_COMFY_API = True
except Exception as _e:
    io = None
    ui = None
    VideoInput = None
    VideoFromFile = None
    VideoFromComponents = None
    VideoComponents = None
    VideoContainer = None
    VideoCodec = None
    HAVE_COMFY_API = False
    print(f"[ComfyUI-MaxedOut] comfy_api not available in wan22nodes: {_e}")

from server import PromptServer
from aiohttp import web

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

routes = PromptServer.instance.routes

def _sort_paths_newest_first(paths: List[str]) -> List[str]:
    """Sort file paths by mtime desc (newest first), stable by normalized path."""
    def _mtime(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    return sorted(
        paths,
        key=lambda p: (-_mtime(p), p.replace("\\", "/").lower()),
    )

def _list_latent_subfolders(latents_root: str) -> List[str]:
    """
    List latent subfolders recursively (e.g. "a", "a/b"), newest first by
    latest latent mtime in each branch.
    """
    files = glob.glob(os.path.join(latents_root, "**", "*.latent"), recursive=True)
    if not files:
        return []

    folder_latest_mtime: Dict[str, float] = {}
    for file_path in files:
        rel_dir = os.path.relpath(os.path.dirname(file_path), latents_root).replace(os.sep, "/").strip("/")
        if not rel_dir or rel_dir == ".":
            continue
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            mtime = 0.0

        # Include each ancestor so both "a" and "a/b" appear as options.
        parts = [p for p in rel_dir.split("/") if p]
        for i in range(1, len(parts) + 1):
            branch = "/".join(parts[:i])
            prev = folder_latest_mtime.get(branch, -1.0)
            if mtime > prev:
                folder_latest_mtime[branch] = mtime

    return [
        folder
        for folder, _ in sorted(
            folder_latest_mtime.items(),
            key=lambda kv: (-kv[1], kv[0].lower()),
        )
    ]

@routes.get("/mxd/videos/input")
async def mxd_list_input_videos(request):
    """
    Return a JSON list of *video* files under the input folder (relative paths),
    sorted by last modified time (newest first) so the combo's 'first' entry
    is always the latest render.
    """
    input_dir = folder_paths.get_input_directory()
    entries = []

    for root, _, filenames in os.walk(input_dir):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in VIDEO_EXTS:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, input_dir).replace("\\", "/")
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    mtime = 0
                entries.append((mtime, rel))

    # 🔁 Sort newest → oldest, to match Comfy's internal behavior
    entries.sort(key=lambda x: x[0], reverse=True)

    files = [rel for _, rel in entries]
    return web.json_response(files)


# ---------- SaveLatent (Comfy-only; saves into input/latents) ----------
class SaveLatentMXD:
    DESCRIPTION = """Save latents to input/latents and keep prompt metadata."""
    TITLE = "Save Latent"
    CATEGORY = "MXD/Latents"
    RETURN_TYPES = ()  # only UI
    FUNCTION = "save_only"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Latent tensor to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "Prefix for saved latent filename."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    def save_only(self, samples, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None, unique_id=None):

        # ---------- Save Latent ----------
        latents_dir = os.path.join(folder_paths.get_input_directory(), "latents")
        os.makedirs(latents_dir, exist_ok=True)

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, latents_dir
        )

        # Metadata
        meta = None
        if not args.disable_metadata:
            meta = {}
            if prompt is not None:
                try: meta["prompt"] = json.dumps(prompt)
                except: pass
            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items():
                    try: meta[k] = json.dumps(v)
                    except: pass
            _attach_source_ksampler_metadata(meta, prompt, unique_id)

        file = os.path.join(full_output_folder, f"{filename}_{counter:05}_.latent")

        payload = {
            "latent_tensor": samples["samples"].contiguous(),
            "latent_format_version_0": torch.tensor([]),
        }

        comfy.utils.save_torch_file(payload, file, metadata=meta)

        return {}  # no previews, no UI

# ---------- SaveLatent I2V (saves latent + conditioning) ----------
class SaveLatent_I2V_MXD:
    """
    I2V-only saver that persists:
      • latent tensor  ->  .latent
      • pos/neg CONDITIONING  ->  .cond.pt
    """
    TITLE = "Save Latent I2V (with Conditioning)"
    CATEGORY = "MXD/Latents (I2V)"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_only"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "High-noise latent to save for later low-noise finishing."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive CONDITIONING after WAN image→video."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative CONDITIONING after WAN image→video."}),
                "filename_prefix": ("STRING", {"default": "I2V", "tooltip": "Prefix for saved files"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    def save_only(self, samples, positive, negative, filename_prefix="I2V",
                  prompt=None, extra_pnginfo=None, unique_id=None):
        _save_i2v_latent_bundle(
            samples=samples,
            positive=positive,
            negative=negative,
            filename_prefix=filename_prefix,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
            unique_id=unique_id,
        )
        return {}

class SaveLatent_VACE22_MXD(SaveLatent_I2V_MXD):
    """
    VACE 2.2 saver: I2V latent + conditioning sidecar + trim_latent value.
    Kept as a separate node so existing I2V workflows stay unchanged.
    """
    TITLE = "Save Latent Vace 2.2"
    CATEGORY = "MXD/Latents (VACE 2.2)"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = SaveLatent_I2V_MXD.INPUT_TYPES()
        inputs["optional"] = {
            "trim_latent": ("INT", {
                "default": 0,
                "min": 0,
                "max": 10000,
                "step": 1,
                "tooltip": "VACE 2.2 trim_latent value to preserve with this latent. Usually 0 or 1."
            }),
        }
        return inputs

    def save_only(self, samples, positive, negative, filename_prefix="I2V",
                  trim_latent=0, prompt=None, extra_pnginfo=None, unique_id=None):
        _save_i2v_latent_bundle(
            samples=samples,
            positive=positive,
            negative=negative,
            filename_prefix=filename_prefix,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
            unique_id=unique_id,
            sidecar_extra={"trim_latent": _coerce_trim_latent(trim_latent)},
        )
        return {}

# ---------- Helpers ----------
def _load_latent_file(latent_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], List[str]]:
    """
    Load safetensors latent with Comfy metadata.
    Returns (samples_dict, metadata_dict, keys_list)
    """
    with safe_open(latent_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        # prefer explicit key we write
        if "latent_tensor" in keys:
            t = f.get_tensor("latent_tensor").float().contiguous()
        else:
            # fall back (some variants might save using a different name)
            first = keys[0]
            t = f.get_tensor(first).float().contiguous()

        meta = f.metadata() or {}

        # if ancient format, rescale (match Comfy behavior)
        if "latent_format_version_0" not in keys:
            t = t * (1.0 / 0.18215)

    return {"samples": t}, meta, keys


def _safe_json_loads(s: Union[str, bytes, None]) -> Optional[Dict[str, Any]]:
    if s is None:
        return None
    if isinstance(s, bytes):
        try:
            s = s.decode("utf-8", "ignore")
        except Exception:
            return None
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        # sometimes double-encoded in metadata
        try:
            return json.loads(json.loads(s))
        except Exception:
            return None


def _node_sort_key(node_id: str) -> Tuple[int, Union[int, str]]:
    s = str(node_id)
    try:
        return (0, int(s))
    except Exception:
        return (1, s)


def _normalize_prompt_graph(prompt_json: Any) -> Dict[str, Any]:
    if not isinstance(prompt_json, dict):
        return {}
    graph = prompt_json.get("prompt", prompt_json)
    return graph if isinstance(graph, dict) else {}


def _get_graph_node(graph: Dict[str, Any], node_id: Any) -> Optional[Dict[str, Any]]:
    if node_id is None or not isinstance(graph, dict):
        return None
    node = graph.get(str(node_id))
    return node if isinstance(node, dict) else None


def _iter_graph_nodes_sorted(graph: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    nodes: List[Tuple[str, Dict[str, Any]]] = []
    for node_id, node in graph.items():
        if isinstance(node, dict):
            nodes.append((str(node_id), node))
    nodes.sort(key=lambda pair: _node_sort_key(pair[0]))
    return nodes


def _linked_node_id(value: Any) -> Optional[str]:
    if isinstance(value, (list, tuple)) and len(value) >= 1:
        return str(value[0])
    return None


def _is_ksampler_node(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    return "KSampler" in str(node.get("class_type", ""))


def _collect_upstream_linked_node_ids(node: Dict[str, Any]) -> List[str]:
    inputs = node.get("inputs", {})
    if not isinstance(inputs, dict):
        return []

    seen = set()
    ordered = []

    # Prefer latent-carrying links first.
    for key in ("samples", "latent", "latent_image"):
        linked = _linked_node_id(inputs.get(key))
        if linked is not None and linked not in seen:
            seen.add(linked)
            ordered.append(linked)

    # Then search all other connected inputs in stable order.
    for key, value in inputs.items():
        if key in ("samples", "latent", "latent_image"):
            continue
        linked = _linked_node_id(value)
        if linked is not None and linked not in seen:
            seen.add(linked)
            ordered.append(linked)

    return ordered


def _find_upstream_ksampler_node_id(graph: Dict[str, Any], start_node_id: Any) -> Optional[str]:
    if not isinstance(graph, dict) or start_node_id is None:
        return None

    queue: deque[str] = deque([str(start_node_id)])
    visited = set()

    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)

        node = _get_graph_node(graph, node_id)
        if not node:
            continue
        if _is_ksampler_node(node):
            return node_id

        for upstream_id in _collect_upstream_linked_node_ids(node):
            if upstream_id not in visited:
                queue.append(upstream_id)

    return None


def _extract_ksampler_params(node: Dict[str, Any]) -> Dict[str, Any]:
    inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
    if not isinstance(inputs, dict):
        inputs = {}

    out: Dict[str, Any] = {}

    def set_int(key: str):
        if key in inputs:
            try:
                out[key] = int(inputs[key])
            except Exception:
                pass

    def set_float(key: str):
        if key in inputs:
            try:
                out[key] = float(inputs[key])
            except Exception:
                pass

    def set_str(key: str):
        if key in inputs and not isinstance(inputs[key], (list, tuple, dict)):
            try:
                out[key] = str(inputs[key]).strip()
            except Exception:
                pass

    set_int("steps")
    set_float("cfg")
    set_str("sampler_name")
    set_str("scheduler")
    set_int("start_at_step")
    set_int("end_at_step")

    return out


def _attach_source_ksampler_metadata(meta: Dict[str, Any], prompt: Any, unique_id: Any) -> None:
    if not isinstance(meta, dict):
        return

    graph = _normalize_prompt_graph(prompt)
    if not graph:
        return

    save_node_id = str(unique_id) if unique_id is not None else ""
    if not save_node_id:
        return

    save_node = _get_graph_node(graph, save_node_id)
    if not save_node:
        return

    source_candidates = _collect_upstream_linked_node_ids(save_node)
    if not source_candidates:
        return

    source_ksampler_id = None
    for start_id in source_candidates:
        source_ksampler_id = _find_upstream_ksampler_node_id(graph, start_id)
        if source_ksampler_id:
            break

    if not source_ksampler_id:
        return

    source_node = _get_graph_node(graph, source_ksampler_id)
    if not source_node:
        return

    meta["mxd_source_save_node_id"] = save_node_id
    meta["mxd_source_ksampler_node_id"] = source_ksampler_id
    try:
        meta["mxd_source_ksampler_params"] = json.dumps(_extract_ksampler_params(source_node))
    except Exception:
        pass


def _build_latent_metadata(prompt=None, extra_pnginfo=None, unique_id=None, extra_meta=None):
    if args.disable_metadata:
        return None

    meta = {}
    if prompt is not None:
        try:
            meta["prompt"] = json.dumps(prompt)
        except Exception:
            pass
    if extra_pnginfo is not None:
        for k, v in extra_pnginfo.items():
            try:
                meta[k] = json.dumps(v)
            except Exception:
                pass
    if isinstance(extra_meta, dict):
        for k, v in extra_meta.items():
            try:
                meta[str(k)] = json.dumps(v)
            except Exception:
                pass
    _attach_source_ksampler_metadata(meta, prompt, unique_id)
    return meta


def _save_i2v_latent_bundle(
    samples,
    positive,
    negative,
    filename_prefix="I2V",
    prompt=None,
    extra_pnginfo=None,
    unique_id=None,
    sidecar_extra=None,
):
    latents_dir = os.path.join(folder_paths.get_input_directory(), "latents")
    os.makedirs(latents_dir, exist_ok=True)

    full_output_folder, filename, counter, _subfolder, _filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, latents_dir
    )

    extra_meta = sidecar_extra if isinstance(sidecar_extra, dict) else None
    meta = _build_latent_metadata(
        prompt=prompt,
        extra_pnginfo=extra_pnginfo,
        unique_id=unique_id,
        extra_meta=extra_meta,
    )

    latent_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.latent")
    payload = {
        "latent_tensor": samples["samples"].contiguous(),
        "latent_format_version_0": torch.tensor([]),
    }
    comfy.utils.save_torch_file(payload, latent_path, metadata=meta)

    sidecar = {"positive": positive, "negative": negative}
    if isinstance(sidecar_extra, dict):
        sidecar.update(sidecar_extra)
    torch.save(sidecar, latent_path.replace(".latent", ".cond.pt"))
    return latent_path


def _load_i2v_conditioning_sidecar(latent_path):
    cond_path = latent_path.replace(".latent", ".cond.pt")
    if not os.path.exists(cond_path):
        return [], [], {}

    try:
        data = torch.load(cond_path, map_location="cpu")
    except Exception:
        return [], [], {}

    if not isinstance(data, dict):
        return [], [], {}

    return data.get("positive", []), data.get("negative", []), data


def _coerce_trim_latent(value, default=0):
    try:
        if isinstance(value, str):
            parsed = _safe_json_loads(value)
            value = parsed if parsed is not None else value
        return int(value)
    except Exception:
        return int(default)


def _extract_prompt_text_from_ksampler(graph: Dict[str, Any], ks_node: Dict[str, Any]) -> Tuple[str, str]:
    pos = ""
    neg = ""

    inputs = ks_node.get("inputs", {}) if isinstance(ks_node, dict) else {}
    if not isinstance(inputs, dict):
        return pos, neg

    def _text_from_clip(link_value: Any) -> str:
        node_id = _linked_node_id(link_value)
        if node_id is None:
            return ""
        node = _get_graph_node(graph, node_id) or {}
        if node.get("class_type") == "CLIPTextEncode":
            return str(node.get("inputs", {}).get("text", "")).strip()
        return ""

    pos = _text_from_clip(inputs.get("positive"))
    neg = _text_from_clip(inputs.get("negative"))
    return pos, neg


def _extract_params_from_prompt_json(
    prompt_json: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, int, float, str, str, int]:
    """
    Returns: (positive, negative, steps, cfg, sampler_name, scheduler, end_at_step)
    parsed from the saved Comfy prompt graph with deterministic KSampler selection.
    """
    pos = ""
    neg = ""
    steps = 20
    cfg = 8.0
    sampler_name = ""
    scheduler = ""
    end_at_step = 0

    graph = _normalize_prompt_graph(prompt_json)
    if not isinstance(graph, dict):
        return pos, neg, steps, cfg, sampler_name, scheduler, end_at_step

    ks_node = None
    extracted_params: Dict[str, Any] = {}

    # 1) Source KSampler id saved directly in latent metadata.
    if isinstance(meta, dict):
        raw_ks = meta.get("mxd_source_ksampler_node_id")
        if raw_ks is not None:
            candidate = _get_graph_node(graph, str(raw_ks))
            if candidate and _is_ksampler_node(candidate):
                ks_node = candidate

    # 2) Source save node id -> trace upstream to nearest KSampler.
    if ks_node is None and isinstance(meta, dict):
        raw_save = meta.get("mxd_source_save_node_id")
        if raw_save is not None:
            save_node = _get_graph_node(graph, str(raw_save))
            if save_node:
                for start_id in _collect_upstream_linked_node_ids(save_node):
                    trace_id = _find_upstream_ksampler_node_id(graph, start_id)
                    if trace_id:
                        candidate = _get_graph_node(graph, trace_id)
                        if candidate and _is_ksampler_node(candidate):
                            ks_node = candidate
                            break

    # 3) Legacy fallback: last KSampler node in graph.
    if ks_node is None:
        for _, node in _iter_graph_nodes_sorted(graph):
            if _is_ksampler_node(node):
                ks_node = node

    if not ks_node:
        return pos, neg, steps, cfg, sampler_name, scheduler, end_at_step

    pos, neg = _extract_prompt_text_from_ksampler(graph, ks_node)
    extracted_params = _extract_ksampler_params(ks_node)

    if "steps" in extracted_params:
        steps = int(extracted_params["steps"])
    if "cfg" in extracted_params:
        cfg = float(extracted_params["cfg"])
    if "end_at_step" in extracted_params:
        end_at_step = int(extracted_params["end_at_step"])
    if "sampler_name" in extracted_params:
        sampler_name = str(extracted_params["sampler_name"]).strip()
    if "scheduler" in extracted_params:
        scheduler = str(extracted_params["scheduler"]).strip()

    # Fallback to saved parameter snapshot if graph parse is incomplete.
    if isinstance(meta, dict):
        saved_params = _safe_json_loads(meta.get("mxd_source_ksampler_params"))
        if isinstance(saved_params, dict):
            if "steps" in saved_params and "steps" not in extracted_params:
                try:
                    steps = int(saved_params["steps"])
                except Exception:
                    pass
            if "cfg" in saved_params and "cfg" not in extracted_params:
                try:
                    cfg = float(saved_params["cfg"])
                except Exception:
                    pass
            if "end_at_step" in saved_params and "end_at_step" not in extracted_params:
                try:
                    end_at_step = int(saved_params["end_at_step"])
                except Exception:
                    pass
            if "sampler_name" in saved_params and "sampler_name" not in extracted_params:
                try:
                    sampler_name = str(saved_params["sampler_name"]).strip()
                except Exception:
                    pass
            if "scheduler" in saved_params and "scheduler" not in extracted_params:
                try:
                    scheduler = str(saved_params["scheduler"]).strip()
                except Exception:
                    pass

    return pos, neg, steps, cfg, sampler_name, scheduler, end_at_step

# ---------- Load a single latent (WITH Comfy params, consistent with folder version) ----------
class LoadLatent_WithParams:
    DESCRIPTION = """Load one latent and return prompts and sampler settings."""
    TITLE = "Load Latent (With Params)"
    CATEGORY = "MXD/Latents"
    RETURN_TYPES = ("FLOAT", "STRING", "STRING", "LATENT", "INT", "FLOAT", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("shift","positive","negative","samples","steps","cfg","sampler_name","scheduler","end_at_step","filename_prefix")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(s):
        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        os.makedirs(latents_root, exist_ok=True)

        files = glob.glob(os.path.join(latents_root, "**", "*.latent"), recursive=True)
        files = _sort_paths_newest_first(files)
        options = [os.path.relpath(f, latents_root).replace(os.sep, "/") for f in files]

        # live enums from KSamplerAdvanced so values wire cleanly
        ks_inputs = KSamplerAdvanced.INPUT_TYPES().get("required", {})
        samplers_enum   = ks_inputs.get("sampler_name", ("STRING",))[0]
        schedulers_enum = ks_inputs.get("scheduler", ("STRING",))[0]

        # overwrite with live enums
        s.RETURN_TYPES = (
            "FLOAT",   # shift
            "STRING",  # positive
            "STRING",  # negative
            "LATENT",
            "INT",
            "FLOAT",
            samplers_enum,
            schedulers_enum,
            "INT",
            "STRING",  # filename_prefix
        )
        s._SAMPLERS_ENUM = samplers_enum
        s._SCHEDULERS_ENUM = schedulers_enum

        return {"required": {"latent": (options, )}}

    def _coerce_enum(self, value, enum_values):
        try:
            return value if (enum_values and value in enum_values) else (enum_values[0] if enum_values else value)
        except Exception:
            return value

    def _strip_counter(self, name: str) -> str:
        # Only strip the trailing pattern we generate when saving: "_<5digits>_"
        # Preserve numeric-only base names like "96".
        stem, _ = os.path.splitext(name)
        m = re.match(r"^(.*?)(?:_\d{5}_)$", stem)
        return m.group(1) if m else stem
    
    def _extract_sd3_shift(self, meta: dict, prompt_json: dict | None) -> float:
        """
        Find SD3 'shift' in several places:
        1) flat meta["shift"]
        2) nested in prompt/workflow JSON:
        - nodes[].{type|class_type} == "ModelSamplingSD3" -> inputs.shift or widgets_values[0]
        - runtime-style prompt dict mapping IDs -> {..., class_type: "ModelSamplingSD3"}
        Falls back to 5.0 if not found.
        """
        def try_float(x):
            try:
                return float(x)
            except Exception:
                return None

        # 1) flat meta
        if isinstance(meta, dict):
            v = try_float(meta.get("shift"))
            if v is not None:
                return v

        # parse any JSON-like strings present in meta
        def safe_load(x):
            try:
                return _safe_json_loads(x) if isinstance(x, str) else x
            except Exception:
                return None

        # Search helper over various JSON shapes
        def search_container(obj):
            # Direct dict containing shift
            if isinstance(obj, dict):
                if "shift" in obj:
                    v = try_float(obj.get("shift"))
                    if v is not None:
                        return v

                # Comfy "nodes": [ {...}, ... ]
                nodes = obj.get("nodes")
                if isinstance(nodes, list):
                    # take the last SD3 node (most recent in graph)
                    ms_nodes = [n for n in nodes if isinstance(n, dict) and (
                        n.get("type") == "ModelSamplingSD3" or
                        n.get("class_type") == "ModelSamplingSD3" or
                        (isinstance(n.get("properties"), dict) and n["properties"].get("Node name for S&R") == "ModelSamplingSD3")
                    )]
                    if ms_nodes:
                        nd = ms_nodes[-1]
                        # Prefer explicit inputs.shift if present and literal
                        inp = nd.get("inputs")
                        if isinstance(inp, dict) and "shift" in inp:
                            vv = inp["shift"]
                            # ignore connection like [node_id, idx]
                            if not isinstance(vv, (list, tuple)):
                                v2 = try_float(vv)
                                if v2 is not None:
                                    return v2
                        # Fallback: first widget is shift for SD3 (as seen in your JSON)
                        w = nd.get("widgets_values")
                        if isinstance(w, list) and len(w) >= 1:
                            v2 = try_float(w[0])
                            if v2 is not None:
                                return v2

                # Runtime prompt map: {"42": {"class_type":"ModelSamplingSD3", "inputs":{...}, "widgets_values":[...]}, ...}
                # Heuristic: values that are dicts with class_type keys
                has_ct = [v for v in obj.values() if isinstance(v, dict) and "class_type" in v]
                if has_ct:
                    for nd in has_ct:
                        if nd.get("class_type") == "ModelSamplingSD3":
                            inp = nd.get("inputs", {})
                            if isinstance(inp, dict) and "shift" in inp:
                                vv = inp["shift"]
                                if not isinstance(vv, (list, tuple)):
                                    v2 = try_float(vv)
                                    if v2 is not None:
                                        return v2
                            w = nd.get("widgets_values")
                            if isinstance(w, list) and len(w) >= 1:
                                v2 = try_float(w[0])
                                if v2 is not None:
                                    return v2

            # Lists / nested
            if isinstance(obj, list):
                for it in obj:
                    v = search_container(it)
                    if v is not None:
                        return v
            return None

        # 2) Look in provided prompt_json
        v = search_container(prompt_json)
        if v is not None:
            return v

        # Also look in common meta fields that can hold the full workflow/prompt
        for key in ("workflow", "prompt", "extra_pnginfo"):
            candidate = meta.get(key)
            cand_obj = safe_load(candidate)
            if isinstance(cand_obj, dict) or isinstance(cand_obj, list):
                v = search_container(cand_obj)
                if v is not None:
                    return v
            # extra_pnginfo can nest "workflow"/"prompt" again
            if isinstance(cand_obj, dict):
                for subkey in ("workflow", "prompt"):
                    sub = safe_load(cand_obj.get(subkey))
                    if isinstance(sub, dict) or isinstance(sub, list):
                        v = search_container(sub)
                        if v is not None:
                            return v

        # default
        return 5.0

    def load(self, latent):
        # ✅ Ensure we prepend "latents/" if missing, but don't duplicate it
        if not latent.startswith("latents/"):
            latent_path = folder_paths.get_annotated_filepath(f"latents/{latent}")
        else:
            latent_path = folder_paths.get_annotated_filepath(latent)

        sample_dict, meta, _ = _load_latent_file(latent_path)
        t = sample_dict["samples"]

        if isinstance(t, torch.Tensor) and t.dim() >= 4 and t.size(0) > 1:
            samples = {"samples": t[0:1].contiguous()}
        elif isinstance(t, torch.Tensor) and t.dim() >= 4 and t.size(0) == 1:
            samples = {"samples": t}
        else:
            samples = {"samples": t.unsqueeze(0)}

        prompt_json = _safe_json_loads(meta.get("prompt"))
        pos, neg, steps, cfg, sampler_name, scheduler, end_at_step = _extract_params_from_prompt_json(prompt_json or {}, meta)

        # SD3 shift (not in KSamplerAdvanced, but we want it)
        shift = self._extract_sd3_shift(meta, prompt_json)

        sampler_name = self._coerce_enum(sampler_name, getattr(self.__class__, "_SAMPLERS_ENUM", ()))
        scheduler    = self._coerce_enum(scheduler,    getattr(self.__class__, "_SCHEDULERS_ENUM", ()))

        def normalize_folder(part: str) -> str:
            part = part.replace("\\", "/").strip("/")
            if not part:
                return ""
            segments = [seg for seg in part.split("/") if seg]
            if segments and segments[0].lower() == "latents":
                segments = segments[1:]
            return "/".join(segments)

        folder_part = normalize_folder(os.path.dirname(latent))
        base_name   = os.path.basename(latent_path)
        clean_stem  = self._strip_counter(base_name)
        prefix      = f"{folder_part}/{clean_stem}" if folder_part else clean_stem

        return (
            float(shift),
            pos,
            neg,
            samples,
            int(steps),
            float(cfg),
            sampler_name,
            scheduler,
            int(end_at_step),
            prefix,
        )

    @classmethod
    def IS_CHANGED(s, latent):
        p = folder_paths.get_annotated_filepath(f"latents/{latent}")
        m = hashlib.sha256()
        with open(p, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, latent):
        check_path = latent if latent.startswith("latents/") else f"latents/{latent}"
        try:
            folder_paths.get_annotated_filepath(check_path)
        except Exception:
            return f"Invalid latent file: {latent}"
        return True

# ---------- Load multiple latents from a folder (WITH Comfy params, list outputs, video-safe) ----------
class LoadLatents_FromFolder_WithParams:
    DESCRIPTION = """Load all latents in a folder with prompts and sampler settings."""
    TITLE = "Load Latents (Folder, With Params)"
    CATEGORY = "MXD/Latents"
    RETURN_TYPES  = (
        "FLOAT", 
        "STRING",  # positive
        "STRING",  # negative
        "LATENT",
        "INT",
        "FLOAT",
        "STRING",
        "STRING",
        "INT",
        "STRING"
    )
    RETURN_NAMES  = (
        "shift",
        "positive",
        "negative",
        "samples",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "end_at_step",
        "filename_prefix"
    )
    OUTPUT_IS_LIST = (True,) * 10
    FUNCTION = "load_batch"

    @classmethod
    def INPUT_TYPES(s):
        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        os.makedirs(latents_root, exist_ok=True)
        subs = [""] + _list_latent_subfolders(latents_root)

        # 🔧 FIX: safely import enums inside function to avoid overwriting RETURN_TYPES
        from nodes import KSamplerAdvanced
        ks_inputs = KSamplerAdvanced.INPUT_TYPES().get("required", {})
        samplers_enum   = ks_inputs.get("sampler_name", ("STRING",))[0]
        schedulers_enum = ks_inputs.get("scheduler", ("STRING",))[0]

        # ✅ Only swap the two enum fields, preserve other return types
        s.RETURN_TYPES = (
            "FLOAT",
            "STRING",
            "STRING",
            "LATENT",
            "INT",
            "FLOAT",
            samplers_enum,
            schedulers_enum,
            "INT",
            "STRING",
        )
        s._SAMPLERS_ENUM = samplers_enum
        s._SCHEDULERS_ENUM = schedulers_enum

        return {"required": {"subfolder": (subs,)}}

    def _coerce_enum(self, value, enum_values):
        try:
            return value if (enum_values and value in enum_values) else (enum_values[0] if enum_values else value)
        except Exception:
            return value

    def _strip_counter(self, name: str) -> str:
        stem, _ = os.path.splitext(name)
        m = re.match(r"^(.*?)(?:_\d{5}_)$", stem)
        return m.group(1) if m else stem
    
    def _extract_sd3_shift(self, meta: dict, prompt_json: dict | None) -> float:
        def try_float(x):
            try: return float(x)
            except Exception: return None

        if isinstance(meta, dict):
            v = try_float(meta.get("shift"))
            if v is not None: return v

        def safe_load(x):
            try: return _safe_json_loads(x) if isinstance(x, str) else x
            except Exception: return None

        def search_container(obj):
            if isinstance(obj, dict):
                if "shift" in obj:
                    v = try_float(obj.get("shift"))
                    if v is not None: return v
                nodes = obj.get("nodes")
                if isinstance(nodes, list):
                    ms_nodes = [n for n in nodes if isinstance(n, dict) and (
                        n.get("type") == "ModelSamplingSD3" or
                        n.get("class_type") == "ModelSamplingSD3" or
                        (isinstance(n.get("properties"), dict) and n["properties"].get("Node name for S&R") == "ModelSamplingSD3")
                    )]
                    if ms_nodes:
                        nd = ms_nodes[-1]
                        inp = nd.get("inputs")
                        if isinstance(inp, dict) and "shift" in inp:
                            vv = inp["shift"]
                            if not isinstance(vv, (list, tuple)):
                                v2 = try_float(vv)
                                if v2 is not None: return v2
                        w = nd.get("widgets_values")
                        if isinstance(w, list) and len(w) >= 1:
                            v2 = try_float(w[0])
                            if v2 is not None: return v2
                has_ct = [v for v in obj.values() if isinstance(v, dict) and "class_type" in v]
                for nd in has_ct:
                    if nd.get("class_type") == "ModelSamplingSD3":
                        inp = nd.get("inputs", {})
                        if isinstance(inp, dict) and "shift" in inp:
                            vv = inp["shift"]
                            if not isinstance(vv, (list, tuple)):
                                v2 = try_float(vv)
                                if v2 is not None: return v2
                        w = nd.get("widgets_values")
                        if isinstance(w, list) and len(w) >= 1:
                            v2 = try_float(w[0])
                            if v2 is not None: return v2
            if isinstance(obj, list):
                for it in obj:
                    v = search_container(it)
                    if v is not None: return v
            return None

        v = search_container(prompt_json)
        if v is not None: return v

        for key in ("workflow", "prompt", "extra_pnginfo"):
            candidate = meta.get(key)
            cand_obj = safe_load(candidate)
            if isinstance(cand_obj, (dict, list)):
                v = search_container(cand_obj)
                if v is not None: return v
            if isinstance(cand_obj, dict):
                for subkey in ("workflow", "prompt"):
                    sub = safe_load(cand_obj.get(subkey))
                    if isinstance(sub, (dict, list)):
                        v = search_container(sub)
                        if v is not None: return v

        return 5.0

    def load_batch(self, subfolder):
        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        base = os.path.join(latents_root, subfolder) if subfolder else latents_root
        files = glob.glob(os.path.join(base, "**", "*.latent"), recursive=True)
        files = _sort_paths_newest_first(files)
        if not files:
            raise RuntimeError(f"[LoadLatents_FromFolder_WithParams] No .latent files found in '{base}'.")

        shifts, samples_list, positives, negatives = [], [], [], []
        steps_list, cfgs, samplers, schedulers, end_steps, filename_prefixes = [], [], [], [], [], []

        for path in files:
            sample_dict, meta, _ = _load_latent_file(path)
            t = sample_dict["samples"]

            if isinstance(t, torch.Tensor) and t.dim() >= 4 and t.size(0) > 1:
                slices = [t[i:i+1].contiguous() for i in range(t.size(0))]
            elif isinstance(t, torch.Tensor) and t.dim() >= 4 and t.size(0) == 1:
                slices = [t]
            else:
                slices = [t.unsqueeze(0)]

            prompt_json = _safe_json_loads(meta.get("prompt"))
            pos, neg, n_steps, cfg, sampler_name, scheduler, end_at_step = _extract_params_from_prompt_json(prompt_json or {}, meta)
            sampler_name = self._coerce_enum(sampler_name, getattr(self.__class__, "_SAMPLERS_ENUM", ()))
            scheduler    = self._coerce_enum(scheduler, getattr(self.__class__, "_SCHEDULERS_ENUM", ()))
            shift_val = self._extract_sd3_shift(meta, prompt_json)

            folder_part = subfolder if subfolder else ""
            clean_stem = self._strip_counter(os.path.basename(path))
            prefix = os.path.join(folder_part, clean_stem) if folder_part else clean_stem

            for sl in slices:
                shifts.append(float(shift_val))
                positives.append(pos)
                negatives.append(neg)
                samples_list.append({"samples": sl})
                steps_list.append(int(n_steps))
                cfgs.append(float(cfg))
                samplers.append(sampler_name)
                schedulers.append(scheduler)
                end_steps.append(int(end_at_step))
                filename_prefixes.append(prefix)

        n = len(samples_list)
        if n == 0 or any(len(lst) != n for lst in (shifts, positives, negatives, steps_list, cfgs, samplers, schedulers, end_steps, filename_prefixes)):
            raise RuntimeError("[LoadLatents_FromFolder_WithParams] Internal length mismatch.")

        return (
            shifts,
            positives,
            negatives,
            samples_list,
            steps_list,
            cfgs,
            samplers,
            schedulers,
            end_steps,
            filename_prefixes,
        )
    
class LoadLatent_I2V_MXD(LoadLatent_WithParams):
    """
    Same outputs as LoadLatent_WithParams plus two CONDITIONING outputs at the end.
    Fixes sampler/scheduler enum wiring by setting enums on THIS subclass.
    """
    TITLE = "Load Latent I2V (With Params + Conditioning)"
    CATEGORY = "MXD/Latents (I2V)"
    FUNCTION = "load"

    RETURN_TYPES = (
        "FLOAT",         # shift
        "CONDITIONING",  # positive conditioning
        "CONDITIONING",  # negative conditioning
        "LATENT",
        "INT",
        "FLOAT",
        "STRING",
        "STRING",
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "shift",
        "positive",
        "negative",
        "samples",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "end_at_step",
        "filename_prefix",
    )

    @classmethod
    def INPUT_TYPES(s):
        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        os.makedirs(latents_root, exist_ok=True)
        files = glob.glob(os.path.join(latents_root, "**", "*.latent"), recursive=True)
        files = _sort_paths_newest_first(files)
        # Clean dropdown display (no "latents/" prefix)
        options = [os.path.relpath(f, latents_root).replace(os.sep, "/") for f in files]

        ks_inputs = KSamplerAdvanced.INPUT_TYPES().get("required", {})
        samplers_enum   = ks_inputs.get("sampler_name", ("STRING",))[0]
        schedulers_enum = ks_inputs.get("scheduler",    ("STRING",))[0]

        s.RETURN_TYPES = (
            "FLOAT", "CONDITIONING", "CONDITIONING", "LATENT",
            "INT", "FLOAT", samplers_enum, schedulers_enum,
            "INT", "STRING",
        )
        s._SAMPLERS_ENUM   = samplers_enum
        s._SCHEDULERS_ENUM = schedulers_enum

        return {"required": {"latent": (options, )}}

    @classmethod
    def IS_CHANGED(s, latent):
        # Fix path lookup (add "latents/" prefix back)
        p = folder_paths.get_annotated_filepath(f"latents/{latent}")
        m = hashlib.sha256()
        with open(p, "rb") as f:
            m.update(f.read())
        side = p.replace(".latent", ".cond.pt")
        if os.path.exists(side):
            with open(side, "rb") as f:
                m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, latent):
        # Pass prefixed path to base validator
        return LoadLatent_WithParams.VALIDATE_INPUTS(f"latents/{latent}")

    def load(self, latent):
        # Use base loader (add prefix so it finds the file)
        base_tuple = super().load(latent)

        # Load .cond.pt (conditioning data)
        latent_path = folder_paths.get_annotated_filepath(f"latents/{latent}")
        cond_path = latent_path.replace(".latent", ".cond.pt")

        positive_conditioning, negative_conditioning = [], []
        if os.path.exists(cond_path):
            try:
                d = torch.load(cond_path, map_location="cpu")
                positive_conditioning = d.get("positive", [])
                negative_conditioning = d.get("negative", [])
            except Exception:
                positive_conditioning, negative_conditioning = [], []

        (
            shift, _pos_text, _neg_text, samples,
            steps, cfg, sampler_name, scheduler,
            end_at_step, prefix,
        ) = base_tuple

        return (
            shift, positive_conditioning, negative_conditioning,
            samples, steps, cfg, sampler_name, scheduler,
            end_at_step, prefix,
        )
    
class LoadLatents_FromFolder_I2V_MXD(LoadLatents_FromFolder_WithParams):
    """
    Same as LoadLatents_FromFolder_WithParams, but includes CONDITIONING outputs
    (positive/negative tensors) loaded from paired `.cond.pt` sidecar files.
    """
    TITLE = "Load Latents (Folder, I2V + Conditioning)"
    CATEGORY = "MXD/Latents (I2V)"
    FUNCTION = "load_batch_i2v"

    # Types MUST declare CONDITIONING here, not STRING
    RETURN_TYPES = (
        "FLOAT",         # shift
        "CONDITIONING",  # positive conditioning
        "CONDITIONING",  # negative conditioning
        "LATENT",
        "INT",
        "FLOAT",
        "STRING",        # will be replaced with sampler enum in INPUT_TYPES
        "STRING",        # will be replaced with scheduler enum in INPUT_TYPES
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "shift",
        "positive",
        "negative",
        "samples",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "end_at_step",
        "filename_prefix",
    )

    # Still a batch node
    OUTPUT_IS_LIST = (True,) * 10

    @classmethod
    def INPUT_TYPES(s):
        # Same folder logic as the base class
        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        os.makedirs(latents_root, exist_ok=True)
        subs = [""] + _list_latent_subfolders(latents_root)

        # Pull live enums from KSamplerAdvanced so sampler/scheduler wire cleanly
        from nodes import KSamplerAdvanced
        ks_inputs = KSamplerAdvanced.INPUT_TYPES().get("required", {})
        samplers_enum   = ks_inputs.get("sampler_name", ("STRING",))[0]
        schedulers_enum = ks_inputs.get("scheduler",    ("STRING",))[0]

        # IMPORTANT: keep CONDITIONING types, only swap the sampler/scheduler slots
        s.RETURN_TYPES = (
            "FLOAT",         # shift
            "CONDITIONING",  # positive conditioning
            "CONDITIONING",  # negative conditioning
            "LATENT",
            "INT",
            "FLOAT",
            samplers_enum,   # enum type for sampler_name
            schedulers_enum, # enum type for scheduler
            "INT",
            "STRING",
        )
        s._SAMPLERS_ENUM   = samplers_enum
        s._SCHEDULERS_ENUM = schedulers_enum

        return {"required": {"subfolder": (subs, )}}

    def load_batch_i2v(self, subfolder):
        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        base = os.path.join(latents_root, subfolder) if subfolder else latents_root
        files = glob.glob(os.path.join(base, "**", "*.latent"), recursive=True)
        files = _sort_paths_newest_first(files)
        if not files:
            raise RuntimeError(f"[LoadLatents_FromFolder_I2V_MXD] No .latent files found in '{base}'.")

        shifts, samples_list = [], []
        positives, negatives = [], []
        steps_list, cfgs, samplers, schedulers, end_steps = [], [], [], [], []
        filename_prefixes = []

        for path in files:
            sample_dict, meta, _ = _load_latent_file(path)
            t = sample_dict["samples"]

            if isinstance(t, torch.Tensor) and t.dim() >= 4 and t.size(0) > 1:
                slices = [t[i:i+1].contiguous() for i in range(t.size(0))]
            else:
                slices = [t if (isinstance(t, torch.Tensor) and t.dim() >= 4 and t.size(0) == 1)
                          else t.unsqueeze(0)]

            prompt_json = _safe_json_loads(meta.get("prompt"))
            pos, neg, n_steps, cfg, sampler_name, scheduler, end_at_step = \
                _extract_params_from_prompt_json(prompt_json or {}, meta)

            sampler_name = self._coerce_enum(sampler_name, getattr(self.__class__, "_SAMPLERS_ENUM", ()))
            scheduler    = self._coerce_enum(scheduler,    getattr(self.__class__, "_SCHEDULERS_ENUM", ()))
            shift_val    = self._extract_sd3_shift(meta, prompt_json)

            # Load sidecar conditionings
            cond_path = path.replace(".latent", ".cond.pt")
            positive_conditioning, negative_conditioning = [], []
            if os.path.exists(cond_path):
                try:
                    d = torch.load(cond_path, map_location="cpu")
                    positive_conditioning = d.get("positive", [])
                    negative_conditioning = d.get("negative", [])
                except Exception:
                    pass

            folder_part = subfolder if subfolder else ""
            clean_stem  = self._strip_counter(os.path.basename(path))
            prefix      = os.path.join(folder_part, clean_stem) if folder_part else clean_stem

            for sl in slices:
                shifts.append(float(shift_val))
                positives.append(positive_conditioning)
                negatives.append(negative_conditioning)
                samples_list.append({"samples": sl})
                steps_list.append(int(n_steps))
                cfgs.append(float(cfg))
                samplers.append(sampler_name)
                schedulers.append(scheduler)
                end_steps.append(int(end_at_step))
                filename_prefixes.append(prefix)

        return (
            shifts,
            positives,
            negatives,
            samples_list,
            steps_list,
            cfgs,
            samplers,
            schedulers,
            end_steps,
            filename_prefixes,
        )

class LoadLatent_VACE22_MXD(LoadLatent_I2V_MXD):
    """
    I2V loader plus the VACE 2.2 trim_latent value saved by Save Latent Vace 2.2.
    """
    TITLE = "Load Latent Vace 2.2"
    CATEGORY = "MXD/Latents (VACE 2.2)"

    RETURN_TYPES = (
        "FLOAT",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "INT",
        "FLOAT",
        "STRING",
        "STRING",
        "INT",
        "STRING",
        "INT",
    )
    RETURN_NAMES = (
        "shift",
        "positive",
        "negative",
        "samples",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "end_at_step",
        "filename_prefix",
        "trim_latent",
    )

    @classmethod
    def INPUT_TYPES(s):
        inputs = LoadLatent_I2V_MXD.INPUT_TYPES.__func__(s)
        sampler_type = s.RETURN_TYPES[6]
        scheduler_type = s.RETURN_TYPES[7]
        s.RETURN_TYPES = (
            "FLOAT", "CONDITIONING", "CONDITIONING", "LATENT",
            "INT", "FLOAT", sampler_type, scheduler_type,
            "INT", "STRING", "INT",
        )
        return inputs

    def load(self, latent):
        base_tuple = super().load(latent)
        latent_ref = latent if str(latent).startswith("latents/") else f"latents/{latent}"
        latent_path = folder_paths.get_annotated_filepath(latent_ref)
        _pos, _neg, sidecar = _load_i2v_conditioning_sidecar(latent_path)
        _sample_dict, meta, _keys = _load_latent_file(latent_path)
        trim_latent = _coerce_trim_latent(sidecar.get("trim_latent", meta.get("trim_latent", 0)))
        return (*base_tuple, trim_latent)


class LoadLatents_FromFolder_VACE22_MXD(LoadLatents_FromFolder_I2V_MXD):
    """
    Batch I2V loader plus a trim_latent list aligned with each returned latent slice.
    """
    TITLE = "Load Latents (Folder, Vace 2.2)"
    CATEGORY = "MXD/Latents (VACE 2.2)"
    FUNCTION = "load_batch_vace22"

    RETURN_TYPES = (
        "FLOAT",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "INT",
        "FLOAT",
        "STRING",
        "STRING",
        "INT",
        "STRING",
        "INT",
    )
    RETURN_NAMES = (
        "shift",
        "positive",
        "negative",
        "samples",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "end_at_step",
        "filename_prefix",
        "trim_latent",
    )
    OUTPUT_IS_LIST = (True,) * 11

    @classmethod
    def INPUT_TYPES(s):
        inputs = LoadLatents_FromFolder_I2V_MXD.INPUT_TYPES.__func__(s)
        sampler_type = s.RETURN_TYPES[6]
        scheduler_type = s.RETURN_TYPES[7]
        s.RETURN_TYPES = (
            "FLOAT", "CONDITIONING", "CONDITIONING", "LATENT",
            "INT", "FLOAT", sampler_type, scheduler_type,
            "INT", "STRING", "INT",
        )
        return inputs

    def load_batch_vace22(self, subfolder):
        base_tuple = super().load_batch_i2v(subfolder)

        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        base = os.path.join(latents_root, subfolder) if subfolder else latents_root
        files = glob.glob(os.path.join(base, "**", "*.latent"), recursive=True)
        files = _sort_paths_newest_first(files)

        trims = []
        for path in files:
            sample_dict, meta, _keys = _load_latent_file(path)
            _pos, _neg, sidecar = _load_i2v_conditioning_sidecar(path)
            trim_latent = _coerce_trim_latent(sidecar.get("trim_latent", meta.get("trim_latent", 0)))
            t = sample_dict["samples"]
            slice_count = int(t.size(0)) if isinstance(t, torch.Tensor) and t.dim() >= 4 and t.size(0) > 1 else 1
            trims.extend([trim_latent] * slice_count)

        return (*base_tuple, trims)

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

    # ✅ Cleaned, WAN 2.2–accurate presets
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
# ---------- WAN 2.2 Image to Video (no scaling; expects pre-sized input) ----------
if HAVE_COMFY_API:
    class Wan22ImageToVideoMXD(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="Wan22ImageToVideoMXD",
                display_name="WAN 2.2 Image to Video MXD",
                category="conditioning/video_models",
                description="WAN 2.2 image to video without scaling or CLIP vision.",
                inputs=[
                    io.Conditioning.Input("positive"),
                    io.Conditioning.Input("negative"),
                    io.Vae.Input("vae"),
                    io.Int.Input("length", default=81, min=1, max=16384, step=4),
                    io.Int.Input("batch_size", default=1, min=1, max=4096),
                    io.Image.Input("start_image", optional=False),
                ],
                outputs=[
                    io.Conditioning.Output(display_name="positive"),
                    io.Conditioning.Output(display_name="negative"),
                    io.Latent.Output(display_name="latent"),
                ],
            )

        @classmethod
        def execute(cls, positive, negative, vae, length, batch_size, start_image) -> io.NodeOutput:
            if start_image is None:
                raise ValueError("start_image must be provided (already pre-sized).")

            frames_in, ih, iw, ch = start_image.shape
            frames_used = min(frames_in, length)
            t = ((length - 1) // 4) + 1

            latent = torch.zeros(
                [batch_size, 16, t, ih // 8, iw // 8],
                device=comfy.model_management.intermediate_device()
            )

            # create placeholder image tensor
            image = torch.ones(
                (length, ih, iw, ch),
                device=start_image.device,
                dtype=start_image.dtype
            ) * 0.5
            image[:frames_used] = start_image[:frames_used]

            # encode using VAE
            concat_latent_image = vae.encode(image[:, :, :, :3])

            # mask zeros out the frames used
            mask = torch.ones(
                (1, 1, t, concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
                device=image.device,
                dtype=image.dtype
            )
            mask[:, :, :((frames_used - 1) // 4) + 1] = 0.0

            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )

            out_latent = {"samples": latent}
            return io.NodeOutput(positive, negative, out_latent)

# ---- Canonical WAN 2.2 buckets ----
BUCKETS_480 = [(832,480), (480,832), (624,624)]      # 16:9, 9:16, 1:1
BUCKETS_720 = [(1280,720), (720,1280), (1024,1024)]  # 16:9, 9:16, 1:1
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

# ---------- WAN22_I2V_Image_Scaler_MXD ----------
# Adds a new “Safe Auto” mode for video extend workflows.
# Normal modes (Auto / 480p / 720p) behave exactly as before.
# “Safe Auto” adds passthrough + strict checks to prevent failures on WAN 2.2 extend.

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


def _resample_video_frames_to_fps(frames, in_fps, out_fps):
    """
    Resample a frame sequence to a target FPS using nearest-frame selection.
    Preserves clip duration approximately by dropping/duplicating frames,
    instead of only changing FPS metadata (which changes playback speed).
    Returns (frames_out, fps_out, changed).
    """
    if frames is None or frames.ndim != 4:
        raise ValueError("Expected frame tensor with shape [T,H,W,C].")

    if in_fps is None:
        raise ValueError("Input video FPS is missing; cannot force FPS safely.")

    in_fps = float(in_fps)
    out_fps = float(out_fps)
    if in_fps <= 0:
        raise ValueError(f"Invalid input FPS: {in_fps}")
    if out_fps <= 0:
        raise ValueError(f"Invalid target FPS: {out_fps}")

    if frames.shape[0] <= 1:
        return frames, float(out_fps), False

    if abs(in_fps - out_fps) < 1e-6:
        return frames, float(out_fps), False

    n_in = int(frames.shape[0])
    # Match the first/last frame span, then pick nearest frames on that timeline.
    n_out = max(1, int(round(((n_in - 1) * out_fps) / in_fps)) + 1)
    if n_out == n_in:
        # Frame count may stay the same for near-equal FPS; metadata still becomes exact.
        return frames, float(out_fps), False

    idx = torch.linspace(0, n_in - 1, steps=n_out, device=frames.device)
    idx = idx.round().to(dtype=torch.long)
    out = frames.index_select(0, idx)
    return out, float(out_fps), True


def _select_frames_start_end(frames, count=1, offset=1, mode="end"):
    total = int(frames.shape[0])
    if total <= 0:
        raise ValueError("No frames available for selection.")

    # Clamp offset and count
    offset = max(1, min(offset, total))
    count = max(1, min(count, total - offset + 1))

    if mode == "start":
        start_idx = offset - 1
        end_idx = start_idx + count
        selected = frames[start_idx:end_idx].clone()
    elif mode == "end":
        start_idx = max(0, total - offset - count + 1)
        end_idx = start_idx + count
        selected = frames[start_idx:end_idx].clone()
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'start' or 'end'.")

    return selected


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

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _pick_bucket(self, iw, ih, tier, crop_to_fit):
        is_squareish = _is_squareish(iw, ih)
        is_landscape = iw >= ih

        # --- Square handling ---
        if is_squareish:
            if tier == "720p":
                return (1024, 1024)
            else:
                return (624, 624)

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

    # -----------------------------
    # Main function
    # -----------------------------
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

        _, ih, iw, _ = image.shape

        # --- Safe Auto logic ---
        if tier == "Safe Auto":
            # passthrough if already WAN-safe
            if _wan22_is_valid_dim(iw, ih):
                return (image,)

            area = iw * ih
            area_480, area_720 = 832 * 480, 1280 * 720
            min_area, max_area = int(area_480 * 0.5), int(area_720 * 1.8)

            if area < min_area or area > max_area:
                size_label = "small" if area < min_area else "large"
                raise ValueError(
                    f"[WAN22_I2V_Image_Scaler_MXD] Input resolution {iw}x{ih} is too {size_label} for WAN 2.2 video buckets.\n"
                    "WAN 2.2 works best around:\n"
                    "  • 480p tier ≈ 832×480 (or 480×832)\n"
                    "  • 720p tier ≈ 1280×720 (or 720×1280)\n"
                    "  • Squares: 624×624 or 1024×1024\n\n"
                    "Please use a source closer to 480p/720p, or first process it "
                    "through your WAN 2.2 workflow. This ensures extend runs without mismatch."
                )
            # fallback to Auto scaling
            tier = "Auto"

        # --- Normal path (Auto / 480p / 720p) ---
        bw, bh = self._pick_bucket(iw, ih, tier, crop_to_fit)
        is_squareish = _is_squareish(iw, ih)

        if is_squareish:
            crop_to_fit = False

        if crop_to_fit:
            bw, bh = _safe_hw(_ceil16(bw), _ceil16(bh))
            out = _resize_then_center_crop(image, bw, bh)
        else:
            bw, bh = _safe_hw(_floor16(bw), _floor16(bh))
            out, _, _ = _resize_fit_inside(image, bw, bh)

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
    
# ---------- MXD Frames Select Start/End (from start or end of sequence) ----------
class Frames_Select_StartEnd_MXD:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of frames to select"
                }),
                "offset": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "How far into the video to start selection (from start or end)"
                }),
                "mode": (["start", "end"], {
                    "default": "end",
                    "tooltip": "Select frames from the start or end of the sequence"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "main"
    CATEGORY     = "MXD/images"

    def main(self, frames=None, count=1, offset=1, mode="end"):
        selected = _select_frames_start_end(frames, count=count, offset=offset, mode=mode)
        return (selected,)

        total = frames.shape[0]

        # Clamp offset and count
        offset = max(1, min(offset, total))
        count = max(1, min(count, total - offset + 1))

        if mode == "start":
            start_idx = offset - 1
            end_idx = start_idx + count
            selected = frames[start_idx:end_idx].clone()
        else:  # mode == "end"
            start_idx = max(0, total - offset - count + 1)
            end_idx = start_idx + count
            selected = frames[start_idx:end_idx].clone()

        return (selected,)
    
# ---------- MXD Frames Select Start/End (from start or end of sequence) ----------
class Frames_Remove_From_Start_MXD:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "count": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of frames to remove from the start"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "main"
    CATEGORY     = "MXD/images"

    def main(self, frames=None, count=10):
        # ✅ Skip the first `count` frames instead of keeping them
        frames_after = frames[count:].clone()
        return (frames_after,)


if HAVE_COMFY_API:
    class CombineVideos_MXD:
        """
        Combine two VIDEO inputs end-to-end (sequentially).
        """

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "front_video": ("VIDEO", {"tooltip": "The first video (plays first)"}),
                    "back_video": ("VIDEO", {"tooltip": "The second video (plays after the first)"}),
                },
            }

        RETURN_TYPES = ("VIDEO",)
        RETURN_NAMES = ("video",)
        FUNCTION = "combine"
        CATEGORY = "MXD/video"

        def combine(self, front_video, back_video):
            comp_a = front_video.get_components()
            comp_b = back_video.get_components()

            # Check frame rate consistency
            if comp_a.frame_rate != comp_b.frame_rate:
                raise ValueError(f"FPS mismatch: {comp_a.frame_rate} vs {comp_b.frame_rate}")

            # ✅ Correct way: concatenate frame tensors along batch/time dimension (dim=0)
            frames_a = torch.stack(comp_a.images) if isinstance(comp_a.images, list) else comp_a.images
            frames_b = torch.stack(comp_b.images) if isinstance(comp_b.images, list) else comp_b.images
            if frames_a.shape[1] != frames_b.shape[1] or frames_a.shape[2] != frames_b.shape[2]:
                raise ValueError(
                    "Resolution mismatch in CombineVideos_MXD: "
                    f"front_video={frames_a.shape[2]}x{frames_a.shape[1]}, "
                    f"back_video={frames_b.shape[2]}x{frames_b.shape[1]}. "
                    "Use 'WAN 2.2 Video Prep I2V MXD' before WAN generation so scaled base video and generated clip match."
                )
            combined_images = torch.cat([frames_a, frames_b], dim=0)

            # ✅ Combine audio sequentially
            combined_audio = None
            if comp_a.audio is not None or comp_b.audio is not None:
                def _extract_audio(audio_obj):
                    if audio_obj is None:
                        return None, None, None, None
                    if torch.is_tensor(audio_obj):
                        return audio_obj, None, "tensor", None
                    if isinstance(audio_obj, dict):
                        wave_key = "waveform" if "waveform" in audio_obj else ("samples" if "samples" in audio_obj else None)
                        if wave_key is None or not torch.is_tensor(audio_obj.get(wave_key)):
                            raise TypeError(f"Unsupported audio dict format. Keys: {list(audio_obj.keys())}")
                        return audio_obj[wave_key], audio_obj.get("sample_rate"), "dict", wave_key
                    waveform = getattr(audio_obj, "waveform", None)
                    sample_rate = getattr(audio_obj, "sample_rate", None)
                    if torch.is_tensor(waveform):
                        return waveform, sample_rate, "object", None
                    raise TypeError(f"Unsupported audio payload type: {type(audio_obj).__name__}")

                wave_a, sr_a, kind_a, wave_key_a = _extract_audio(comp_a.audio)
                wave_b, sr_b, kind_b, wave_key_b = _extract_audio(comp_b.audio)
                rank_a = wave_a.ndim if wave_a is not None else None
                rank_b = wave_b.ndim if wave_b is not None else None

                def _to_bct(w):
                    if w is None:
                        return None
                    if w.ndim == 1:
                        return w.unsqueeze(0).unsqueeze(0)  # [1,1,T]
                    if w.ndim == 2:
                        return w.unsqueeze(0)  # [1,C,T]
                    if w.ndim == 3:
                        return w  # [B,C,T]
                    raise ValueError(f"Unsupported audio tensor rank: {w.ndim}")

                wave_a = _to_bct(wave_a)
                wave_b = _to_bct(wave_b)

                if wave_a is None and wave_b is not None:
                    wave_a = torch.zeros((wave_b.shape[0], wave_b.shape[1], 0), dtype=wave_b.dtype, device=wave_b.device)
                if wave_b is None and wave_a is not None:
                    wave_b = torch.zeros((wave_a.shape[0], wave_a.shape[1], 0), dtype=wave_a.dtype, device=wave_a.device)

                if wave_a is not None and wave_b is not None:
                    if wave_a.shape[0] != wave_b.shape[0]:
                        if wave_a.shape[0] == 1:
                            wave_a = wave_a.expand(wave_b.shape[0], -1, -1)
                        elif wave_b.shape[0] == 1:
                            wave_b = wave_b.expand(wave_a.shape[0], -1, -1)
                        else:
                            raise ValueError(f"Audio batch mismatch: {wave_a.shape[0]} vs {wave_b.shape[0]}")

                    if wave_a.shape[1] != wave_b.shape[1]:
                        if wave_a.shape[1] == 1:
                            wave_a = wave_a.expand(-1, wave_b.shape[1], -1)
                        elif wave_b.shape[1] == 1:
                            wave_b = wave_b.expand(-1, wave_a.shape[1], -1)
                        else:
                            raise ValueError(f"Audio channel mismatch: {wave_a.shape[1]} vs {wave_b.shape[1]}")

                if sr_a is not None and sr_b is not None and sr_a != sr_b:
                    raise ValueError(f"Audio sample-rate mismatch: {sr_a} vs {sr_b}")

                combined_wave = torch.cat([wave_a, wave_b], dim=2)
                out_sr = sr_a if sr_a is not None else sr_b

                target_rank = rank_a if rank_a is not None else rank_b
                if target_rank == 1 and combined_wave.shape[0] == 1 and combined_wave.shape[1] == 1:
                    combined_wave = combined_wave.squeeze(0).squeeze(0)
                elif target_rank == 2 and combined_wave.shape[0] == 1:
                    combined_wave = combined_wave.squeeze(0)

                out_kind = kind_a if kind_a is not None else kind_b
                if out_kind == "dict":
                    out_key = wave_key_a if kind_a == "dict" else wave_key_b
                    combined_audio = {out_key or "waveform": combined_wave}
                    if out_sr is not None:
                        combined_audio["sample_rate"] = out_sr
                else:
                    combined_audio = combined_wave



            combined_video = VideoFromComponents(
                VideoComponents(
                    images=combined_images,
                    audio=combined_audio,
                    frame_rate=comp_a.frame_rate,
                )
            )

            return (combined_video,)

    class WAN22_I2V_Video_Prep_MXD:
        """
        Prepare a source video for iterative WAN 2.2 extension:
        - scale entire video using WAN bucket logic
        - output the scaled frame batch directly
        - keep default workflow simple for common use
        """
        CATEGORY = "MXD/video"
        FUNCTION = "prepare"
        RETURN_TYPES = ("VIDEO", "IMAGE", "FLOAT")
        RETURN_NAMES = ("scaled_video", "images", "fps")

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "video": ("VIDEO",),
                    "tier": (["Auto", "480p", "720p"], {"default": "Auto"}),
                    "crop_to_fit": ("BOOLEAN", {
                        "default": True,
                        "label_on": "Perfect Fit (Crops Edges)",
                        "label_off": "Closest Fit (No Crop)"
                    }),
                    "force_fps": ("BOOLEAN", {
                        "default": False,
                        "label_on": "Force FPS",
                        "label_off": "Keep Source FPS",
                        "tooltip": "When enabled, resample frames (drop/duplicate) and set exact target fps."
                    }),
                    "target_fps": ("INT", {
                        "default": 16,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Used when Force FPS is enabled. Output video fps will be set exactly to this value."
                    }),
                    "aspect_mode": (["Auto", "Tall", "Wide", "Square"], {
                        "default": "Auto",
                        "tooltip": "Auto picks wide/tall/square from the source. Use Square/Tall/Wide to force the target bucket shape."
                    }),
                },
            }

        def prepare(self, video, tier="Auto", crop_to_fit=True, force_fps=False, target_fps=16, aspect_mode="Auto"):
            comp = video.get_components()
            if isinstance(comp.images, list):
                if len(comp.images) == 0:
                    raise ValueError("[WAN22_I2V_Video_Prep_MXD] Input video has zero frames.")
                frames = torch.stack(comp.images)
            else:
                frames = comp.images

            if frames is None:
                raise ValueError("[WAN22_I2V_Video_Prep_MXD] Input video has no frames.")
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            if frames.ndim != 4:
                raise ValueError(f"[WAN22_I2V_Video_Prep_MXD] Unexpected frame tensor shape: {tuple(frames.shape)}")
            if frames.shape[0] <= 0:
                raise ValueError("[WAN22_I2V_Video_Prep_MXD] Input video has zero frames.")

            out_frame_rate = float(comp.frame_rate) if comp.frame_rate is not None else None
            if force_fps:
                frames, out_frame_rate, _ = _resample_video_frames_to_fps(
                    frames, comp.frame_rate, target_fps
                )

            # "Auto" in video prep uses the safer extend-friendly behavior.
            # Keep accepting legacy "Safe Auto" values from older saved workflows.
            internal_tier = "Safe Auto" if tier == "Auto" else tier
            scaled_frames, _, _, _ = _wan22_scale_image_core(
                frames,
                tier=internal_tier,
                crop_to_fit=crop_to_fit,
                aspect_mode=aspect_mode,
            )

            scaled_video = VideoFromComponents(
                VideoComponents(
                    images=scaled_frames,
                    audio=comp.audio,
                    frame_rate=out_frame_rate,
                )
            )

            fps = float(out_frame_rate) if out_frame_rate is not None else 0.0
            return (scaled_video, scaled_frames, fps)
    
    # ---------- Load Video MXD (video-only picker with refresh) ----------
    class LoadVideoMXD:
        """Load a video from /input with a refresh button (videos only)."""

        CATEGORY = "image/video"
        FUNCTION = "load"
        RETURN_TYPES = ("VIDEO", "STRING")
        RETURN_NAMES = ("video", "video_path")
        TITLE = "Load Video MXD"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "file": ("COMBO", {
                        # Only allow video uploads in the picker
                        "video_upload": True,
                        # Custom route that returns ONLY videos in /input
                        "remote": {
                            "route": "/mxd/videos/input",
                            "refresh_button": True,
                            "control_after_refresh": "first",
                        },
                    }),
                }
            }

        # --- helpers --------------------------------------------------------------

        @staticmethod
        def _resolve_video_path(file: str) -> str:
            """
            Try to resolve `file` in a backwards-compatible way:
            1. If it's an annotated path, let folder_paths handle it.
            2. Otherwise treat it as relative to the input directory.
            """
            # 1) Try annotated style (old workflows / uploads)
            try:
                return folder_paths.get_annotated_filepath(file)
            except Exception:
                pass

            # 2) Fall back to /input relative
            base = folder_paths.get_input_directory()
            candidate = os.path.join(base, file)
            if os.path.isfile(candidate):
                return candidate

            # If all else fails, just return what we got (will error later)
            return candidate

        @staticmethod
        def _is_video_file(path: str) -> bool:
            _, ext = os.path.splitext(path)
            return ext.lower() in VIDEO_EXTS

        # --- main function --------------------------------------------------------

        def load(self, file: str):
            video_path = self._resolve_video_path(file)

            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"[LoadVideoMXD] File not found: {video_path}")

            if not self._is_video_file(video_path):
                raise ValueError(f"[LoadVideoMXD] Not a video file: {video_path}")

            print(f"[LoadVideoMXD] Loaded exactly: {video_path}")
            return (VideoFromFile(video_path), video_path)

        # --- nice-to-haves --------------------------------------------------------

        @classmethod
        def IS_CHANGED(cls, file: str):
            try:
                p = cls._resolve_video_path(file)
                return os.path.getmtime(p)
            except Exception:
                return 0

        @classmethod
        def VALIDATE_INPUTS(cls, file: str):
            # First, try the annotated path (for backwards compat)
            if folder_paths.exists_annotated_filepath(file):
                resolved = folder_paths.get_annotated_filepath(file)
                if not cls._is_video_file(resolved):
                    return f"This node only accepts video files ({', '.join(sorted(VIDEO_EXTS))})."
                return True

            # Then, try treating it as /input-relative
            base = folder_paths.get_input_directory()
            candidate = os.path.join(base, file)
            if os.path.isfile(candidate):
                if not cls._is_video_file(candidate):
                    return f"This node only accepts video files ({', '.join(sorted(VIDEO_EXTS))})."
                return True

            return f"Invalid video file: {file}"
    
    # ---------- Save Video MXD (auto-increment clean filenames) ----------
    class SaveVideoMXD(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="SaveVideoMXD",
                display_name="Save Video MXD",
                category="image/video",
                description="Save a new version next to the original with clean counters.",
                inputs=[
                    io.Video.Input("video"),
                    io.String.Input("video_path"),
                    io.Combo.Input("save_to_outputs", options=[False, True], default=False),
                    io.Combo.Input("format", options=VideoContainer.as_input(), default="auto"),
                    io.Combo.Input("codec", options=VideoCodec.as_input(), default="auto"),
                ],
                outputs=[],
                hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
                is_output_node=True,
            )

        @classmethod
        def execute(cls, video: VideoInput, video_path: str, save_to_outputs: bool, format: str, codec: str):
            base_dir, base_filename = os.path.split(video_path)
            base_name, ext = os.path.splitext(base_filename)

            # 🧹 Clean trailing counters like "__001__002" → remove them all
            base_clean = re.sub(r'(__\d+)+$', '', base_name)

            # 🧮 Find the next available counter
            pattern = re.compile(rf"^{re.escape(base_clean)}__(\d+){re.escape(ext)}$")
            existing = [
                int(m.group(1))
                for f in os.listdir(base_dir)
                if (m := pattern.match(f))
            ]
            next_counter = max(existing, default=0) + 1

            new_filename = f"{base_clean}__{next_counter:03d}{ext}"
            save_path = os.path.join(base_dir, new_filename)

            # 💾 Metadata
            saved_metadata = None
            if not args.disable_metadata:
                metadata = {}
                if cls.hidden.extra_pnginfo is not None:
                    metadata.update(cls.hidden.extra_pnginfo)
                if cls.hidden.prompt is not None:
                    metadata["prompt"] = cls.hidden.prompt
                if metadata:
                    saved_metadata = metadata

            # 🚀 Save main copy
            video.save_to(save_path, format=format, codec=codec, metadata=saved_metadata)

            # 🪣 Optional copy to outputs folder
            if save_to_outputs:
                out_dir = folder_paths.get_output_directory()
                os.makedirs(out_dir, exist_ok=True)
                alt_path = os.path.join(out_dir, new_filename)
                video.save_to(alt_path, format=format, codec=codec, metadata=saved_metadata)
                print(f"[SaveVideoMXD] Also saved copy to outputs: {alt_path}")

            print(f"[SaveVideoMXD] Saved clean new version: {new_filename}")

            rel_folder = os.path.relpath(base_dir, folder_paths.get_output_directory())
            return io.NodeOutput(
                ui=ui.PreviewVideo([
                    ui.SavedResult(new_filename, rel_folder, io.FolderType.output)
                ])
            )

    class PreviewVideoMXD(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="PreviewVideoMXD",
                display_name="Preview Video MXD",
                category="image/video",
                description="Preview a video without saving output (optional pass-through).",
                inputs=[
                    io.Video.Input("input_video", tooltip="Video to preview."),
                ],
                outputs=[
                    io.Video.Output("output_video", tooltip="Passes the same video forward."),
                ],
                # Allow this node to run even when output_video is not connected.
                is_output_node=True,
            )

        @classmethod
        def execute(cls, input_video: VideoInput):
            # Save a temporary H264 file so ComfyUI has something to preview
            out_dir = os.path.join(folder_paths.get_output_directory(), "previews")
            os.makedirs(out_dir, exist_ok=True)

            preview_path = os.path.join(out_dir, "preview_temp.mp4")
            input_video.save_to(preview_path, format="mp4", codec="h264")

            # ✅ Return the raw video object (not a tuple)
            return io.NodeOutput(
                input_video,
                ui=ui.PreviewVideo([
                    ui.SavedResult("preview_temp.mp4", "previews", io.FolderType.output)
                ])
            )


class GroupVideoFramesMXD:
    CATEGORY = "MXD/Video"
    TITLE = "Group Video Frames (MXD)"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE_GROUPS",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "group_frames"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "group_size": ("INT", {"default": 81, "min": 1, "max": 5000, "step": 1}),
            }
        }

    def group_frames(self, frames, group_size):
        import math, torch

        all_frames = list(frames)
        total = len(all_frames)
        num_groups = math.ceil(total / group_size)
        grouped_tensors = []

        for i in range(num_groups):
            start = i * group_size
            end = min(start + group_size, total)
            group = all_frames[start:end]

            clean = []
            for f in group:
                # ✅ drop redundant singleton batch dim if present
                if f.ndim == 4 and f.shape[0] == 1:
                    f = f.squeeze(0)  # (H,W,C)
                # ✅ ensure shape (H,W,C)
                if f.ndim != 3:
                    print(f"[GroupVideoFramesMXD] weird frame shape {f.shape}")
                    continue
                clean.append(f)

            # ✅ stack back to (N,H,W,C)
            if len(clean) == 0:
                continue
            stacked = torch.stack(clean, dim=0)
            grouped_tensors.append(stacked)

        print(f"[GroupVideoFramesMXD] Split {total} frames into {len(grouped_tensors)} groups of up to {group_size}.")
        return (grouped_tensors,)

if HAVE_COMFY_API:
    class Wan22FirstLastImageToVideoMXD(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="Wan22FirstLastImageToVideoMXD",
                display_name="WAN 2.2 First & Last I2V MXD",
                category="conditioning/video_models",
                inputs=[
                    io.Conditioning.Input("positive"),
                    io.Conditioning.Input("negative"),
                    io.Vae.Input("vae"),
                    io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                    io.Int.Input("batch_size", default=1, min=1, max=4096),
                    io.Image.Input("start_image", optional=True),
                    io.Image.Input("end_image", optional=True),
                ],
                outputs=[
                    io.Conditioning.Output(display_name="positive"),
                    io.Conditioning.Output(display_name="negative"),
                    io.Latent.Output(display_name="latent"),
                ],
            )

        @classmethod
        def execute(cls, positive, negative, vae, length, batch_size, start_image=None, end_image=None) -> io.NodeOutput:
            spacial_scale = vae.spacial_compression_encode()

            # Assume incoming images are already pre-sized by upstream nodes.
            height, width = start_image.shape[1], start_image.shape[2] if start_image is not None else (vae.latent_channels * spacial_scale, vae.latent_channels * spacial_scale)

            latent = torch.zeros(
                [batch_size, vae.latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale],
                device=comfy.model_management.intermediate_device()
            )

            image = torch.ones((length, height, width, 3)) * 0.5
            mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

            if start_image is not None:
                image[:start_image.shape[0]] = start_image
                mask[:, :, :start_image.shape[0] + 3] = 0.0

            if end_image is not None:
                image[-end_image.shape[0]:] = end_image
                mask[:, :, -end_image.shape[0]:] = 0.0

            concat_latent_image = vae.encode(image[:, :, :, :3])
            mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

            out_latent = {"samples": latent}
            return io.NodeOutput(positive, negative, out_latent)


# ============================================================
# LTX Video Image Scaler MXD
# ============================================================
# Official LTX-2.3 rules (Lightricks model card + example workflows):
#   - Width & height must be divisible by 32; frame count must be 8n+1.
#   - The distilled two-stage workflow generates Stage 1 low-res, then the
#     ltx-2.3-spatial-upscaler-x2 doubles it (exactly 2x) for Stage 2.
#   - The one published two-stage resolution is Stage 1 960x544 -> 1920x1088.
#
# Tiers below are FINAL (Stage 2) sizes; Stage 1 is exactly half. Finals are
# kept /64 so Stage 1 stays /32 (the latent constraint). Only the 1080p 16:9
# row is officially published by Lightricks; the portrait/square rows and the
# 720p/576p tiers are /32-aligned siblings at the same pixel budget.
#
# Buckets (FINAL size, all /64) -> Stage 1 (half, all /32):
#   1080p: 1920x1088 / 1088x1920 / 1408x1408  (Stage 1: 960x544 / 544x960 / 704x704)
#   720p:  1280x704  / 704x1280  / 960x960     (Stage 1: 640x352 / 352x640 / 480x480)
#   576p:  1024x576  / 576x1024  / 768x768     (Stage 1: 512x288 / 288x512 / 384x384)
#
# Fit (no pad):  proportional resize <= target, /64 aligned.
# Crop (no pad): resize-to-cover then center-crop to exact bucket.
# Square images map to each tier's square bucket.
# ============================================================

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


# ---------- Node registration ----------
NODE_CLASS_MAPPINGS = {
    "SaveLatentMXD": SaveLatentMXD,
    "LoadLatent_WithParams": LoadLatent_WithParams,
    "LoadLatents_FromFolder_WithParams": LoadLatents_FromFolder_WithParams,
    "Wan2_2EmptyLatentImageMXD": Wan2_2EmptyLatentImageMXD,
    "wan22EmptyHunyuanLatentVideoMXD": wan22EmptyHunyuanLatentVideoMXD,
    "SaveLatent_I2V_MXD": SaveLatent_I2V_MXD,
    "LoadLatent_I2V_MXD": LoadLatent_I2V_MXD,
    "LoadLatents_FromFolder_I2V_MXD": LoadLatents_FromFolder_I2V_MXD,
    "SaveLatent_VACE22_MXD": SaveLatent_VACE22_MXD,
    "LoadLatent_VACE22_MXD": LoadLatent_VACE22_MXD,
    "LoadLatents_FromFolder_VACE22_MXD": LoadLatents_FromFolder_VACE22_MXD,
    "WAN22_I2V_Image_Scaler_MXD": WAN22_I2V_Image_Scaler_MXD,
    "LTX_Image_Scaler_MXD": LTX_Image_Scaler_MXD,
    "WAN22_I2V_Match_Resolution_MXD": WAN22_I2V_Match_Resolution_MXD,
    "Frames_Remove_From_Start_MXD": Frames_Remove_From_Start_MXD,
    "GroupVideoFramesMXD": GroupVideoFramesMXD,
    "Frames_Select_StartEnd_MXD": Frames_Select_StartEnd_MXD,
    "PadImageForOutpaintingMXD": PadImageForOutpaintingMXD,
}

if HAVE_COMFY_API:
    NODE_CLASS_MAPPINGS.update({
        "Wan22ImageToVideoMXD": Wan22ImageToVideoMXD,
        "WAN22_I2V_Video_Prep_MXD": WAN22_I2V_Video_Prep_MXD,
        "CombineVideos_MXD": CombineVideos_MXD,
        "LoadVideoMXD": LoadVideoMXD,
        "SaveVideoMXD": SaveVideoMXD,
        "PreviewVideoMXD": PreviewVideoMXD,
        "Wan22FirstLastImageToVideoMXD": Wan22FirstLastImageToVideoMXD,
    })

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveLatentMXD": "Save Latent MXD",
    "LoadLatent_WithParams": "Load Latent MXD",
    "LoadLatents_FromFolder_WithParams": "Load Latent Batch MXD",
    "Wan2_2EmptyLatentImageMXD": "Wan 2.2 Empty Latent Image MXD",
    "wan22EmptyHunyuanLatentVideoMXD": "WAN2.2 Empty Latent Video MXD",
    "SaveLatent_I2V_MXD": "Save Latent I2V MXD",
    "LoadLatent_I2V_MXD": "Load Latent I2V MXD",
    "LoadLatents_FromFolder_I2V_MXD": "Load Latent Batch I2V MXD",
    "SaveLatent_VACE22_MXD": "Save Latent Vace 2.2 MXD",
    "LoadLatent_VACE22_MXD": "Load Latent Vace 2.2 MXD",
    "LoadLatents_FromFolder_VACE22_MXD": "Load Latent Batch Vace 2.2 MXD",
    "WAN22_I2V_Image_Scaler_MXD": "Image Scaler Wan 2.2 I2V MXD",
    "LTX_Image_Scaler_MXD": "LTX Video Image Scaler MXD",
    "WAN22_I2V_Match_Resolution_MXD": "Match Resolution Wan 2.2 I2V MXD",
    "Frames_Remove_From_Start_MXD": "Remove Frames From Start MXD",
    "GroupVideoFramesMXD": "Group Video Frames MXD",
    "Frames_Select_StartEnd_MXD": "Select Frames MXD",
    "PadImageForOutpaintingMXD": "Pad Image for Outpainting MXD",
}

if HAVE_COMFY_API:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "Wan22ImageToVideoMXD": "Wan 2.2 Image to Video MXD",
        "WAN22_I2V_Video_Prep_MXD": "WAN 2.2 Video Prep I2V MXD",
        "CombineVideos_MXD": "Combine Videos MXD",
        "LoadVideoMXD": "Load Video MXD",
        "SaveVideoMXD": "Save Video MXD",
        "PreviewVideoMXD": "Preview Video MXD",
        "Wan22FirstLastImageToVideoMXD": "Wan 2.2 I2V First & Last Frame MXD",
    })

def _add_mxd_aliases(class_map, display_map):
    alias_sources = {}
    for key in list(class_map.keys()):
        if "MXD" in key.upper():
            continue
        alias = f"{key} MXD"
        if alias in class_map:
            continue
        class_map[alias] = class_map[key]
        alias_sources[alias] = key
    for alias, source in alias_sources.items():
        if alias not in display_map:
            display_map[alias] = display_map.get(source, alias)
    return alias_sources

_add_mxd_aliases(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
