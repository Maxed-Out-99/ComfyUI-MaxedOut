from __future__ import annotations
import os, re, glob, json, hashlib
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
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def save_only(self, samples, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):

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
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def save_only(self, samples, positive, negative, filename_prefix="I2V",
                  prompt=None, extra_pnginfo=None):

        # ---- save latent (.latent) ----
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

        latent_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.latent")

        payload = {
            "latent_tensor": samples["samples"].contiguous(),
            "latent_format_version_0": torch.tensor([]),
        }
        comfy.utils.save_torch_file(payload, latent_path, metadata=meta)

        # ---- save conditioning sidecar (.cond.pt) ----
        cond_path = latent_path.replace(".latent", ".cond.pt")
        torch.save({"positive": positive, "negative": negative}, cond_path)

        # No preview logic at all
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


def _extract_params_from_prompt_json(prompt_json: Dict[str, Any]) -> Tuple[str, str, int, float, str, str, int]:
    """
    Returns: (positive, negative, steps, cfg, sampler_name, scheduler, end_at_step)
    parsed from the saved Comfy prompt graph (KSamplerAdvanced only).
    """
    pos = ""
    neg = ""
    steps = 20
    cfg = 8.0
    sampler_name = ""
    scheduler = ""
    end_at_step = 0

    # unwrap if saved as {"prompt": {...}}
    graph = prompt_json.get("prompt", prompt_json) if isinstance(prompt_json, dict) else {}
    if not isinstance(graph, dict):
        return pos, neg, steps, cfg, sampler_name, scheduler, end_at_step

    # find the KSampler/KSamplerAdvanced node
    ks = None
    for _, v in graph.items():
        if "KSampler" in v.get("class_type", ""):  # matches KSamplerAdvanced too
            ks = v
            break
    if not ks:
        return pos, neg, steps, cfg, sampler_name, scheduler, end_at_step

    kin = ks.get("inputs", {})

    # follow links to CLIPTextEncode nodes for prompts
    def _as_node_id(x):
        return str(x[0]) if isinstance(x, (list, tuple)) and x else None

    def _text_from_clip(node_id):
        n = graph.get(str(node_id), {})
        if n.get("class_type") == "CLIPTextEncode":
            return str(n.get("inputs", {}).get("text", "")).strip()
        return ""

    pos = _text_from_clip(_as_node_id(kin.get("positive")))
    neg = _text_from_clip(_as_node_id(kin.get("negative")))

    # numeric params
    if "steps" in kin:
        try:
            steps = int(kin["steps"])
        except Exception:
            pass
    if "cfg" in kin:
        try:
            cfg = float(kin["cfg"])
        except Exception:
            pass
    if "end_at_step" in kin:
        try:
            end_at_step = int(kin["end_at_step"])
        except Exception:
            pass

    # strings (combos)
    sampler_name = str(kin.get("sampler_name", "")).strip()
    scheduler    = str(kin.get("scheduler", "")).strip()

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
        files.sort()
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
        pos, neg, steps, cfg, sampler_name, scheduler, end_at_step = _extract_params_from_prompt_json(prompt_json or {})

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
        subs = [""] + sorted([
            d for d in os.listdir(latents_root)
            if os.path.isdir(os.path.join(latents_root, d))
        ])

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
        files.sort()
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
            pos, neg, n_steps, cfg, sampler_name, scheduler, end_at_step = _extract_params_from_prompt_json(prompt_json or {})
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
        files.sort()
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
        subs = [""] + sorted([
            d for d in os.listdir(latents_root)
            if os.path.isdir(os.path.join(latents_root, d))
        ])

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
        files.sort()
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
                _extract_params_from_prompt_json(prompt_json or {})

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
BUCKETS_480 = [(832,480), (480,832), (624,624)]   # 16:9, 9:16, 1:1
BUCKETS_720 = [(1280,720), (720,1280)]            # 16:9, 9:16
SQUARE_TOL  = 0.03  # ±3% aspect-ratio tolerance counts as "square-ish"

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
    (624, 624), (720, 720),
}

def _wan22_is_valid_dim(w, h):
    return (w, h) in _WAN22_VALID_RES


def _wan22_pick_bucket(iw, ih, tier, crop_to_fit):
    is_squareish = _is_squareish(iw, ih)
    is_landscape = iw >= ih

    # --- Square handling ---
    if is_squareish:
        if tier == "720p":
            return (720, 720)
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


def _wan22_scale_image_core(image, tier="Auto", crop_to_fit=False):
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
                "  - Squares: 624x624 or 720x720\n\n"
                "Please use a source closer to 480p/720p, or first process it "
                "through your WAN 2.2 workflow. This ensures extend runs without mismatch."
            )
        # fallback to Auto scaling
        tier = "Auto"

    # --- Normal path (Auto / 480p / 720p) ---
    bw, bh = _wan22_pick_bucket(iw, ih, tier, crop_to_fit)
    is_squareish = _is_squareish(iw, ih)

    if is_squareish:
        crop_to_fit = False

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
        * 720p: ~square → 720×720
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
                return (720, 720)
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
    def scale(self, image, tier="Auto", crop_to_fit=False):
        # Keep legacy "Safe Auto" values from old workflows working, but expose only one Auto in UI.
        internal_tier = "Safe Auto" if tier == "Auto" else tier
        out, _, _, _ = _wan22_scale_image_core(image, tier=internal_tier, crop_to_fit=crop_to_fit)
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
                    "  • Squares: 624×624 or 720×720\n\n"
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
                "  - 624x624 / 720x720\n\n"
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
        - output start/end frames from the full scaled video
        - keep default workflow simple for common use
        """
        CATEGORY = "MXD/video"
        FUNCTION = "prepare"
        RETURN_TYPES = ("VIDEO", "IMAGE", "IMAGE", "INT", "INT", "FLOAT")
        RETURN_NAMES = ("scaled_video", "start_image", "end_image", "width", "height", "fps")

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
                    "fps_mode": (["none", "force"], {
                        "default": "none",
                        "tooltip": "none = keep source fps. force = resample frames (drop/duplicate) and set exact target fps."
                    }),
                    "target_fps": ("FLOAT", {
                        "default": 16.0,
                        "min": 0.001,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "Used when fps_mode=force. Output video fps will be set exactly to this value."
                    }),
                },
            }

        def prepare(self, video, tier="Auto", crop_to_fit=True, fps_mode="none", target_fps=16.0):
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
            if fps_mode == "force":
                frames, out_frame_rate, _ = _resample_video_frames_to_fps(
                    frames, comp.frame_rate, target_fps
                )

            # "Auto" in video prep uses the safer extend-friendly behavior.
            # Keep accepting legacy "Safe Auto" values from older saved workflows.
            internal_tier = "Safe Auto" if tier == "Auto" else tier
            scaled_frames, out_w, out_h, _ = _wan22_scale_image_core(
                frames, tier=internal_tier, crop_to_fit=crop_to_fit
            )

            start_image = scaled_frames[0:1].clone()
            end_image = scaled_frames[-1:].clone()

            scaled_video = VideoFromComponents(
                VideoComponents(
                    images=scaled_frames,
                    audio=comp.audio,
                    frame_rate=out_frame_rate,
                )
            )

            fps = float(out_frame_rate) if out_frame_rate is not None else 0.0
            return (scaled_video, start_image, end_image, out_w, out_h, fps)

    class WAN22_I2V_Video_Prep_Advanced_MXD:
        """
        Advanced variant of WAN22_I2V_Video_Prep_MXD with frame-selection controls.
        """
        CATEGORY = "MXD/video"
        FUNCTION = "prepare"
        RETURN_TYPES = ("VIDEO", "IMAGE", "IMAGE", "IMAGE", "INT", "INT", "FLOAT")
        RETURN_NAMES = ("scaled_video", "selected_frames", "start_image", "end_image", "width", "height", "fps")

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
                    "fps_mode": (["none", "force"], {
                        "default": "none",
                        "tooltip": "none = keep source fps. force = resample frames (drop/duplicate) and set exact target fps."
                    }),
                    "target_fps": ("FLOAT", {
                        "default": 16.0,
                        "min": 0.001,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "Used when fps_mode=force. Output video fps will be set exactly to this value."
                    }),
                    "mode": (["start", "end"], {"default": "end"}),
                    "count": ("INT", {"default": 1, "min": 1, "max": 10000}),
                    "offset": ("INT", {"default": 1, "min": 1, "max": 10000}),
                },
            }

        def prepare(self, video, tier="Auto", crop_to_fit=True, fps_mode="none", target_fps=16.0, mode="end", count=1, offset=1):
            comp = video.get_components()
            if isinstance(comp.images, list):
                if len(comp.images) == 0:
                    raise ValueError("[WAN22_I2V_Video_Prep_Advanced_MXD] Input video has zero frames.")
                frames = torch.stack(comp.images)
            else:
                frames = comp.images

            if frames is None:
                raise ValueError("[WAN22_I2V_Video_Prep_Advanced_MXD] Input video has no frames.")
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            if frames.ndim != 4:
                raise ValueError(f"[WAN22_I2V_Video_Prep_Advanced_MXD] Unexpected frame tensor shape: {tuple(frames.shape)}")
            if frames.shape[0] <= 0:
                raise ValueError("[WAN22_I2V_Video_Prep_Advanced_MXD] Input video has zero frames.")

            out_frame_rate = float(comp.frame_rate) if comp.frame_rate is not None else None
            if fps_mode == "force":
                frames, out_frame_rate, _ = _resample_video_frames_to_fps(
                    frames, comp.frame_rate, target_fps
                )

            # "Auto" in video prep uses the safer extend-friendly behavior.
            # Keep accepting legacy "Safe Auto" values from older saved workflows.
            internal_tier = "Safe Auto" if tier == "Auto" else tier
            scaled_frames, out_w, out_h, _ = _wan22_scale_image_core(
                frames, tier=internal_tier, crop_to_fit=crop_to_fit
            )

            selected_frames = _select_frames_start_end(
                scaled_frames, count=count, offset=offset, mode=mode
            )
            start_image = selected_frames[0:1].clone()
            end_image = selected_frames[-1:].clone()

            scaled_video = VideoFromComponents(
                VideoComponents(
                    images=scaled_frames,
                    audio=comp.audio,
                    frame_rate=out_frame_rate,
                )
            )

            fps = float(out_frame_rate) if out_frame_rate is not None else 0.0
            return (scaled_video, selected_frames, start_image, end_image, out_w, out_h, fps)
    
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
    "WAN22_I2V_Image_Scaler_MXD": WAN22_I2V_Image_Scaler_MXD,
    "WAN22_I2V_Match_Resolution_MXD": WAN22_I2V_Match_Resolution_MXD,
    "Frames_Remove_From_Start_MXD": Frames_Remove_From_Start_MXD,
    "GroupVideoFramesMXD": GroupVideoFramesMXD,
    "Frames_Select_StartEnd_MXD": Frames_Select_StartEnd_MXD,
}

if HAVE_COMFY_API:
    NODE_CLASS_MAPPINGS.update({
        "Wan22ImageToVideoMXD": Wan22ImageToVideoMXD,
        "WAN22_I2V_Video_Prep_MXD": WAN22_I2V_Video_Prep_MXD,
        "WAN22_I2V_Video_Prep_Advanced_MXD": WAN22_I2V_Video_Prep_Advanced_MXD,
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
    "WAN22_I2V_Image_Scaler_MXD": "Image Scaler Wan 2.2 I2V MXD",
    "WAN22_I2V_Match_Resolution_MXD": "Match Resolution Wan 2.2 I2V MXD",
    "Frames_Remove_From_Start_MXD": "Remove Frames From Start MXD",
    "GroupVideoFramesMXD": "Group Video Frames MXD",
    "Frames_Select_StartEnd_MXD": "Select Frames MXD",
}

if HAVE_COMFY_API:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "Wan22ImageToVideoMXD": "Wan 2.2 Image to Video MXD",
        "WAN22_I2V_Video_Prep_MXD": "WAN 2.2 Video Prep I2V MXD",
        "WAN22_I2V_Video_Prep_Advanced_MXD": "WAN 2.2 Video Prep I2V MXD Advanced",
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
