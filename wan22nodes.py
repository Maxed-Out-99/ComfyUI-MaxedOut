from __future__ import annotations
import os, re, glob, json, hashlib, uuid, math
from typing import Any, Dict, Tuple, Optional, List, Union

import torch
import numpy as np
from PIL import Image
from safetensors import safe_open

import folder_paths
import comfy.utils
import comfy.model_management
from comfy.cli_args import args
from nodes import KSamplerAdvanced
import node_helpers
import nodes

# Comfy API
from comfy_api.latest import io, ui
from comfy_api.input import VideoInput
from comfy_api.input_impl import VideoFromFile, VideoFromComponents
from comfy_api.util import VideoComponents, VideoContainer, VideoCodec

import imageio.v3 as iio

# ---------- SaveLatent (Comfy-only; saves into input/latents) ----------
class SaveLatentMXD:
    DESCRIPTION = """
    - Saves latents to `.latent` files under `input/latents/`.

    - Also decodes & saves preview images for quick inspection in the UI.

    - Preserves prompt & extra Comfy metadata inside the file.
    """
    TITLE = "Save Latent (with Preview)"
    CATEGORY = "MXD/Latents"
    RETURN_TYPES = ()  # only UI
    FUNCTION = "save_and_preview"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Latent tensor to save & preview."}),
                "vae": ("VAE", {"tooltip": "VAE used to decode preview images."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "Prefix for saved latent filename."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()  # only UI
    FUNCTION = "save_and_preview"
    OUTPUT_NODE = True
    CATEGORY = "MXD/Latents"

    def save_and_preview(self, samples, vae, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):

        # ---------- Save Latent ----------
        latents_dir = os.path.join(folder_paths.get_input_directory(), "latents")
        os.makedirs(latents_dir, exist_ok=True)

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, latents_dir
        )

        prompt_info = ""
        if prompt is not None:
            try:
                prompt_info = json.dumps(prompt)
            except Exception:
                pass

        metadata = None
        if not args.disable_metadata:
            metadata = {"prompt": prompt_info}
            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items():
                    try:
                        metadata[k] = json.dumps(v)
                    except Exception:
                        pass

        file = f"{filename}_{counter:05}_.latent"
        file = os.path.join(full_output_folder, file)

        output = {"latent_tensor": samples["samples"].contiguous(),
                  "latent_format_version_0": torch.tensor([])}

        comfy.utils.save_torch_file(output, file, metadata=metadata)

        # ---------- Decode + Save preview images ----------
        images = vae.decode(samples["samples"])
        if len(images.shape) == 5:  # merge video/batched latents
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        # Save previews to TEMP so the UI can locate them with type="temp"
        temp_dir = folder_paths.get_temp_directory()
        # build a preview name using the same helper so subfolder/counter are valid
        w, h = images[0].shape[1], images[0].shape[0]
        preview_prefix = filename_prefix + "_preview"
        full_temp_folder, preview_name, temp_counter, temp_subfolder, _ = folder_paths.get_save_image_path(
            preview_prefix, temp_dir, w, h
        )

        results = []
        for batch_number, image in enumerate(images):
            np_img = (255.0 * image.cpu().numpy())
            img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))

            fn_with_batch = preview_name.replace("%batch_num%", str(batch_number))
            preview_file = f"{fn_with_batch}_{temp_counter:05}_.png"
            img.save(os.path.join(full_temp_folder, preview_file), compress_level=1)

            results.append({
                "filename": preview_file,
                "subfolder": temp_subfolder,
                "type": "temp"  # 👈 matches temp_dir so UI can render
            })
            temp_counter += 1

        return {"ui": {"images": results}}

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
    DESCRIPTION = """
    - Loads a single latent file from `input/latents/`.

    - Extracts saved prompt text, sampler, steps, cfg, and scheduler if present.

    - Ensures compatibility with KSamplerAdvanced inputs.
    """
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
        from nodes import KSamplerAdvanced
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
        stem, _ = os.path.splitext(name)
        while stem and (stem[-1] == '_' or stem[-1] == '-' or stem[-1].isdigit()):
            stem = stem[:-1]
        return stem
    
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
    DESCRIPTION = """
    - Loads all `.latent` files from a chosen subfolder under `input/latents/`.

    - Returns lists of latents, prompts, and sampler settings for batch workflows.
    """
    TITLE = "Load Latents (Folder, With Params)"
    CATEGORY = "MXD/Latents"
    RETURN_TYPES  = ("FLOAT", "STRING", "STRING", "LATENT", "INT", "FLOAT", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES  = ("shift","positive","negative","samples","steps","cfg","sampler_name","scheduler","end_at_step","filename_prefix")
    OUTPUT_IS_LIST = (True,     True,      True,      True,    True,   True,           True,        True,          True,   True)
    FUNCTION = "load_batch"

    @classmethod
    def INPUT_TYPES(s):
        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        os.makedirs(latents_root, exist_ok=True)
        subs = [""] + sorted([d for d in os.listdir(latents_root)
                              if os.path.isdir(os.path.join(latents_root, d))])

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

        return {"required": {"subfolder": (subs, )}}

    def _coerce_enum(self, value, enum_values):
        try:
            return value if (enum_values and value in enum_values) else (enum_values[0] if enum_values else value)
        except Exception:
            return value

    def _strip_counter(self, name: str) -> str:
        stem, _ = os.path.splitext(name)
        while stem and (stem[-1] == '_' or stem[-1] == '-' or stem[-1].isdigit()):
            stem = stem[:-1]
        return stem
    
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


    def load_batch(self, subfolder):
        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        base = os.path.join(latents_root, subfolder) if subfolder else latents_root
        files = glob.glob(os.path.join(base, "**", "*.latent"), recursive=True)
        files.sort()
        if not files:
            raise RuntimeError(f"[LoadLatents_FromFolder_WithParams] No .latent files found in '{base}'.")

        shifts = []
        samples_list, positives, negatives = [], [], []
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
            pos, neg, n_steps, cfg, sampler_name, scheduler, end_at_step = _extract_params_from_prompt_json(prompt_json or {})

            sampler_name = self._coerce_enum(sampler_name, getattr(self.__class__, "_SAMPLERS_ENUM", ()))
            scheduler    = self._coerce_enum(scheduler,    getattr(self.__class__, "_SCHEDULERS_ENUM", ()))

            # folder part comes from the selected subfolder
            folder_part = subfolder if subfolder else ""
            # full basename with extension
            base_name = os.path.basename(path)
            # strip trailing counters like _00005
            clean_stem = self._strip_counter(base_name)
            # combine into prefix
            prefix = os.path.join(folder_part, clean_stem) if folder_part else clean_stem
            shift_val = self._extract_sd3_shift(meta, prompt_json)


            for sl in slices:
                shifts.append(float(shift_val))
                samples_list.append({"samples": sl})
                positives.append(pos)
                negatives.append(neg)
                steps_list.append(int(n_steps))
                cfgs.append(float(cfg))
                samplers.append(sampler_name)
                schedulers.append(scheduler)
                end_steps.append(int(end_at_step))
                filename_prefixes.append(prefix)

        n = len(samples_list)
        if (
            n == 0
            or len(shifts) != n
            or any(
                l != n
                for l in (
                    len(positives),
                    len(negatives),
                    len(steps_list),
                    len(cfgs),
                    len(samplers),
                    len(schedulers),
                    len(end_steps),
                    len(filename_prefixes),
                )
            )
        ):
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


# ---------- Empty latent image generator (for video nodes) ----------
class Wan2_2EmptyLatentImageMXD:
    """
    Utility node for WAN 2.2 workflows.
    Generates an empty latent tensor at common video-friendly resolutions.
    """

    DESCRIPTION = """
    - Creates an empty latent tensor sized for WAN 2.2 video generation.
    - Includes presets for 480p and 720p in multiple aspect ratios.
    - Enforces dimensions divisible by 8 for model compatibility.
    - Batch size supported (all latents share the same resolution).
    """
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

# ---------- I2V-specific latent save/load (sidecar conditioning; subclassed loader) ----------
class SaveLatent_I2V_MXD:
    """
    I2V-only saver that persists:
      • latent tensor  ->  .latent   (safetensors via comfy.utils.save_torch_file)
      • pos/neg CONDITIONING  ->  .cond.pt (torch.save; robust for nested tensors)
      • optional preview images to TEMP for UI
    """
    TITLE = "Save Latent I2V (with Conditioning)"
    CATEGORY = "MXD/Latents (I2V)"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_and_preview"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "High-noise latent to save for later low-noise finishing."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive CONDITIONING after WAN image→video."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative CONDITIONING after WAN image→video."}),
                "vae": ("VAE", {"tooltip": "Used to decode preview images for UI convenience."}),
                "filename_prefix": ("STRING", {"default": "I2V", "tooltip": "Prefix for saved files"}),
                "show_preview": ("BOOLEAN", {"default": False, "tooltip": "Show decoded preview images (slower)"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def save_and_preview(self, samples, positive, negative, vae, filename_prefix="I2V",
                         show_preview=False, prompt=None, extra_pnginfo=None):

        # ---- save latent (.latent) ----
        latents_dir = os.path.join(folder_paths.get_input_directory(), "latents")
        os.makedirs(latents_dir, exist_ok=True)

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, latents_dir
        )

        meta = None
        if not args.disable_metadata:
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

        latent_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.latent")

        payload = {
            "latent_tensor": samples["samples"].contiguous(),
            "latent_format_version_0": torch.tensor([]),
        }
        comfy.utils.save_torch_file(payload, latent_path, metadata=meta)

        # ---- save conditioning sidecar (.cond.pt) ----
        cond_path = latent_path.replace(".latent", ".cond.pt")
        torch.save({"positive": positive, "negative": negative}, cond_path)

        # ---- optional preview images ----
        if not show_preview:
            return {}  # skip VAE decode and preview generation

        images = vae.decode(samples["samples"])
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        temp_dir = folder_paths.get_temp_directory()
        w, h = images[0].shape[1], images[0].shape[0]
        preview_prefix = filename_prefix + "_preview"
        full_temp_folder, preview_name, temp_counter, temp_subfolder, _ = folder_paths.get_save_image_path(
            preview_prefix, temp_dir, w, h
        )

        results = []
        for b, image in enumerate(images):
            np_img = (255.0 * image.cpu().numpy())
            img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))
            fn_with_batch = preview_name.replace("%batch_num%", str(b))
            preview_file = f"{fn_with_batch}_{temp_counter:05}_.png"
            img.save(os.path.join(full_temp_folder, preview_file), compress_level=1)
            results.append({"filename": preview_file, "subfolder": temp_subfolder, "type": "temp"})
            temp_counter += 1

        return {"ui": {"images": results}}
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

    OUTPUT_IS_LIST = (True,) * 10  # same length for all outputs

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

    
# ---------- WAN 2.2 Image to Video (no scaling; expects pre-sized input) ----------
class WanImageToVideoMXD:
    """
    WAN 2.2 Image → Video (MXD)
    ⚙️ No scaling — expects pre-sized input.
    """

    TITLE = "WAN Image to Video MXD (No Scaling)"
    CATEGORY = "conditioning/video_models"
    DESCRIPTION = "Encodes a pre-scaled image for WAN 2.2 video conditioning."

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "start_image": ("IMAGE",),
            }
        }

    def run(self, positive, negative, vae, length, batch_size,
            clip_vision_output=None, start_image=None):

        if start_image is None:
            raise ValueError("start_image must be provided (already scaled).")

        # dims from the provided (pre-scaled) image
        frames_in, ih, iw, ch = start_image.shape
        frames_used = min(frames_in, length)
        t = ((length - 1) // 4) + 1

        # latent grid sized off the spatial dims and length-derived t
        latent = torch.zeros(
            [batch_size, 16, t, ih // 8, iw // 8],
            device=comfy.model_management.intermediate_device()
        )

        # ▶ Build a full-length (length, H, W, C) tensor and copy the given frames
        image = torch.ones(
            (length, ih, iw, ch),
            device=start_image.device,
            dtype=start_image.dtype
        ) * 0.5
        image[:frames_used] = start_image[:frames_used]

        # ▶ Encode the full-length tensor so its latent T matches t
        concat_latent_image = vae.encode(image[:, :, :, :3])

        # ▶ Make mask with T = t, and zero only the used frame-chunks
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

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        return (positive, negative, {"samples": latent})
    

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

# ---------- WAN 2.2 Image Scaler (no padding; fit or crop modes; square-aware) ----------
class WAN22_I2V_Image_Scaler_MXD:
    """
    MXD Image Scaler for WAN 2.2 (NO PADDING)
    - Modes: Auto / 480p / 720p
    - Fit (no pad): proportional resize ≤ target; returns resized dims.
    - Crop (no pad): resize-to-cover then center-crop to exact target.
    - Square handling:
        * Auto: ~square → 624×624
        * 480p: ~square → 624×624
        * 720p: ~square → 720×720  (explicitly supported)
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
                "crop_to_fit": ("BOOLEAN", {"default": False, "label_on": "Perfect Fit (Crops Edges)", "label_off": "Closest Fit (No Crop)"}),
            }
        }

    def _pick_bucket(self, iw, ih, tier, crop_to_fit):
        in_ar = _ar(iw, ih)
        is_squareish = _is_squareish(iw, ih)
        is_landscape = iw >= ih

        # --- Square images always map to fixed square buckets ---
        if is_squareish:
            if tier == "720p":
                return (720, 720)
            else:
                # both Auto and 480p get 624x624
                return (624, 624)

        # --- Non-square images below ---
        if tier == "720p":
            buckets = [(1280, 720)] if is_landscape else [(720, 1280)]
            return _closest_bucket(iw, ih, buckets, cover=crop_to_fit)

        if tier == "480p":
            buckets = [(832,480)] if is_landscape else [(480,832)]
            return _closest_bucket(iw, ih, buckets, cover=crop_to_fit)

        # Auto mode: pick 480 or 720 family depending on size
        buckets = BUCKETS_720 if max(iw, ih) > 720 else BUCKETS_480
        buckets = [(b[0], b[1]) for b in buckets if (b[0] > b[1]) == is_landscape]
        return _closest_bucket(iw, ih, buckets, cover=crop_to_fit)

    def scale(self, image, tier="Auto", crop_to_fit=False):
        _, ih, iw, _ = image.shape
        bw, bh = self._pick_bucket(iw, ih, tier, crop_to_fit)
        is_squareish = _is_squareish(iw, ih)
        if is_squareish:
            crop_to_fit = False
        bw, bh = _safe_hw(_ceil16(bw), _ceil16(bh)) if crop_to_fit else _safe_hw(_floor16(bw), _floor16(bh))
        if crop_to_fit:
            out = _resize_then_center_crop(image, bw, bh)
        else:
            out, _, _ = _resize_fit_inside(image, bw, bh)
        return (out,)
    
class Frames_Select_End_MXD:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "count": ("INT", {"default": 10, "min": 1, "max": 10000, "tooltip": "Number of frames to select from the end"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "main"
    CATEGORY     = "MXD/images"

    def main(self, frames=None, count=10):
        total = frames.shape[0]
        start = max(0, total - count)
        frames_end = frames[start:].clone()
        return (frames_end,)
    
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


class CombineVideos_MXD:
    """
    Combine two VIDEO inputs end-to-end (sequentially).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_a": ("VIDEO", {"tooltip": "The first video (plays first)"}),
                "video_b": ("VIDEO", {"tooltip": "The second video (plays after video_a)"}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "combine"
    CATEGORY = "MXD/video"

    def combine(self, video_a, video_b):
        comp_a = video_a.get_components()
        comp_b = video_b.get_components()

        # Check frame rate consistency
        if comp_a.frame_rate != comp_b.frame_rate:
            raise ValueError(f"FPS mismatch: {comp_a.frame_rate} vs {comp_b.frame_rate}")

        # ✅ Correct way: concatenate frame tensors along batch/time dimension (dim=0)
        frames_a = torch.stack(comp_a.images) if isinstance(comp_a.images, list) else comp_a.images
        frames_b = torch.stack(comp_b.images) if isinstance(comp_b.images, list) else comp_b.images
        combined_images = torch.cat([frames_a, frames_b], dim=0)

        # ✅ Combine audio sequentially
        combined_audio = None
        if comp_a.audio is not None or comp_b.audio is not None:
            audio_a = comp_a.audio if comp_a.audio is not None else torch.zeros((1, 0))
            audio_b = comp_b.audio if comp_b.audio is not None else torch.zeros((1, 0))
            combined_audio = torch.cat([audio_a, audio_b], dim=1)



        combined_video = VideoFromComponents(
            VideoComponents(
                images=combined_images,
                audio=combined_audio,
                frame_rate=comp_a.frame_rate,
            )
        )

        return (combined_video,)
    

class LoadVideoMXD(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return io.Schema(
            node_id="LoadVideoMXD",
            display_name="Load Video MXD (Auto-Reload)",
            category="image/video",
            description="Always reloads the newest video with the same base name each time you run.",
            inputs=[
                io.Combo.Input(
                    "file",
                    options=sorted(files),
                    upload=io.UploadType.video,
                    tooltip="Pick or upload the base video. The newest version will be auto-loaded on next run."
                ),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("video_path"),
            ],
        )

    @classmethod
    def _find_latest(cls, base_path: str) -> str:
        base_dir, base_filename = os.path.split(base_path)
        base_name, _ = os.path.splitext(base_filename)

        related = [
            os.path.join(base_dir, f)
            for f in os.listdir(base_dir)
            if f.startswith(base_name) and os.path.isfile(os.path.join(base_dir, f))
        ]
        if not related:
            return base_path
        newest = max(related, key=os.path.getmtime)
        return newest

    @classmethod
    def execute(cls, file):
        video_path = folder_paths.get_annotated_filepath(file)
        latest = cls._find_latest(video_path)
        if latest != video_path:
            print(f"[LoadVideoMXD] Reloading latest: {os.path.basename(latest)}")
        return io.NodeOutput(VideoFromFile(latest), latest)

    # 🔥 This is the missing piece — forces reload whenever a newer file exists
    @classmethod
    def fingerprint_inputs(cls, file):
        video_path = folder_paths.get_annotated_filepath(file)
        latest = cls._find_latest(video_path)
        return os.path.getmtime(latest)


class LoadVideoMXD(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return io.Schema(
            node_id="LoadVideoMXD",
            display_name="Load Video MXD",
            category="image/video",
            description="Always reloads the newest video with the same base name each time you run.",
            inputs=[
                io.Combo.Input(
                    "file",
                    options=sorted(files),
                    upload=io.UploadType.video,
                    tooltip="Pick or upload the base video. The newest version will be auto-loaded on next run."
                ),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("video_path"),
            ],
        )

    @classmethod
    def _find_latest(cls, base_path: str) -> str:
        base_dir, base_filename = os.path.split(base_path)
        base_name, _ = os.path.splitext(base_filename)
        related = [
            os.path.join(base_dir, f)
            for f in os.listdir(base_dir)
            if f.startswith(base_name) and os.path.isfile(os.path.join(base_dir, f))
        ]
        if not related:
            return base_path
        return max(related, key=os.path.getmtime)

    @classmethod
    def execute(cls, file):
        video_path = folder_paths.get_annotated_filepath(file)
        latest = cls._find_latest(video_path)
        if latest != video_path:
            print(f"[LoadVideoMXD] Reloading latest: {os.path.basename(latest)}")
        return io.NodeOutput(VideoFromFile(latest), latest)

    @classmethod
    def fingerprint_inputs(cls, file):
        video_path = folder_paths.get_annotated_filepath(file)
        latest = cls._find_latest(video_path)
        return os.path.getmtime(latest)


class SaveVideoMXD(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideoMXD",
            display_name="Save Video MXD",
            category="image/video",
            description="Saves a new version of the video next to the original, auto-incrementing filenames cleanly.",
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

        # 🧮 Now find the next available counter
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
        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(new_filename, rel_folder, io.FolderType.output)]))
    
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

import os
import torch
from comfy_api.latest import io, ui
from comfy_api.util import VideoComponents, VideoContainer, VideoCodec
from comfy_api.input_impl import VideoFromFile, VideoFromComponents
import folder_paths
from comfy.cli_args import args


class SaveAndMergeWhenComplete_MXD(io.ComfyNode):
    """
    Saves batched video parts and merges them automatically once all expected parts are present.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveAndMergeWhenComplete_MXD",
            display_name="Save & Merge When Complete (MXD)",
            category="MXD/video",
            description="Saves each incoming video part, and merges all parts once the expected count is reached.",
            inputs=[
                io.Video.Input("video", tooltip="Video to save (one per batch item)."),
                io.String.Input(
                    "folder_name",
                    default="temp_batch_merge",
                    tooltip="Subfolder under output/video/ to save temporary parts."
                ),
                io.Int.Input(
                    "expected_parts",
                    default=2,
                    min=1,
                    max=9999,
                    tooltip="Total number of parts to wait for before merging."
                ),
                io.Combo.Input(
                    "format",
                    options=VideoContainer.as_input(),
                    default="auto",
                    tooltip="Container format for the saved videos."
                ),
                io.Combo.Input(
                    "codec",
                    options=VideoCodec.as_input(),
                    default="auto",
                    tooltip="Codec to use for the video."
                ),
            ],
            outputs=[
                io.String.Output("final_video_path", tooltip="Full path of the merged video (only once complete)."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=False,
        )

    @classmethod
    def execute(cls, video, folder_name, expected_parts, format, codec) -> io.NodeOutput:
        output_dir = os.path.join(folder_paths.get_output_directory(), "video", folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # --- Save incoming video part ---
        part_index = len([f for f in os.listdir(output_dir) if f.endswith(".mp4")])
        part_path = os.path.join(output_dir, f"part_{part_index+1:03d}.mp4")

        # --- Metadata (same as SaveVideo) ---
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if metadata:
                saved_metadata = metadata

        video.save_to(part_path, format=format, codec=codec, metadata=saved_metadata)
        print(f"[SaveAndMergeWhenComplete_MXD] 💾 Saved part {part_index+1}/{expected_parts} → {part_path}")

        # --- Check how many parts exist ---
        part_files = sorted([
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.lower().endswith(".mp4")
        ])

        if len(part_files) < expected_parts:
            print(f"[SaveAndMergeWhenComplete_MXD] Waiting for all parts ({len(part_files)}/{expected_parts})...")
            return io.NodeOutput(final_video_path="")  # Not ready yet

        # ✅ All parts present → merge them
        print(f"[SaveAndMergeWhenComplete_MXD] All {expected_parts} parts found. Starting merge...")

        comp_ref = VideoFromFile(part_files[0]).get_components()
        all_frames = []
        all_audio = []
        frame_rate = comp_ref.frame_rate

        for path in part_files:
            vid = VideoFromFile(path)
            comp = vid.get_components()
            frames = torch.stack(comp.images) if isinstance(comp.images, list) else comp.images
            all_frames.append(frames)
            if comp.audio is not None:
                all_audio.append(comp.audio)

        merged_frames = torch.cat(all_frames, dim=0)
        merged_audio = torch.cat(all_audio, dim=1) if all_audio else None

        combined_video = VideoFromComponents(
            VideoComponents(images=merged_frames, audio=merged_audio, frame_rate=frame_rate)
        )

        final_path = os.path.join(output_dir, "merged_final.mp4")
        combined_video.save_to(final_path, format=format, codec=codec, metadata=saved_metadata)

        print(f"[SaveAndMergeWhenComplete_MXD] ✅ Merged {expected_parts} parts → {final_path}")

        return io.NodeOutput(final_path)



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
    "WanImageToVideoMXD": WanImageToVideoMXD,
    "WAN22_I2V_Image_Scaler_MXD": WAN22_I2V_Image_Scaler_MXD,
    "Frames_Select_End_MXD": Frames_Select_End_MXD,
    "Frames_Remove_From_Start_MXD": Frames_Remove_From_Start_MXD,
    "CombineVideos_MXD": CombineVideos_MXD,
    "LoadVideoMXD": LoadVideoMXD,
    "SaveVideoMXD": SaveVideoMXD,
    "GroupVideoFramesMXD": GroupVideoFramesMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveLatentMXD": "Save Latent MXD",
    "LoadLatent_WithParams": "Load Latent MXD",
    "LoadLatents_FromFolder_WithParams": "Load Latent Batch MXD",
    "Wan2_2EmptyLatentImageMXD": "Wan 2.2 Empty Latent Image MXD",
    "wan22EmptyHunyuanLatentVideoMXD": "WAN2.2 Empty Latent Video MXD",
    "SaveLatent_I2V_MXD": "Save Latent I2V MXD",
    "LoadLatent_I2V_MXD": "Load Latent I2V MXD",
    "LoadLatents_FromFolder_I2V_MXD": "Load Latent Batch I2V MXD",
    "WanImageToVideoMXD": "WAN Image to Video MXD",
    "WAN22_I2V_Image_Scaler_MXD": "WAN 2.2 I2V Image Scaler MXD",
    "Frames_Select_End_MXD": "Frames Select End MXD",
    "Frames_Remove_From_Start_MXD": "Frames Remove From Start MXD",
    "CombineVideos_MXD": "Combine Videos MXD",
    "LoadVideoMXD": "Load Video MXD",
    "SaveVideoMXD": "Save Video MXD",
    "GroupVideoFramesMXD": "Group Video Frames MXD",
}
