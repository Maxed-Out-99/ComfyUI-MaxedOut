import os, json, hashlib, glob
from typing import Any, Dict, Tuple, Optional, List, Union
import torch
from safetensors import safe_open
from comfy.cli_args import args
import numpy as np
from PIL import Image
import folder_paths
import comfy.utils
from nodes import KSamplerAdvanced


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
    RETURN_TYPES = ("LATENT", "STRING", "STRING", "INT", "FLOAT", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("samples", "positive", "negative", "steps", "cfg", "sampler_name", "scheduler", "end_at_step", "filename_prefix")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(s):
        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        os.makedirs(latents_root, exist_ok=True)

        files = glob.glob(os.path.join(latents_root, "**", "*.latent"), recursive=True)
        files.sort()
        options = [os.path.relpath(f, folder_paths.get_input_directory()).replace(os.sep, "/") for f in files]

        # live enums from KSamplerAdvanced so values wire cleanly
        from nodes import KSamplerAdvanced
        ks_inputs = KSamplerAdvanced.INPUT_TYPES().get("required", {})
        samplers_enum   = ks_inputs.get("sampler_name", ("STRING",))[0]
        schedulers_enum = ks_inputs.get("scheduler", ("STRING",))[0]

        # overwrite with live enums
        s.RETURN_TYPES = (
            "LATENT",
            "STRING",
            "STRING",
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

    def load(self, latent):
        latent_path = folder_paths.get_annotated_filepath(latent)
        sample_dict, meta, _ = _load_latent_file(latent_path)
        t = sample_dict["samples"]

        if isinstance(t, torch.Tensor) and t.dim() >= 4 and t.size(0) > 1:
            # for safety, only take first slice (multi-batch handling is folder loader’s job)
            samples = {"samples": t[0:1].contiguous()}
        elif isinstance(t, torch.Tensor) and t.dim() >= 4 and t.size(0) == 1:
            samples = {"samples": t}
        else:
            samples = {"samples": t.unsqueeze(0)}

        prompt_json = _safe_json_loads(meta.get("prompt"))
        pos, neg, steps, cfg, sampler_name, scheduler, end_at_step = _extract_params_from_prompt_json(prompt_json or {})

        sampler_name = self._coerce_enum(sampler_name, getattr(self.__class__, "_SAMPLERS_ENUM", ()))
        scheduler    = self._coerce_enum(scheduler,    getattr(self.__class__, "_SCHEDULERS_ENUM", ()))

        folder_part = os.path.dirname(latent).replace("\\", "/")
        base_name   = os.path.basename(latent_path)
        clean_stem  = self._strip_counter(base_name)
        prefix      = os.path.join(folder_part, clean_stem) if folder_part else clean_stem

        return (samples, pos, neg, int(steps), float(cfg), sampler_name, scheduler, int(end_at_step), prefix)

    @classmethod
    def IS_CHANGED(s, latent):
        p = folder_paths.get_annotated_filepath(latent)
        m = hashlib.sha256()
        with open(p, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, latent):
        if not folder_paths.exists_annotated_filepath(latent):
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
    RETURN_TYPES = ("LATENT", "STRING", "STRING", "INT", "FLOAT", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES  = ("samples", "positive", "negative", "steps", "cfg", "sampler_name", "scheduler", "end_at_step", "filename_prefix")
    OUTPUT_IS_LIST = (True,     True,      True,      True,    True,   True,           True,        True,          True)
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
            "LATENT",
            "STRING",
            "STRING",
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

    def load_batch(self, subfolder):
        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        base = os.path.join(latents_root, subfolder) if subfolder else latents_root
        files = glob.glob(os.path.join(base, "**", "*.latent"), recursive=True)
        files.sort()
        if not files:
            raise RuntimeError(f"[LoadLatents_FromFolder_WithParams] No .latent files found in '{base}'.")

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


            for sl in slices:
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
        lens = [n, len(positives), len(negatives), len(steps_list), len(cfgs), len(samplers), len(schedulers), len(end_steps), len(filename_prefixes)]
        if n == 0 or any(l != n for l in lens):
            raise RuntimeError("[LoadLatents_FromFolder_WithParams] Internal length mismatch.")

        return (samples_list, positives, negatives, steps_list, cfgs, samplers, schedulers, end_steps, filename_prefixes)
    

# ---------- Node registration ----------
NODE_CLASS_MAPPINGS = {
    "SaveLatentMXD": SaveLatentMXD,
    "LoadLatent_WithParams": LoadLatent_WithParams,
    "LoadLatents_FromFolder_WithParams": LoadLatents_FromFolder_WithParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveLatentMXD": "Save Latent MXD",
    "LoadLatent_WithParams": "Load Latent MXD",
    "LoadLatents_FromFolder_WithParams": "Load Latent Batch MXD",
}
