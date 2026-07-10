# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Forked from https://github.com/city96/ComfyUI-GGUF
# Modified by Maxed-Out-99
#
# Not vendored verbatim (see UPSTREAM.md) — this is our own unified
# loader layer on top of the vendored dequant/loader/ops engine files.
import logging
import collections
import inspect

import comfy.sd
import comfy.lora
import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import folder_paths
import nodes
import torch

from .ops import GGMLOps, move_patch_to_device
from .loader import gguf_sd_loader, gguf_clip_loader
from .dequant import is_quantized, is_torch_compatible

def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")

# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            # TODO: do we ever have legitimate duplicate patches? (i.e. patch on top of patched weight)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = comfy.lora.calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    def pin_weight_to_device(self, key):
        op_key = key.rsplit('.', 1)[0]
        if not self.mmap_released and op_key in self.named_modules_to_munmap:
            # TODO: possible to OOM, find better way to detach
            self.named_modules_to_munmap[op_key].to(self.load_device).to(self.offload_device)
            del self.named_modules_to_munmap[op_key]
        super().pin_weight_to_device(key)

    mmap_released = False
    named_modules_to_munmap = {}

    def load(self, *args, force_patch_weights=False, **kwargs):
        if not self.mmap_released:
            self.named_modules_to_munmap = dict(self.model.named_modules())

        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.named_modules_to_munmap.items():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked and self.load_device != self.offload_device:
                logging.info(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True
            self.named_modules_to_munmap = {}

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        # GGUF specific clone values below
        n.patch_on_device = getattr(self, "patch_on_device", False)
        n.mmap_released = getattr(self, "mmap_released", False)
        if src_cls != GGUFModelPatcher:
            n.size = 0 # force recalc
        return n

class UNETLoaderUnified:
    @classmethod
    def INPUT_TYPES(s):
        # Combine both lists: standard + gguf
        model_list = folder_paths.get_filename_list("diffusion_models") + folder_paths.get_filename_list("unet_gguf")
        return {
            "required": {
                "unet_name": (sorted(set(model_list)),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/loaders"
    TITLE = "Smart Model Loader MXD"

    def load_unet(self, unet_name):
        if unet_name.endswith(".gguf"):
            return self.load_gguf_unet(unet_name)
        else:
            return self.load_standard_unet(unet_name)

    def load_standard_unet(self, unet_name):
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path)
        return (model,)

    def load_gguf_unet(self, unet_name):
        unet_path = folder_paths.get_full_path("unet", unet_name)
        ops = GGMLOps()

        # Load state dict from GGUF
        loaded = gguf_sd_loader(unet_path)
        if loaded is None:
            raise RuntimeError(f"Failed to load GGUF model: {unet_path}")

        if isinstance(loaded, tuple) and len(loaded) == 2:
            sd, extra = loaded
        else:
            sd, extra = loaded, {}

        kwargs = {}
        valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
        if "metadata" in valid_params:
            kwargs["metadata"] = extra.get("metadata", {})

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}, **kwargs
        )
        if model is None:
            raise RuntimeError(f"Could not detect GGUF model type for: {unet_path}")
        model = GGUFModelPatcher.clone(model)
        return (model,)

##########################################################################################################################################

class SmartCLIPLoaderBase:
    def get_clip_paths(self, *clip_names):
        return [folder_paths.get_full_path("clip", name) for name in clip_names]

    def load_clip_models(self, clip_paths, clip_type):
        if all(path is not None and not path.endswith(".gguf") for path in clip_paths):
            clip = comfy.sd.load_clip(
                ckpt_paths=clip_paths,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                model_options={},
            )
            return (clip,)

        clip_data = []
        use_ggml_ops = False

        for path in clip_paths:
            if path.endswith(".gguf"):
                sd = gguf_clip_loader(path)
                use_ggml_ops = True
            else:
                sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
                sd, _metadata = comfy.utils.convert_old_quants(sd, model_prefix="", metadata=metadata)
            clip_data.append(sd)

        model_options = {
            "initial_device": comfy.model_management.text_encoder_offload_device(),
        }
        if use_ggml_ops:
            model_options["custom_operations"] = GGMLOps

        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=clip_type,
            state_dicts=clip_data,
            model_options=model_options,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        if use_ggml_ops and hasattr(clip, "patcher") and clip.patcher is not None:
            clip.patcher = GGUFModelPatcher.clone(clip.patcher)

        return (clip,)

##########################################################################################################################################

class DualCLIPLoaderUnified(SmartCLIPLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        clip_files = folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip_gguf")
        clip_options = (sorted(set(clip_files)),)
        return {
            "required": {
                "clip_name1": clip_options,
                "clip_name2": clip_options,
                "type": nodes.DualCLIPLoader.INPUT_TYPES()["required"]["type"],
                }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"
    TITLE = "Smart Dual CLIP Loader MXD"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip_paths = self.get_clip_paths(clip_name1, clip_name2)
        return self.load_clip_models(clip_paths, clip_type)

##########################################################################################################################################

class TripleCLIPLoaderUnified(SmartCLIPLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        clip_files = folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip_gguf")
        clip_options = (sorted(set(clip_files)),)
        return {
            "required": {
                "clip_name1": clip_options,
                "clip_name2": clip_options,
                "clip_name3": clip_options,
                "type": nodes.CLIPLoader.INPUT_TYPES()["required"]["type"],
                }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"
    TITLE = "Smart Triple CLIP Loader MXD"

    def load_clip(self, clip_name1, clip_name2, clip_name3, type):
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip_paths = self.get_clip_paths(clip_name1, clip_name2, clip_name3)
        return self.load_clip_models(clip_paths, clip_type)

##########################################################################################################################################


class QuadrupleCLIPLoaderUnified(SmartCLIPLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        clip_files = folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip_gguf")
        clip_options = (sorted(set(clip_files)),)
        return {
            "required": {
                "clip_name1": clip_options,
                "clip_name2": clip_options,
                "clip_name3": clip_options,
                "clip_name4": clip_options,
                "type": nodes.CLIPLoader.INPUT_TYPES()["required"]["type"],
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"
    TITLE = "Smart Quad CLIP Loader MXD"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, type):
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip_paths = self.get_clip_paths(clip_name1, clip_name2, clip_name3, clip_name4)
        return self.load_clip_models(clip_paths, clip_type)

##########################################################################################################################################

class CLIPLoaderUnified(SmartCLIPLoaderBase):
    @classmethod
    def INPUT_TYPES(cls):
        clip_files = folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip_gguf")
        clip_options = (sorted(set(clip_files)),)
        return {
            "required": {
                "clip_name1": clip_options,
                "type": nodes.CLIPLoader.INPUT_TYPES()["required"]["type"],
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"
    TITLE = "Smart CLIP Loader MXD"

    def load_clip(self, clip_name1, type):
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return self.load_clip_models(self.get_clip_paths(clip_name1), clip_type)

##########################################################################################################################################

NODE_CLASS_MAPPINGS = {
    # Keep the original keys for backward compatibility with existing ComfyUI workflows.
    "UNETLoaderUnified": UNETLoaderUnified,
    "CLIPLoaderUnified": CLIPLoaderUnified,
    "DualCLIPLoaderUnified": DualCLIPLoaderUnified,
    "TripleCLIPLoaderUnified": TripleCLIPLoaderUnified,
    "QuadrupleCLIPLoaderUnified": QuadrupleCLIPLoaderUnified,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UNETLoaderUnified": "Smart Model Loader MXD",
    "CLIPLoaderUnified": "Smart CLIP Loader MXD",
    "DualCLIPLoaderUnified": "Smart Dual CLIP Loader MXD",
    "TripleCLIPLoaderUnified": "Smart Triple CLIP Loader MXD",
    "QuadrupleCLIPLoaderUnified": "Smart Quad CLIP Loader MXD",
}
