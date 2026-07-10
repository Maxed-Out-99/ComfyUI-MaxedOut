"""Checkpoint load/save nodes.

Registered nodes:
  LoadCheckpointMXD  Load Checkpoint MXD (core loader + MXD info-icon UI)
  SaveCheckpointMXD  Save Checkpoint MXD (core saver with the FakeDevice fix)

Import-time side effect: replaces comfy.diffusers_convert.cat_tensors with a
version that materializes lazily-cast weights first (see comment below).
"""
import torch
import folder_paths
import comfy.sd
import comfy.diffusers_convert
from comfy_extras.nodes_model_merging import save_checkpoint


class LoadCheckpointMXD:
    DESCRIPTION = (
        "Loads a diffusion model checkpoint, same as the core Load Checkpoint node, "
        "with the MXD info-icon UI (CivitAI lookup, cached metadata, local notes)."
    )
    TITLE = "Load Checkpoint MXD"
    CATEGORY = "MXD/Loaders"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return out[:3]


# comfy's checkpoint saver builds the CLIP state dict via lazy "casting" params
# (comfy.model_patcher.LazyCastingParam / LazyCastingParamPiece) whose .device
# property returns a fake namedtuple ("FakeDevice") instead of a real torch.device,
# so it can defer materializing weights until safetensors actually calls .to() on
# them during serialization. comfy.diffusers_convert.cat_tensors() (used to merge
# the split q/k/v weights of SD1/SD2/SDXL CLIP text encoders back into a single
# in_proj_weight/in_proj_bias) reads tensor.device/tensor.dtype *before* that .to()
# call ever happens, so torch.empty() is handed the fake device object and crashes
# with: "empty() received an invalid combination of arguments - got (list,
# dtype=torch.dtype, device=FakeDevice)". This resolves any lazy tensor to its
# real, materialized form first (which is exactly what their own .to() override
# is designed to do) before touching .device/.dtype.
def _cat_tensors_fixed(tensors):
    resolved = [t if isinstance(t.device, torch.device) else t.to("cpu") for t in tensors]

    x = 0
    for t in resolved:
        x += t.shape[0]
    shape = [x] + list(resolved[0].shape)[1:]
    out = torch.empty(shape, device=resolved[0].device, dtype=resolved[0].dtype)

    x = 0
    for t in resolved:
        out[x:x + t.shape[0]] = t
        x += t.shape[0]

    return out


comfy.diffusers_convert.cat_tensors = _cat_tensors_fixed


class SaveCheckpointMXD:
    DESCRIPTION = (
        "Fixed drop-in replacement for the core Save Checkpoint node. The core node "
        "crashes with 'empty() received an invalid combination of arguments ... "
        "FakeDevice' when saving any model whose CLIP uses lazily-cast weights "
        "(e.g. fp8/quantized text encoders) because it reads .device/.dtype off an "
        "unmaterialized weight. This node patches that saving path."
    )
    TITLE = "Save Checkpoint MXD"
    CATEGORY = "MXD/Utils"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "filename_prefix": ("STRING", {"default": "checkpoints/ComfyUI"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    def save(self, model, clip, vae, filename_prefix, prompt=None, extra_pnginfo=None):
        save_checkpoint(
            model,
            clip=clip,
            vae=vae,
            filename_prefix=filename_prefix,
            output_dir=self.output_dir,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        return {}


NODE_CLASS_MAPPINGS = {
    "LoadCheckpointMXD": LoadCheckpointMXD,
    "SaveCheckpointMXD": SaveCheckpointMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCheckpointMXD": "Load Checkpoint MXD",
    "SaveCheckpointMXD": "Save Checkpoint MXD",
}
