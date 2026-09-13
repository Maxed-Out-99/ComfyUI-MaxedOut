"""MXD source patch backed by the installed ComfyUI-Krea2Edit node."""

import sys

import comfy.patcher_extension
import nodes as comfy_nodes


def _source_patch_class():
    # Resolve after custom nodes have loaded, regardless of pack import order.
    node_class = comfy_nodes.NODE_CLASS_MAPPINGS.get("Krea2EditModelPatch")
    if node_class is None:
        raise RuntimeError(
            "Krea2 Edit (source patch) MXD requires ComfyUI-Krea2Edit. "
            "Install or enable it and restart ComfyUI."
        )
    return node_class


class Krea2EditModelPatchMXD:
    TITLE = "Krea2 Edit (source patch) MXD"
    CATEGORY = "MXD/Krea"
    DESCRIPTION = (
        "Adds the Krea2 edit source-preservation path using the installed "
        "ComfyUI-Krea2Edit source patch. Requires ComfyUI-Krea2Edit."
    )
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "source_image": ("IMAGE",),
                "vae": ("VAE",),
                "target_latent": ("LATENT", {"tooltip": "Connect the same latent that feeds the sampler so sources are encoded at the target resolution before sampling."}),
                "ref_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                        "tooltip": "Reference attention strength for the last source image. 1.0 = unchanged."}),
                "ref_boost_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                          "tooltip": "Reference attention strength for the first image in two-reference workflows. No effect with one image."}),
                "fit_mode": (["fit", "crop (legacy)"], {"default": "fit",
                             "tooltip": "Fit preserves the source aspect ratio at a centered offset; crop (legacy) center-crops to the target aspect ratio."}),
            },
            "optional": {
                "source_image_b": ("IMAGE", {"tooltip": "Optional second reference (subject); the first image supplies the scene."}),
                "ref_boost_mask": ("MASK", {"tooltip": "Optional region on the last reference to boost, such as the face."}),
            },
        }

    def patch(self, model, source_image, vae, target_latent, ref_boost=1.0,
              source_image_b=None, ref_boost_a=1.0, fit_mode="fit", ref_boost_mask=None):
        upstream = sys.modules[_source_patch_class().__module__]
        height, width = target_latent["samples"].shape[-2:]
        images = [source_image]
        if source_image_b is not None:
            images.append(source_image_b)
        cache = {}

        # Encode before sampling so loading the VAE cannot evict the active DiT.
        def encode_sources(h, w):
            return [
                model.model.process_latent_in(upstream._fit_encode_image(
                    image, vae, h, w, cache, (index, h, w), fit_mode
                ))
                for index, image in enumerate(images)
            ]

        sources = encode_sources(height, width)

        def wrapper(executor, x, timesteps, context, *args, **kwargs):
            transformer_options = kwargs.get("transformer_options")
            if transformer_options is None:
                transformer_options = next(
                    (arg for arg in reversed(args) if isinstance(arg, dict)), {}
                )
            h, w = x.shape[-2:]
            refs = sources if (h, w) == (height, width) else encode_sources(h, w)
            return upstream.krea2_edit_forward(
                executor.class_obj, x, timesteps, context, refs, transformer_options,
                ref_boost=ref_boost, ref_boost_a=ref_boost_a,
                ref_boost_mask=ref_boost_mask, ref_native=(fit_mode == "fit"),
                pos_mode=("stride1" if fit_mode == "fit" else "anchor"),
            )

        patched = model.clone()
        options = patched.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "krea2_edit", wrapper, options,
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "Krea2EditModelPatchMXD": Krea2EditModelPatchMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2EditModelPatchMXD": "Krea2 Edit (source patch) MXD",
}
