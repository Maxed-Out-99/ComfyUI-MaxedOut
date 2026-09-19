"""Krea 2 editing nodes.

Registered nodes:
  Krea2EditModelPatchMXD       Krea 2 Edit MXD
  Krea2EditGroundedEncodeMXD   Krea2 Edit MXD
"""

import comfy.patcher_extension
import comfy.utils

from .krea2_edit_core import fit_encode_image, krea2_edit_forward


class Krea2EditModelPatchMXD:
    TITLE = "Krea 2 Edit MXD"
    CATEGORY = "MXD/Krea"
    DESCRIPTION = (
        "Adds the Krea 2 edit source-preservation path with independent boost "
        "and mask controls for each reference image."
    )
    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "source_latent")
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "image_1": (
                    "IMAGE",
                    {"tooltip": "Primary reference image, usually the scene or image to edit."},
                ),
            },
            "optional": {
                "image_2": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional second reference, usually a subject to place into Image 1."
                        )
                    },
                ),
                "image_1_boost": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "round": 0.001,
                        "tooltip": "Attention strength for Image 1. 1.0 = unchanged.",
                    },
                ),
                "image_1_boost_mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Optional mask limiting Image 1 Boost to a region such as a face. "
                            "White areas are boosted."
                        )
                    },
                ),
                "image_2_boost": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "round": 0.001,
                        "tooltip": (
                            "Attention strength for Image 2. 1.0 = unchanged; no effect when "
                            "Image 2 is disconnected."
                        ),
                    },
                ),
                "image_2_boost_mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Optional mask limiting Image 2 Boost to a region such as a face. "
                            "White areas are boosted; no effect when Image 2 is disconnected."
                        )
                    },
                ),
                "vae": (
                    "VAE",
                    {"tooltip": "Required. VAE used to encode the reference images."},
                ),
            },
        }

    def patch(
        self,
        model,
        image_1,
        image_2=None,
        image_1_boost=1.0,
        image_1_boost_mask=None,
        image_2_boost=1.0,
        image_2_boost_mask=None,
        vae=None,
    ):
        if vae is None:
            raise ValueError("Connect a VAE to Krea 2 Edit MXD.")

        images = [image_1]
        boosts = [image_1_boost]
        boost_masks = [image_1_boost_mask]
        if image_2 is not None:
            images.append(image_2)
            boosts.append(image_2_boost)
            boost_masks.append(image_2_boost_mask)
        cache = {}

        # Cache each source at the actual sampled resolution.
        def encode_sources(height, width):
            return [
                model.model.process_latent_in(
                    fit_encode_image(
                        image,
                        vae,
                        height,
                        width,
                        cache,
                        (index, height, width),
                        "fit",
                    )
                )
                for index, image in enumerate(images)
            ]

        def wrapper(executor, x, timesteps, context, *args, **kwargs):
            transformer_options = kwargs.get("transformer_options")
            if transformer_options is None:
                transformer_options = next(
                    (arg for arg in reversed(args) if isinstance(arg, dict)), {}
                )
            height, width = x.shape[-2:]
            refs = encode_sources(height, width)
            return krea2_edit_forward(
                executor.class_obj,
                x,
                timesteps,
                context,
                refs,
                transformer_options,
                image_boosts=boosts,
                image_boost_masks=boost_masks,
                ref_native=True,
                pos_mode="stride1",
            )

        patched = model.clone()
        options = patched.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "krea2_edit",
            wrapper,
            options,
        )
        # This is deliberately only the first image's ordinary VAE latent.
        # The optional second reference remains internal to the edit wrapper.
        source_latent = {"samples": vae.encode(image_1[..., :3])}
        return (patched, source_latent)


class Krea2EditGroundedEncodeMXD:
    """Encode a Krea 2 edit instruction together with its reference image."""

    TITLE = "Krea2 Edit MXD"
    CATEGORY = "MXD/Krea"
    DESCRIPTION = (
        "Encodes the edit instruction grounded on the source image using the "
        "training-matched Krea 2 semantic path."
    )
    DEFAULT_SYSTEM = (
        "Describe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:"
    )
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    @classmethod
    def _template(cls, image_count):
        vision_tokens = "<|vision_start|><|image_pad|><|vision_end|>" * image_count
        return (
            "<|im_start|>system\n"
            + cls.DEFAULT_SYSTEM
            + "<|im_end|>\n<|im_start|>user\n"
            + vision_tokens
            + "{}<|im_end|>\n<|im_start|>assistant\n"
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional second reference (subject) for multi-reference "
                            "LoRAs; the first image is the scene."
                        )
                    },
                ),
                "grounding_px": (
                    "INT",
                    {
                        "default": 768,
                        "min": 0,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Maximum longest side fed to Qwen3-VL; 0 uses native resolution.",
                    },
                ),
            },
        }

    @staticmethod
    def _prep(image, grounding_px):
        samples = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
        height, width = samples.shape[2], samples.shape[3]
        if grounding_px and max(height, width) > grounding_px:
            scale = grounding_px / max(height, width)
            samples = comfy.utils.common_upscale(
                samples,
                round(width * scale),
                round(height * scale),
                "area",
                "disabled",
            )
        return samples.movedim(1, -1)[:, :, :, :3]

    def encode(self, clip, prompt, image=None, image_b=None, grounding_px=768):
        if image is None:
            tokens = clip.tokenize(prompt)
            return (clip.encode_from_tokens_scheduled(tokens),)

        images = [self._prep(image, grounding_px)]
        if image_b is not None:
            images.append(self._prep(image_b, grounding_px))
        tokens = clip.tokenize(
            prompt,
            images=images,
            llama_template=self._template(len(images)),
        )
        return (clip.encode_from_tokens_scheduled(tokens),)


NODE_CLASS_MAPPINGS = {
    "Krea2EditModelPatchMXD": Krea2EditModelPatchMXD,
    "Krea2EditGroundedEncodeMXD": Krea2EditGroundedEncodeMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2EditModelPatchMXD": "Krea 2 Edit MXD",
    "Krea2EditGroundedEncodeMXD": "Krea2 Edit MXD",
}
