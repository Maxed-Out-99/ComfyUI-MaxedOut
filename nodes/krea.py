"""Modular Krea 2 editing nodes.

Published compatibility nodes (original mapping keys and input schemas):
  Krea2EditModelPatchMXD       Krea 2 Edit MXD
  Krea2EditGroundedEncodeMXD   Krea2 Edit Prompt MXD

Focused editing nodes:
  Krea2ImgEditMXD             Krea2 Img Edit MXD
  Krea2ImgEditAdvMXD          Krea2 Img Edit Adv MXD
  Krea2TwoImgEditMXD          Krea2 2 Img Edit MXD
  Krea2EditAdvancedMXD        Krea2 2 Img Edit Adv MXD
  Krea2ImgPromptMXD           Krea2 Img Edit Prompt MXD
  Krea2TwoImgPromptMXD        Krea2 2 Img Edit Prompt MXD

The advanced two-image key retains the advanced node introduced locally.
All new editors share the same edit path; conditioning and sampling stay separate.
"""

import comfy.patcher_extension
import comfy.utils

from .krea2_edit_core import REF_POSITIONS, fit_encode_image, fit_boost_mask, krea2_edit_forward


IMAGE_ROLES = {"Base": REF_POSITIONS[0], "Ref": REF_POSITIONS[1]}


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


class _KreaEditImplementation:
    TITLE = "Krea 2 Edit Advanced MXD"
    CATEGORY = "MXD/Krea"
    DESCRIPTION = (
        "Krea 2 editing with independent Base/Ref roles, attention boosts, and "
        "boost masks for each image. Masks limit attention boosts; they are not edit masks."
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
                # Keep each image role independent.
                "image_1_position": (
                    list(IMAGE_ROLES),
                    {"default": "Base", "tooltip": "Base: follow this image's layout and framing. Ref: use this image's appearance with more freedom to reposition and reframe its contents. Base keeps existing V1.2 behavior. This does not change source_latent, which always uses Image 1."},
                ),
                "image_2_position": (
                    list(IMAGE_ROLES),
                    {"default": "Base", "tooltip": "Base: follow this image's layout and framing. Ref: use this image's appearance with more freedom to reposition and reframe its contents. Base keeps existing V1.2 behavior. No effect when Image 2 is disconnected. Set both images to Ref to restage two subjects in a new scene."},
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
        image_1_position="Base",
        image_2_position="Base",
    ):
        if vae is None:
            raise ValueError("Connect a VAE to Krea 2 Edit MXD.")

        images = [image_1]
        boosts = [image_1_boost]
        boost_masks = [image_1_boost_mask]
        positions = [IMAGE_ROLES.get(image_1_position, image_1_position)]
        if image_2 is not None:
            images.append(image_2)
            boosts.append(image_2_boost)
            boost_masks.append(image_2_boost_mask)
            positions.append(IMAGE_ROLES.get(image_2_position, image_2_position))
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
                image_boost_masks=[fit_boost_mask(mask, image, height * 8, width * 8)
                                   for mask, image in zip(boost_masks, images)],
                ref_native=True,
                pos_mode="stride1",
                image_positions=positions,
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


def _edit_inputs(*, two=False, advanced=False):
    """Build explicit one/two-image schemas without exposing unused image controls."""
    source = _KreaEditImplementation.INPUT_TYPES()
    options = source["optional"]
    required = {"model": ("MODEL",)}
    if two:
        if advanced:
            required.update({"image_1": source["required"]["image_1"],
                             "image_2": options["image_2"]})
        else:
            required.update({
                "img_base": ("IMAGE", {"tooltip": "Image to edit. Guides layout and framing; supplies source_latent."}),
                "img_ref": ("IMAGE", {"tooltip": "Appearance reference with freer placement than the base image."}),
            })
    else:
        required["image"] = ("IMAGE", {"tooltip": "The image to edit; supplies source_latent."})
    required["vae"] = options["vae"]
    controls = {}
    for i in range(1, 3 if two else 2):
        if not two:
            prefix, label = "", "the image"
        elif advanced:
            prefix, label = f"image_{i}_", f"Image {i}"
        elif i == 1:
            prefix, label = "img_base_", "the base image"
        else:
            prefix, label = "img_ref_", "the reference image"
        boost = options[f"image_{i}_boost"]
        if two and i == 2:
            boost[1]["default"] = 2.0
        boost[1]["tooltip"] = f"Attention strength for {label}. 1.0 = unchanged."
        controls[prefix + "boost"] = boost
        if advanced:
            controls[prefix + "boost_mask"] = (
                "MASK", {"tooltip": f"Limit the boost for {label} to white areas. This is not an edit or inpainting mask."}
            )
            controls[prefix + "position"] = (
                list(IMAGE_ROLES), {
                    "default": "Ref" if two and i == 2 else "Base",
                    "tooltip": (
                        "Base: follow this image's layout and framing. Ref: use its appearance "
                        "with freer positioning. "
                        "This does not change which image supplies source_latent."
                    ),
                }
            )
    return {"required": required, "optional": controls}


class Krea2ImgEditMXD(_KreaEditImplementation):
    TITLE = "Krea2 Img Edit MXD"
    DESCRIPTION = "Edit one image with one attention boost. Connect conditioning and a sampler separately."

    @classmethod
    def INPUT_TYPES(cls):
        return _edit_inputs()

    def patch(self, model, image, vae, boost=1.0):
        return super().patch(model, image, image_1_boost=boost, vae=vae)


class Krea2ImgEditAdvMXD(_KreaEditImplementation):
    TITLE = "Krea2 Img Edit Adv MXD"
    DESCRIPTION = "Edit or restage one image with a Base/Ref role, attention boost, and optional boost mask."

    @classmethod
    def INPUT_TYPES(cls):
        return _edit_inputs(advanced=True)

    def patch(self, model, image, vae, boost=1.0, boost_mask=None, position="Base"):
        return super().patch(model, image, image_1_boost=boost,
                             image_1_boost_mask=boost_mask, vae=vae,
                             image_1_position=position)


class Krea2TwoImgEditMXD(_KreaEditImplementation):
    TITLE = "Krea2 2 Img Edit MXD"
    DESCRIPTION = "Edit a base image using a second appearance reference. Base follows layout; Ref allows freer placement."

    @classmethod
    def INPUT_TYPES(cls):
        return _edit_inputs(two=True)

    def patch(self, model, img_base, img_ref, vae, img_base_boost=1.0, img_ref_boost=2.0):
        return super().patch(model, img_base, image_2=img_ref,
                             image_1_boost=img_base_boost, image_2_boost=img_ref_boost,
                             vae=vae, image_1_position="Base", image_2_position="Ref")


class Krea2EditAdvancedMXD(_KreaEditImplementation):
    TITLE = "Krea2 2 Img Edit Adv MXD"
    DESCRIPTION = "Two images with independent Base/Ref roles, attention boosts, and boost masks. Use Ref for both to restage two subjects."

    @classmethod
    def INPUT_TYPES(cls):
        return _edit_inputs(two=True, advanced=True)

    def patch(self, model, image_1, image_2=None, image_1_boost=1.0,
              image_1_boost_mask=None, image_2_boost=2.0, image_2_boost_mask=None,
              vae=None, image_1_position="Base", image_2_position="Ref"):
        return super().patch(
            model, image_1, image_2=image_2,
            image_1_boost=image_1_boost, image_1_boost_mask=image_1_boost_mask,
            image_2_boost=image_2_boost, image_2_boost_mask=image_2_boost_mask,
            vae=vae, image_1_position=image_1_position, image_2_position=image_2_position,
        )


class Krea2EditGroundedEncodeMXD:
    """Encode a Krea 2 edit instruction together with its reference image."""

    TITLE = "Krea2 Edit Prompt MXD"
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
        # Legacy workflows retain their grounding control. New prompt nodes use 768px.
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


class Krea2ImgPromptMXD(Krea2EditGroundedEncodeMXD):
    TITLE = "Krea2 Img Edit Prompt MXD"
    DESCRIPTION = "Encode an edit instruction with the same image used by the single-image editor. Image grounding is fixed at 768px."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "image": ("IMAGE", {"tooltip": "Connect the same image as the single-image edit node."}),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
        }}

    def encode(self, clip, image, prompt):
        return super().encode(clip, prompt, image=image, grounding_px=768)


class Krea2TwoImgPromptMXD(Krea2EditGroundedEncodeMXD):
    TITLE = "Krea2 2 Img Edit Prompt MXD"
    DESCRIPTION = "Encode an edit instruction with both images in the same order as the two-image editor. Image grounding is fixed at 768px."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "img_base": ("IMAGE", {"tooltip": "The base image, or Image 1 when using the advanced editor."}),
            "img_ref": ("IMAGE", {"tooltip": "The reference image, or Image 2 when using the advanced editor."}),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
        }}

    def encode(self, clip, img_base, img_ref, prompt):
        return super().encode(clip, prompt, image=img_base, image_b=img_ref, grounding_px=768)


NODE_CLASS_MAPPINGS = {
    "Krea2EditModelPatchMXD": Krea2EditModelPatchMXD,
    "Krea2EditGroundedEncodeMXD": Krea2EditGroundedEncodeMXD,
    "Krea2ImgEditMXD": Krea2ImgEditMXD,
    "Krea2ImgEditAdvMXD": Krea2ImgEditAdvMXD,
    "Krea2TwoImgEditMXD": Krea2TwoImgEditMXD,
    "Krea2EditAdvancedMXD": Krea2EditAdvancedMXD,
    "Krea2ImgPromptMXD": Krea2ImgPromptMXD,
    "Krea2TwoImgPromptMXD": Krea2TwoImgPromptMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {key: cls.TITLE for key, cls in NODE_CLASS_MAPPINGS.items()}
