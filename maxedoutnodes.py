from __future__ import annotations
import torch, math, comfy, os, folder_paths, node_helpers, comfy.model_management, comfy.utils, json, hashlib, re
import torch.nn.functional as F
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import numpy as np
from PIL import Image, ImageOps, ImageSequence, ImageFilter
try:
    from comfy_api.latest import io
    HAVE_COMFY_API = True
except Exception as _e:
    io = None
    HAVE_COMFY_API = False
    print(f"[ComfyUI-MaxedOut] comfy_api not available in maxedoutnodes: {_e}")

########################################################################################################################
# Flux Empty Latent Image (SD3-compatible)
class FluxEmptyLatentImage:
    DESCRIPTION = """Select a Flux resolution and create an empty latent batch."""
    TITLE = "Flux Empty Latent Image"
    CATEGORY = "MXD/Latent"

    RESOLUTIONS = {
        "— High Resolutions —": None,
        "Square (1:1) 1408x1408": (1408, 1408),
        "Standard (4:3) 1664x1216": (1664, 1216),
        "Landscape (3:2) 1728x1152": (1728, 1152),
        "Widescreen (16:9) 1920x1088": (1920, 1088),
        "Ultrawide (21:9) 2176x960": (2176, 960),

        "— Standard Resolutions —": None,
        "Square (1:1) 1024x1024": (1024, 1024),
        "Standard (4:3) 1152x896": (1152, 896),
        "Landscape (3:2) 1216x832": (1216, 832),
        "Widescreen (16:9) 1344x768": (1344, 768),
        "Ultrawide (21:9) 1536x640": (1536, 640),

        "— Low Resolutions —": None,
        "Square (1:1) 320x320": (320, 320),
        "Standard (4:3) 448x320": (448, 320),
        "Landscape (3:2) 384x256": (384, 256),
        "Widescreen (16:9) 448x256": (448, 256),
        "Ultrawide (21:9) 576x256": (576, 256),
    }
    
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "resolution": (
                    list(cls.RESOLUTIONS.keys()),
                    {"default": "Square (1:1) 1024x1024"}
                ),
                "vertical": ("BOOLEAN", {"default": False}),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "The number of latent images in the batch."
                    }
                )
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "generate"

    def generate(self, resolution, vertical, batch_size=1) -> tuple:
        size = self.RESOLUTIONS.get(resolution)
        if size is None:
            raise ValueError(f"'{resolution}' is a header or invalid option.")

        width, height = size
        if vertical:
            width, height = height, width

        latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=self.device)
        return ({"samples": latent},)

########################################################################################################################
# Flux 2 Empty Latent Image (Flux2-compatible)
class Flux2EmptyLatentImage:
    DESCRIPTION = """Select a Flux resolution and create an empty Flux 2 latent batch."""
    TITLE = "Flux 2 Empty Latent Image"
    CATEGORY = "MXD/Latent"

    RESOLUTIONS = FluxEmptyLatentImage.RESOLUTIONS

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "resolution": (
                    list(cls.RESOLUTIONS.keys()),
                    {"default": "Square (1:1) 1024x1024"}
                ),
                "vertical": ("BOOLEAN", {"default": False}),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "The number of latent images in the batch."
                    }
                )
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty Flux 2 latent image batch.",)
    FUNCTION = "generate"

    def generate(self, resolution, vertical, batch_size=1) -> tuple:
        size = self.RESOLUTIONS.get(resolution)
        if size is None:
            raise ValueError(f"'{resolution}' is a header or invalid option.")

        width, height = size
        if vertical:
            width, height = height, width

        latent = torch.zeros([batch_size, 128, height // 16, width // 16], device=self.device)
        return ({"samples": latent},)

########################################################################################################################
# Flux Resolution Selector (for feeding into FluxEmptyLatentImage)
class FluxResolutionSelector:
    DESCRIPTION = """Pick a Flux resolution string for Flux Empty Latent Image."""
    TITLE = "Flux Resolution Selector"
    CATEGORY = "MXD/Latent"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "resolution": (
                    list(FluxEmptyLatentImage.RESOLUTIONS.keys()),  # Include ALL keys including headers
                    {"default": "Square (1:1) 1024x1024"}
                ),
            }
        }

    RETURN_TYPES = (list(FluxEmptyLatentImage.RESOLUTIONS.keys()),)
    RETURN_NAMES = ("resolution",)
    OUTPUT_TOOLTIPS = ("The selected resolution string for FluxEmptyLatentImage.",)
    FUNCTION = "select_resolution"

    def select_resolution(self, resolution) -> tuple:
        return (resolution,)

########################################################################################################################
# Sdxl Empty Latent Image
class SdxlEmptyLatentImage:
    DESCRIPTION = """Select an SDXL resolution and create an empty latent batch."""
    TITLE = "Sdxl Empty Latent Image (With Resolutions)"
    CATEGORY = "MXD/Latent"

    # SDXL predefined resolutions (width, height)
    RESOLUTIONS = {
        "Square (1:1) 1024x1024": (1024, 1024),
        "Standard (4:3) 1152x896": (1152, 896),
        "Landscape (3:2) 1216x832": (1216, 832),
        "Widescreen (16:9) 1344x768": (1344, 768),
        "Ultra-Wide (21:9) 1536x640": (1536, 640),
    }

    def __init__(self):
        # Retrieve the intermediate device (usually the GPU) from ComfyUI's model management.
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                # Dropdown selection for one of the predefined SDXL resolutions.
                "resolution": (list(cls.RESOLUTIONS.keys()),),
                # Toggle for vertical mode (swaps width and height).
                "vertical": ("BOOLEAN", {"default": False}),
                # Number of latent images to create in the batch.
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "The number of latent images in the batch."
                    }
                )
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "generate"

    def generate(self, resolution, vertical, batch_size=1) -> tuple:
        # Get the selected resolution tuple (width, height)
        width, height = self.RESOLUTIONS[resolution]
        # If vertical mode is enabled, swap width and height.
        if vertical:
            width, height = height, width

        # Create an empty latent tensor.
        # Typically, the latent space has 4 channels and each spatial dimension is 1/8th of the image.
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples": latent},)
    
########################################################################################################################
# Z-Image Turbo Empty Latent Image (SD3-compatible) — Flux-style grouping
class ZImageTurboEmptyLatentImage:
    DESCRIPTION = """Select a Z-Image Turbo resolution and create an empty latent batch."""
    TITLE = "Z-Image Turbo Empty Latent Image"
    CATEGORY = "MXD/Latent"

    # Same resolutions as your original, just grouped like Flux
    RESOLUTIONS = {
        "— High Resolutions —": None,
        "Square (1:1) 1536x1536": (1536, 1536),
        "Square (1:1) 1280x1280": (1280, 1280),
        "Widescreen (16:9) 2048x1152": (2048, 1152),
        "Ultrawide (21:9) 2016x864": (2016, 864),

        "— Standard Resolutions —": None,
        "Square (1:1) 1024x1024": (1024, 1024),
        "Standard (3:2) 1536x1024": (1536, 1024),
        "Widescreen (16:9) 1920x1088": (1920, 1088),
        "Ultrawide (21:9) 1680x720": (1680, 720),

        "— Low Resolutions —": None,
        "Square (1:1) 768x768": (768, 768),
        "Standard (3:2) 1216x832": (1216, 832),
        "Widescreen (16:9) 1280x720": (1280, 720),
    }

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "resolution": (
                    list(cls.RESOLUTIONS.keys()),
                    {"default": "Square (1:1) 1024x1024"}
                ),
                "vertical": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Swap width and height."}
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4096, "tooltip": "Number of latent images in the batch."}
                )
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty Z-Image Turbo latent batch.",)
    FUNCTION = "generate"

    def generate(self, resolution, vertical, batch_size=1) -> tuple:
        size = self.RESOLUTIONS.get(resolution)
        if size is None:
            raise ValueError(f"'{resolution}' is a header or invalid option.")

        width, height = size
        if vertical:
            width, height = height, width

        latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=self.device)
        return ({"samples": latent},)

########################################################################################################################
# Image Scale To Total Pixels (SDXL Safe)
class SDXLImageScaleToTotalPixelsSafe:
    DESCRIPTION = """Scale to a target megapixel count and keep aspect ratio. Skips SDXL-safe sizes."""
    upscale_methods = ["bilinear", "bicubic", "lanczos", "nearest-exact", "area"]

    # SDXL-safe resolutions (width, height) – store one orientation only,
    # the code will check both (w, h) and (h, w)
    SDXL_SAFE_RESOLUTIONS = [
        (1024, 1024),
        (1152, 896),
        (1216, 832),
        (1344, 768),
        (1536, 640),
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (cls.upscale_methods, {"default": "bilinear"}),
                "total_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 128.0,
                        "step": 0.01,
                        "tooltip": "Set the total megapixels (e.g., 1.0 = 1 MP)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "MXD/Upscaling"

    def upscale(self, image, upscale_method, total_megapixels):
        if upscale_method in ["nearest-exact", "area"]:
            raise Exception(
                f"❌ '{upscale_method}' gives poor results.\n\n"
                f"👉 Go to the Scale SDXL Image MXD node and switch to another like 'lanczos'.\n\n"
                f"Node may be hidden behind KSampler."
            )
        
        b, h, w, c = image.shape

        # Skip scaling if the image already matches an SDXL-safe resolution
        if (w, h) in self.SDXL_SAFE_RESOLUTIONS or (h, w) in self.SDXL_SAFE_RESOLUTIONS:
            return (image,)

        # ComfyUI-native megapixel math
        samples = image.movedim(-1, 1)
        orig_h, orig_w = samples.shape[2], samples.shape[3]

        target_pixels = int(round(total_megapixels * 1024 * 1024))
        scale_by = math.sqrt(target_pixels / (orig_w * orig_h))

        new_w = max(1, round(orig_w * scale_by))
        new_h = max(1, round(orig_h * scale_by))

        scaled = comfy.utils.common_upscale(samples, new_w, new_h, upscale_method, "disabled")
        scaled = scaled.movedim(1, -1)
        return (scaled,)

########################################################################################################################
# Flux Image Scale To Total Pixels (Flux Safe)
class FluxImageScaleToTotalPixelsSafe:
    DESCRIPTION = """Scale to a target megapixel count and keep aspect ratio. Skips Flux-safe sizes."""
    upscale_methods = ["bilinear", "bicubic", "lanczos", "nearest-exact", "area"]

    # Flux-safe resolutions (width, height) – stored in one orientation only
    FLUX_SAFE_RESOLUTIONS = [
        (1408, 1408),
        (1728, 1152),
        (1664, 1216),
        (1920, 1088),
        (2176, 960),
        (1024, 1024),  
        (1216, 832),
        (1152, 896),
        (1344, 768),
        (1536, 640),
        (320, 320),
        (384, 256),
        (448, 320),
        (448, 256),
        (576, 256),
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (cls.upscale_methods, {"default": "bilinear"}),
                "total_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 128.0,
                        "step": 0.01,
                        "tooltip": "Set the total megapixels (e.g., 1.0 = 1 MP)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "MXD/Upscaling"

    def upscale(self, image, upscale_method, total_megapixels):
        if upscale_method in ["nearest-exact", "area"]:
            raise Exception(
                f"❌ '{upscale_method}' gives poor results.\n\n"
                f"👉 Go to the Scale Flux Image  MXD node and switch to another like 'lanczos'.\n\n"
                f"Node may be hidden behind KSampler."
            )
        
        b, h, w, c = image.shape

        # Skip scaling if image matches any Flux-safe resolution
        if (w, h) in self.FLUX_SAFE_RESOLUTIONS or (h, w) in self.FLUX_SAFE_RESOLUTIONS:
            return (image,)

        samples = image.movedim(-1, 1) 
        orig_h, orig_w = samples.shape[2], samples.shape[3]

        target_pixels = int(round(total_megapixels * 1024 * 1024))
        scale_by = math.sqrt(target_pixels / (orig_w * orig_h))

        new_w = max(1, round(orig_w * scale_by))
        new_h = max(1, round(orig_h * scale_by))

        scaled = comfy.utils.common_upscale(samples, new_w, new_h, upscale_method, "disabled")
        scaled = scaled.movedim(1, -1)
        return (scaled,)
    
########################################################################################################################
# Prompt with Guidance (Flux)
class PromptWithGuidance(ComfyNodeABC):
    DESCRIPTION = """Encode text and apply Flux guidance in one node."""
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1})
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    FUNCTION = "encode_and_guide"
    CATEGORY = "MXD/conditioning"

    def encode_and_guide(self, text, clip, guidance):
        if clip is None:
            raise RuntimeError("CLIP model is None. Your checkpoint may not contain a text encoder.")
        
        tokens = clip.tokenize(text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return (conditioning,)

########################################################################################################################
if HAVE_COMFY_API:
    class QwenImageEditSingleMXD(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="QwenImageEditSingleMXD",
                display_name="Qwen Image Edit + Latent MXD",
                category="MXD/conditioning",
                description="Encode prompt/image and output a matching empty latent.",
                inputs=[
                    io.Clip.Input("clip"),
                    io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                    io.Vae.Input("vae", optional=True),
                    io.Image.Input("image", optional=True),
                    io.Int.Input("batch_size", default=1, min=1, max=4096),
                ],
                outputs=[
                    io.Conditioning.Output(),
                    io.Latent.Output(), # New Output
                ],
            )

        @classmethod
        def execute(cls, clip, prompt, vae=None, image=None, batch_size=1) -> io.NodeOutput:
            ref_latents = []
            images_vl = []
            llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            image_prompt = ""

            # Default fallback size if no image is provided (1024x1024)
            final_width, final_height = 1024, 1024

            if image is not None:
                samples = image.movedim(-1, 1)

                # --- VISION SCALING (384px area) ---
                total_vl = int(384 * 384)
                scale_vl = math.sqrt(total_vl / (samples.shape[3] * samples.shape[2]))
                width_vl = round(samples.shape[3] * scale_vl)
                height_vl = round(samples.shape[2] * scale_vl)

                s_vl = comfy.utils.common_upscale(samples, width_vl, height_vl, "area", "disabled")
                images_vl.append(s_vl.movedim(1, -1))

                # --- LATENT/VAE SCALING (1024px area) ---
                total_lat = int(1024 * 1024)
                scale_lat = math.sqrt(total_lat / (samples.shape[3] * samples.shape[2]))
                # Calculate final dimensions to be multiples of 8
                final_width = round(samples.shape[3] * scale_lat / 8.0) * 8
                final_height = round(samples.shape[2] * scale_lat / 8.0) * 8

                if vae is not None:
                    s_lat = comfy.utils.common_upscale(samples, final_width, final_height, "area", "disabled")
                    ref_latents.append(vae.encode(s_lat.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"

            # 1. Generate the Empty Latent (SD3 Style: 16 channels, 1/8th resolution)
            # This replaces the need for the separate EmptySD3LatentImage node
            latent_tensor = torch.zeros(
                [batch_size, 16, final_height // 8, final_width // 8],
                device=comfy.model_management.intermediate_device()
            )
            latent_output = {"samples": latent_tensor}

            # 2. Process Conditioning
            tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
            conditioning = clip.encode_from_tokens_scheduled(tokens)

            if len(ref_latents) > 0:
                conditioning = node_helpers.conditioning_set_values(
                    conditioning,
                    {"reference_latents": ref_latents},
                    append=True,
                )

            return io.NodeOutput(conditioning, latent_output)

    ########################################################################################################################
    class QwenImageEditTripleMXD(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="QwenImageEditTripleMXD",
                display_name="Qwen Image Edit Prompt MXD (Triple)",
                category="advanced/conditioning",
                inputs=[
                    io.Clip.Input("clip"),
                    io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                    io.Vae.Input("vae", optional=True),
                    io.Image.Input("image1", optional=True),
                    io.Image.Input("image2", optional=True),
                    io.Image.Input("image3", optional=True),
                    io.Int.Input("batch_size", default=1, min=1, max=4096),
                ],
                outputs=[
                    io.Conditioning.Output(),
                    io.Latent.Output(),
                ],
            )

        @classmethod
        def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None, batch_size=1) -> io.NodeOutput:
            ref_latents = []
            images = [image1, image2, image3]
            images_vl = []
            llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            image_prompt = ""

            # Default fallback
            latent_width = 1024
            latent_height = 1024

            for i, image in enumerate(images):
                if image is not None:
                    samples = image.movedim(-1, 1)

                    # 1. VL Model Scaling (LLM Vision)
                    total_vl = int(384 * 384)
                    scale_by_vl = math.sqrt(total_vl / (samples.shape[3] * samples.shape[2]))
                    width_vl = round(samples.shape[3] * scale_by_vl)
                    height_vl = round(samples.shape[2] * scale_by_vl)
                    s_vl = comfy.utils.common_upscale(samples, width_vl, height_vl, "area", "disabled")
                    images_vl.append(s_vl.movedim(1, -1))

                    # 2. VAE Scaling (Synchronized to 16-step for SD3 compatibility)
                    if vae is not None:
                        total_ref = int(1024 * 1024)
                        scale_by_ref = math.sqrt(total_ref / (samples.shape[3] * samples.shape[2]))

                        # Pixels as multiple of 16 ensures Latent (Pixels/8) is always even
                        width_ref = round(samples.shape[3] * scale_by_ref / 16.0) * 16
                        height_ref = round(samples.shape[2] * scale_by_ref / 16.0) * 16

                        if i == 0:
                            latent_width = width_ref
                            latent_height = height_ref

                        s_ref = comfy.utils.common_upscale(samples, width_ref, height_ref, "area", "disabled")
                        ref_latents.append(vae.encode(s_ref.movedim(1, -1)[:, :, :, :3]))

                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

            # Process tokens and conditioning
            tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
            conditioning = clip.encode_from_tokens_scheduled(tokens)

            if len(ref_latents) > 0:
                conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)

            # Create Output Latent
            latent = torch.zeros([batch_size, 16, latent_height // 8, latent_width // 8], device=comfy.model_management.intermediate_device())

            # FIXED: Return outputs positionally to match the schema defined above
            # Output 1: Conditioning, Output 2: Latent Dictionary
            return io.NodeOutput(conditioning, {"samples": latent})

########################################################################################################################    
class FluxResolutionMatcher:
    DESCRIPTION = """Match the closest Flux resolution and orientation for the input image."""
    CATEGORY = "MXD/Latent"
    FUNCTION = "match_resolution"
    RETURN_NAMES = ("resolution", "vertical")

    # Full set kept for compatibility (enum list must match FluxEmptyLatentImage)
    RESOLUTIONS = {
        "— High Resolutions —": None,
        "Square (1:1) 1408x1408": (1408, 1408),
        "Standard (4:3) 1664x1216": (1664, 1216),
        "Landscape (3:2) 1728x1152": (1728, 1152),
        "Widescreen (16:9) 1920x1088": (1920, 1088),
        "Ultrawide (21:9) 2176x960": (2176, 960),

        "— Standard Resolutions —": None,
        "Square (1:1) 1024x1024": (1024, 1024),
        "Standard (4:3) 1152x896": (1152, 896),
        "Landscape (3:2) 1216x832": (1216, 832),
        "Widescreen (16:9) 1344x768": (1344, 768),
        "Ultrawide (21:9) 1536x640": (1536, 640),

        "— Low Resolutions —": None,
        "Square (1:1) 320x320": (320, 320),
        "Standard (4:3) 448x320": (448, 320),
        "Landscape (3:2) 384x256": (384, 256),
        "Widescreen (16:9) 448x256": (448, 256),
        "Ultrawide (21:9) 576x256": (576, 256),
    }

    # Keep same enum type so it connects to FluxEmptyLatentImage
    RETURN_TYPES = (list(RESOLUTIONS.keys()), "BOOLEAN")

    # Precompute aspect ratio groups (only for standard resolutions)
    ASPECT_RATIO_GROUPS = {}
    for res_str, dims in RESOLUTIONS.items():
        if dims is None:
            continue
        # ✅ Skip high and low groups for logic
        if "High" in res_str or "Low" in res_str:
            continue
        group_name = " ".join(res_str.split(' ')[:-1])
        if group_name not in ASPECT_RATIO_GROUPS:
            w, h = dims
            ratio = w / h
            ASPECT_RATIO_GROUPS[group_name] = {'ratio': ratio, 'resolutions': []}
        ASPECT_RATIO_GROUPS[group_name]['resolutions'].append(res_str)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    def match_resolution(self, image: torch.Tensor):
        if image.dim() < 4 or image.shape[1] < 1 or image.shape[2] < 1:
            print("Warning: Invalid image tensor received. Falling back to default resolution.")
            return ("Square (1:1) 1024x1024", False)

        _batch, height, width, _channels = image.shape
        is_vertical = height > width
        img_aspect_ratio = (height / width) if is_vertical else (width / height)
        img_area = height * width

        best_ar_group_name = min(
            self.ASPECT_RATIO_GROUPS.keys(),
            key=lambda name: abs(img_aspect_ratio - self.ASPECT_RATIO_GROUPS[name]['ratio'])
        )

        candidate_res_strings = self.ASPECT_RATIO_GROUPS[best_ar_group_name]['resolutions']

        best_res_string = min(
            candidate_res_strings,
            key=lambda res_str: abs(img_area - (self.RESOLUTIONS[res_str][0] * self.RESOLUTIONS[res_str][1]))
        )

        return (best_res_string, is_vertical)
########################################################################################################################

class SDXLResolutionMatcher:
    DESCRIPTION = """Match the closest SDXL resolution and orientation for the input image."""
    CATEGORY = "MXD/Latent"
    FUNCTION = "match_resolution"
    RETURN_NAMES = ("resolution", "vertical")

    # Use the exact same enum list as SdxlEmptyLatentImage
    RESOLUTIONS = SdxlEmptyLatentImage.RESOLUTIONS

    RETURN_TYPES = (list(RESOLUTIONS.keys()), "BOOLEAN")

    ASPECT_RATIO_GROUPS = {}
    for res_str, dims in RESOLUTIONS.items():
        if dims is None:
            continue
        group_name = " ".join(res_str.split(" ")[:-1])
        if group_name not in ASPECT_RATIO_GROUPS:
            w, h = dims
            ratio = w / h
            ASPECT_RATIO_GROUPS[group_name] = {"ratio": ratio, "resolutions": []}
        ASPECT_RATIO_GROUPS[group_name]["resolutions"].append(res_str)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    def match_resolution(self, image: torch.Tensor):
        if image.dim() < 4 or image.shape[1] < 1 or image.shape[2] < 1:
            print("Warning: Invalid image tensor received. Falling back to default resolution.")
            return ("Square (1:1) 1024x1024", False)

        _batch, height, width, _channels = image.shape
        is_vertical = height > width
        img_aspect_ratio = (height / width) if is_vertical else (width / height)
        img_area = height * width

        best_ar_group_name = min(
            self.ASPECT_RATIO_GROUPS.keys(),
            key=lambda name: abs(img_aspect_ratio - self.ASPECT_RATIO_GROUPS[name]["ratio"])
        )

        candidate_res_strings = self.ASPECT_RATIO_GROUPS[best_ar_group_name]["resolutions"]

        best_res_string = min(
            candidate_res_strings,
            key=lambda res_str: abs(img_area - (self.RESOLUTIONS[res_str][0] * self.RESOLUTIONS[res_str][1]))
        )

        return (best_res_string, is_vertical)
########################################################################################################################

class LatentHalfMasks:
    DESCRIPTION = """Split a latent into left and right half masks."""
    TITLE = "Latent Half Masks"
    CATEGORY = "MXD/Latent"

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask_left", "mask_right")
    OUTPUT_TOOLTIPS = (
        "Mask covering the left half of the latent.",
        "Mask covering the right half of the latent.",
    )
    FUNCTION = "make_masks"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask_left", "mask_right")
    FUNCTION = "make_masks"
    CATEGORY = "MXD/latent"

    def make_masks(self, latent):
        # Infer width/height from latent (assumes 8x scale)
        samples = latent.get("samples", None)
        if samples is None or not isinstance(samples, torch.Tensor):
            raise ValueError("LatentHalfMasks: invalid latent or missing 'samples' tensor.")
        h_lat, w_lat = samples.shape[-2], samples.shape[-1]
        w, h = int(w_lat * 8), int(h_lat * 8)

        # Always vertical, center split, no feather, no swap
        split_px = w // 2
        left = torch.zeros((h, w), dtype=torch.float32)
        right = torch.zeros((h, w), dtype=torch.float32)
        left[:, :split_px] = 1.0
        right[:, split_px:] = 1.0

        return left, right
    
########################################################################################################################

# Grow Blur Mask MXD (single-image friendly)
class GrowBlurMaskMXD:
    DESCRIPTION = """Expand or contract a mask and blur only the expanded ring (core stays solid)."""
    TITLE = "Grow Blur Mask MXD"
    CATEGORY = "MXD/Mask"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "grow_blur": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"

    def _normalize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor):
            raise ValueError("GrowBlurMaskMXD: mask must be a torch.Tensor.")

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        if mask.dim() != 3:
            raise ValueError("GrowBlurMaskMXD: mask must have shape (H,W) or (B,H,W).")

        return mask.float().clamp(0.0, 1.0)

    def _dilate(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask
        x = mask.unsqueeze(1)
        k = 2 * radius + 1
        y = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
        return y.squeeze(1)

    def _erode(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask
        x = mask.unsqueeze(1)
        k = 2 * radius + 1
        y = 1.0 - F.max_pool2d(1.0 - x, kernel_size=k, stride=1, padding=radius)
        return y.squeeze(1)

    def _blur_ring(self, ring: torch.Tensor, radius: int, device: torch.device) -> torch.Tensor:
        if radius <= 0:
            return ring
        ring_cpu = ring.detach().cpu().numpy()
        blurred = []
        for i in range(ring_cpu.shape[0]):
            arr = (ring_cpu[i] * 255.0).astype(np.uint8)
            pil = Image.fromarray(arr, mode="L")
            pil = pil.filter(ImageFilter.GaussianBlur(radius))
            out = np.array(pil).astype(np.float32) / 255.0
            blurred.append(torch.from_numpy(out))
        blurred_t = torch.stack(blurred, dim=0).to(device)
        return blurred_t.clamp(0.0, 1.0)

    def run(self, mask, grow_blur):
        mask = self._normalize_mask(mask)
        device = mask.device

        if grow_blur == 0:
            return (mask,)

        radius = abs(int(grow_blur))

        if grow_blur < 0:
            eroded = self._erode(mask, radius)
            return (eroded.clamp(0.0, 1.0),)

        core = mask
        expanded = self._dilate(mask, radius)
        ring = (expanded - core).clamp(0.0, 1.0)
        blurred_ring = self._blur_ring(ring, radius, device)
        blurred_ring = (blurred_ring * expanded).clamp(0.0, 1.0)
        out = (core + blurred_ring).clamp(0.0, 1.0)
        return (out,)
    
########################################################################################################################

# Get Latent Size
class GetLatentSizeMXD:
    DESCRIPTION = """Get image width/height from a latent."""
    TITLE = "Get Latent Size"
    CATEGORY = "MXD/Latent"

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    OUTPUT_TOOLTIPS = ("Latent-derived image width in pixels.", "Latent-derived image height in pixels.")
    FUNCTION = "get_size"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    def get_size(self, latent):
        if isinstance(latent, dict):
            width = latent.get("width")
            height = latent.get("height")
            if width is not None and height is not None:
                try:
                    return (int(width), int(height))
                except Exception:
                    pass

            samples = latent.get("samples")
        else:
            samples = None

        if samples is None or not isinstance(samples, torch.Tensor):
            raise ValueError("GetLatentSizeMXD: invalid latent or missing 'samples' tensor.")

        channels = samples.shape[1] if samples.dim() >= 2 else 0
        scale = 16 if channels >= 64 else 8

        h_lat, w_lat = samples.shape[-2], samples.shape[-1]
        return (int(w_lat * scale), int(h_lat * scale))

########################################################################################################################

# --- Helper function to find the bounding box of a mask ---
def get_bounding_box(mask_tensor):
    """
    Finds the bounding box of a non-zero region in a mask tensor.
    The mask is expected to be a 2D tensor (H, W).
    Returns a tuple (x_min, y_min, x_max, y_max) or None if the mask is empty.
    """
    # Get non-zero coordinates from the mask
    non_zero_coords = torch.nonzero(mask_tensor, as_tuple=False)

    # If the mask is empty, there is no bounding box
    if non_zero_coords.numel() == 0:
        return None

    # Find the min and max coordinates for y (dim 0) and x (dim 1)
    min_y = non_zero_coords[:, 0].min().item()
    max_y = non_zero_coords[:, 0].max().item()
    min_x = non_zero_coords[:, 1].min().item()
    max_x = non_zero_coords[:, 1].max().item()

    # The bounding box for PIL needs (left, upper, right, lower).
    # We add +1 to the max values because the upper bound is exclusive.
    return (min_x, min_y, max_x + 1, max_y + 1)

# --- Tensor to PIL and PIL to Tensor conversion helpers ---
def tensor_to_pil(tensor):
    """Converts a torch tensor (B, H, W, C) to a list of PIL Images."""
    if tensor is None:
        return []
        
    # Handle different tensor dimensions
    if tensor.dim() == 4: # Batch of images
        images = []
        for i in range(tensor.shape[0]):
            img_np = 255. * tensor[i].cpu().numpy()
            images.append(Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8)))
        return images
    elif tensor.dim() == 3: # Single image
        img_np = 255. * tensor.cpu().numpy()
        return [Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))]
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")

def pil_to_tensor(pil_images):
    """Converts a list of PIL Images back to a torch tensor (B, H, W, C)."""
    if not isinstance(pil_images, list):
        pil_images = [pil_images]
    
    tensors = []
    for img in pil_images:
        # Convert to RGB, then to a numpy array, normalize, and create a tensor
        img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img_np).unsqueeze(0))
    
    # Stack all tensors into a single batch tensor
    return torch.cat(tensors, dim=0)

# --------------------------------------------------------------------
# ✨ The Main Node Class ✨
# --------------------------------------------------------------------
class PlaceImageByMask:
    Description = """Place an overlay image inside the mask bounds on a base image."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "mask": ("MASK",),
                "overlay_image": ("IMAGE",),
            },
            "optional": {
                "maintain_aspect_ratio": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "place_image"
    CATEGORY = "MXD/Image"

    def place_image(self, base_image, overlay_image, mask, maintain_aspect_ratio=True):
        # Convert input tensors to lists of PIL Images
        base_pils = tensor_to_pil(base_image)
        overlay_pils = tensor_to_pil(overlay_image)
        
        processed_images = []
        
        # Process each image in the batch
        for i, base_pil in enumerate(base_pils):
            # Work with an RGBA version of the base image for clean pasting
            composited_image = base_pil.convert("RGBA")
            
            # Select the corresponding overlay and mask for the current base image
            # Clamping the index prevents errors if batch sizes are mismatched
            overlay_pil = overlay_pils[min(i, len(overlay_pils) - 1)].convert("RGBA")
            current_mask = mask[min(i, mask.shape[0] - 1)]

            # Find the bounding box from the mask
            bbox = get_bounding_box(current_mask)
            
            # If no mask is found, just use the original base image and skip to the next
            if not bbox:
                raise ValueError("The base image must be masked where you want the overlay to appear.")

            x_min, y_min, x_max, y_max = bbox
            box_width = x_max - x_min
            box_height = y_max - y_min

            # If the bounding box has no area, skip to the next image
            if box_width <= 0 or box_height <= 0:
                processed_images.append(base_pil)
                continue

            # --- Resize the overlay image using the specified method ---
            if maintain_aspect_ratio:
                # Resize to fit *within* the box, preserving aspect ratio (like a thumbnail)
                resized_overlay = overlay_pil.copy()
                resized_overlay.thumbnail((box_width, box_height), Image.Resampling.LANCZOS)
                
                # Calculate position to center the resized overlay within the bounding box
                paste_x = x_min + (box_width - resized_overlay.width) // 2
                paste_y = y_min + (box_height - resized_overlay.height) // 2
                paste_pos = (paste_x, paste_y)
            else:
                # As originally requested: stretch to fill the bounding box exactly
                resized_overlay = overlay_pil.resize((box_width, box_height), resample=Image.Resampling.LANCZOS)
                paste_pos = (x_min, y_min)

            # --- Paste the resized overlay onto the base image ---
            # The alpha channel of the overlay itself is used as the mask for pasting.
            # This ensures transparent areas of the overlay are handled correctly.
            composited_image.paste(resized_overlay, paste_pos, resized_overlay)
            
            processed_images.append(composited_image)

        # Convert the list of processed PIL images back to a single batch tensor for output
        output_tensor = pil_to_tensor(processed_images)
        return (output_tensor,)

######################################################################################################################################

class CropImageByMask:
    DESCRIPTION = """Crop images to the mask bounds when a mask is provided."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "crop"
    CATEGORY = "MXD/image"

    def crop(self, image, mask=None):
        # If no mask is provided or the mask is completely empty, return the original image
        if mask is None or not torch.any(mask > 0):
            return (image, )

        B, H, W, C = image.shape
        mask = mask.round()
        
        # Find bounding box for each batch
        crops = []
        
        for b in range(B):
            current_mask = mask[min(b, mask.shape[0]-1)]
            
            # Check if the mask for this specific image is empty.
            if not torch.any(current_mask > 0):
                # If a specific mask in a batch is empty, we can't crop.
                # To prevent errors with torch.cat later due to different sizes,
                # we'll skip cropping for the whole batch and return the original.
                # This ensures the output is always a valid tensor.
                print("Warning: An empty mask was found in a batch. Returning original images.")
                return (image, )

            # Get coordinates of non-zero elements
            rows = torch.any(current_mask > 0, dim=1)
            cols = torch.any(current_mask > 0, dim=0)
            
            # Find boundaries
            y_min, y_max = torch.where(rows)[0][[0, -1]]
            x_min, x_max = torch.where(cols)[0][[0, -1]]
            
            # Crop image
            crop = image[b:b+1, y_min:y_max+1, x_min:x_max+1, :]            
            crops.append(crop)
        
        # Note: This will raise an error if the crops have different sizes.
        # The original code had this limitation.
        cropped_images = torch.cat(crops, dim=0)
        
        return (cropped_images, )

########################################################################################################################
# ---------- Helpers (copied from latent loader style) ----------
def _safe_json_loads(s):
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
        try:
            return json.loads(json.loads(s))
        except Exception:
            return None


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

def _strip_counter(name: str) -> str:
    # Only strip the trailing pattern we generate when saving: "_<5digits>_"
    # Preserve numeric-only base names like "96".
    stem, _ = os.path.splitext(name)
    m = re.match(r"^(.*?)(?:_\d{5}_)$", stem)
    return m.group(1) if m else stem

# ---------- Node ----------
class LoadImageBatchMXD:
    DESCRIPTION = """Load images from an outputs folder, make masks from alpha, and read prompts."""
    TITLE = "Load Image Batch (Outputs + Prompts)"
    CATEGORY = "MXD/Image"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "positive", "negative")
    OUTPUT_IS_LIST = (True, True, True, True)
    FUNCTION = "load_batch"

    @classmethod
    def INPUT_TYPES(cls):
        outputs_root = folder_paths.get_output_directory()
        subdirs = [""] + sorted(
            [d for d in os.listdir(outputs_root) if os.path.isdir(os.path.join(outputs_root, d))]
        )
        return {"required": {"folder": (tuple(subdirs), {"default": ""})}}

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

    def load_batch(self, folder: str):
        outputs_root = folder_paths.get_output_directory()
        folder_path = os.path.join(outputs_root, folder) if folder else outputs_root

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"No such folder: {folder_path}")

        valid_exts = (".png", ".jpg", ".jpeg", ".webp")
        files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path))
                 if f.lower().endswith(valid_exts)]

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
        return {"required": {"image": (sorted(files), {"image_upload": True})}}

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

from nodes import PreviewImage, SaveImage
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
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    OUTPUT_TOOLTIPS = ("Saves and/or previews the images.",)

    def save(self, images, filename_prefix, mode, prompt=None, extra_pnginfo=None):
        if mode.startswith("Preview"):
            return PreviewImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        result = SaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        if mode == "Save Only" and isinstance(result, dict):
            # Strip UI previews so nothing shows up in the ComfyUI viewer.
            return {k: v for k, v in result.items() if k != "ui"}
        return result


########################################################################################################################
# Dummy Node (for workflow missing-node testing)
class DummyNodeMXD:
    DESCRIPTION = """Basic dummy node for missing-node workflow tests."""
    TITLE = "Dummy Node"
    CATEGORY = "MXD/Test"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "text": ("STRING", {"default": "hello"}),
                "repeat": ("INT", {"default": 1, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("The input text repeated N times.",)
    FUNCTION = "run"

    def run(self, text, repeat) -> tuple:
        return (text * int(repeat),)



# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "Flux Empty Latent Image": FluxEmptyLatentImage,
    "Flux 2 Empty Latent Image": Flux2EmptyLatentImage,
    "Sdxl Empty Latent Image": SdxlEmptyLatentImage,
    "Flux Resolution Selector": FluxResolutionSelector,
    "Image Scale To Total Pixels (SDXL Safe)": SDXLImageScaleToTotalPixelsSafe,
    "Flux Image Scale To Total Pixels (Flux Safe)": FluxImageScaleToTotalPixelsSafe,
    "Prompt With Guidance (Flux)": PromptWithGuidance,
    "FluxResolutionMatcher": FluxResolutionMatcher,
    "SDXLResolutionMatcher": SDXLResolutionMatcher,
    "LatentHalfMasks": LatentHalfMasks,
    "Grow Blur Mask MXD": GrowBlurMaskMXD,
    "Get Latent Size": GetLatentSizeMXD,
    "Place Image By Mask": PlaceImageByMask,
    "Crop Image By Mask": CropImageByMask,
    "Load Image Batch MXD": LoadImageBatchMXD,
    "LoadImageWithPromptsMXD": LoadImageWithPromptsMXD,
    "ZImageTurboEmptyLatentImage": ZImageTurboEmptyLatentImage,
    "Save Image MXD": SaveImage_MXD,
    "Dummy Node MXD": DummyNodeMXD,
}

if HAVE_COMFY_API:
    NODE_CLASS_MAPPINGS.update({
        "QwenImageEditSingleMXD": QwenImageEditSingleMXD,
        "QwenImageEditTripleMXD": QwenImageEditTripleMXD,
    })

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux Empty Latent Image": "Flux Empty Latent Image MXD",
    "Flux 2 Empty Latent Image": "Flux 2 Empty Latent Image MXD",
    "Sdxl Empty Latent Image": "SDXL Empty Latent Image MXD",
    "Flux Resolution Selector": "Flux Resolution Selector MXD",
    "Image Scale To Total Pixels (SDXL Safe)": "Scale SDXL Image MXD",
    "Flux Image Scale To Total Pixels (Flux Safe)": "Scale Flux Image MXD",
    "Prompt With Guidance (Flux)": "Prompt with Flux Guidance MXD",
    "FluxResolutionMatcher": "Flux Resolution Matcher MXD",
    "SDXLResolutionMatcher": "SDXL Resolution Matcher MXD",
    "LatentHalfMasks": "Latent to L/R Masks MXD",
    "Grow Blur Mask MXD": "Grow Blur Mask MXD",
    "Get Latent Size": "Get Latent Size MXD",
    "Place Image By Mask": "Place Image by Mask MXD",
    "Crop Image By Mask": "Crop Image by Mask MXD",
    "Load Image Batch MXD": "Load Image Batch MXD",
    "LoadImageWithPromptsMXD": "Load Image MXD",
    "ZImageTurboEmptyLatentImage": "ZImageTurbo Empty Latent Image MXD",
    "Save Image MXD": "Save Image MXD",
    "Dummy Node MXD": "Dummy Node MXD",
}

if HAVE_COMFY_API:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "QwenImageEditSingleMXD": "Qwen Image Edit + Latent MXD",
        "QwenImageEditTripleMXD": "Qwen Image Edit Prompt MXD (Triple)",
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
