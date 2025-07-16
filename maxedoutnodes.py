import torch, math, comfy
import comfy.utils
import comfy.model_management
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import node_helpers
from __future__ import annotations

########################################################################################################################
# Flux Empty Latent Image (SD3-compatible)
class FluxEmptyLatentImage:
    DESCRIPTION = """
    - Provides a wide selection of resolutions for easy selection.

    - Meant to save time from manually entering
    the resolution in the "Empty Latent Image" node.
    """
    TITLE = "Flux Empty Latent Image"
    CATEGORY = "KJNodes/Latent"

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
# Sdxl Empty Latent Image
class SdxlEmptyLatentImage:
    DESCRIPTION = """
    - Generates empty latent images. 

    - All supported SDXL resolutions 
    are predefined for ease of use.
    
    - Meant to save time from manually entering
    the resolution in the "Empty Latent Image" node.
    """
    TITLE = "Sdxl Empty Latent Image (With Resolutions)"
    CATEGORY = "KJNodes/Latent"

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
# Image Scale To Total Pixels (SDXL Safe)
class ImageScaleToTotalPixelsSafe:
    DESCRIPTION = """
    - Scales to target megapixel count, preserving aspect ratio.

    - If image matches SDXL resolutions (e.g. those used in
    "SDXL Empty Latent Image" node), scaling is skipped. 

    - Meant for SDXL workflows (e.g. image-to-image, inpainting)
    to auto-scale random images but not images already made with SDXL.
    """
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

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
                "upscale_method": (cls.upscale_methods,),
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
    CATEGORY = "KJNodes/Upscaling"

    def upscale(self, image, upscale_method, total_megapixels):
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
    DESCRIPTION = """
    - Scales to target megapixel count, preserving aspect ratio.

    - If image matches Flux-safe resolutions (e.g. those used in
    the Flux Empty Latent Image node), scaling is skipped. 

    - Meant for image-to-image or inpainting workflows to auto-scale 
    arbitrary images, but skip images already matching Flux resolutions.
    """
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

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
                "upscale_method": (cls.upscale_methods, ),
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
    CATEGORY = "KJNodes/Upscaling"

    def upscale(self, image, upscale_method, total_megapixels):
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
    DESCRIPTION = """
    - Combines clip text encode with flux guidance to lower node count.

    - Also removes the need to convert them into node group within ComfyUI. 
    """
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
    CATEGORY = "KJNodes/conditioning"

    def encode_and_guide(self, text, clip, guidance):
        if clip is None:
            raise RuntimeError("CLIP model is None. Your checkpoint may not contain a text encoder.")
        
        tokens = clip.tokenize(text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return (conditioning,)
    
########################################################################################################################

class DummyNegativePrompt(ComfyNodeABC):
    """
    A placeholder node that encodes an empty string into conditioning
    to satisfy KSampler's required negative prompt input.

    This node does *not* expose a text input and is not meant to be modified.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the (empty) text."}),
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    FUNCTION = "encode_empty"
    CATEGORY = "KJNodes/conditioning"

    def encode_empty(self, clip):
        if clip is None:
            raise RuntimeError("CLIP model is None. Your checkpoint may not contain a text encoder.")
        
        # Use a literal space instead of empty string to avoid tokenization issues
        tokens = clip.tokenize(" ")
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return (conditioning,)
    
########################################################################################################################    

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "Flux Empty Latent Image": FluxEmptyLatentImage,
    "Sdxl Empty Latent Image": SdxlEmptyLatentImage,
    "Image Scale To Total Pixels (SDXL Safe)": ImageScaleToTotalPixelsSafe,
    "Flux Image Scale To Total Pixels (Flux Safe)": FluxImageScaleToTotalPixelsSafe,
    "Prompt With Guidance (Flux)": PromptWithGuidance,
    "Dummy Negative Prompt": DummyNegativePrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux Empty Latent Image": "Flux Empty Latent Image MXD",
    "Sdxl Empty Latent Image": "SDXL Empty Latent Image MXD",
    "Image Scale To Total Pixels (SDXL Safe)": "Scale Image (SDXL Safe) MXD",
    "Flux Image Scale To Total Pixels (Flux Safe)": "Scale Image (Flux Safe) MXD",
    "Prompt With Guidance (Flux)": "Prompt with Flux Guidance MXD",
    "Dummy Negative Prompt": "Dummy Prompt MXD",
}
