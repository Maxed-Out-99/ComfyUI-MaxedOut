import torch
import comfy
import comfy.model_management
import math
import comfy.utils
########################################################################################################################
# Flux Empty Latent Image
class FluxEmptyLatentImage:
    TITLE = "Flux Empty Latent Image (With Resolutions)"
    CATEGORY = "latent"
    # Predefined resolutions from your Flux Resolutions node
    RESOLUTIONS = {
        "High Res (1:1) Square 1408x1408": (1408, 1408),
        "High Res (3:2) Landscape 1728x1152": (1728, 1152),
        "High Res (4:3) Standard 1664x1216": (1664, 1216),
        "High Res (16:9) Widescreen 1920x1088": (1920, 1088),
        "High Res (21:9) Ultrawide 2176x960": (2176, 960),
        "Standard Res (1:1) Square 1024x1024": (1024, 1024),  
        "Standard Res (3:2) Landscape 1216x832": (1216, 832),
        "Standard Res (4:3) Standard 1152x896": (1152, 896),
        "Standard Res (16:9) Widescreen 1344x768": (1344, 768),
        "Standard Res (21:9) Ultrawide 1536x640": (1536, 640),
        "Low Res (1:1) Square 320x320": (320, 320),
        "Low Res (3:2) Landscape 384x256": (384, 256),
        "Low Res (4:3) Standard 448x320": (448, 320),
        "Low Res (16:9) Widescreen 448x256": (448, 256),
        "Low Res (21:9) Ultrawide 576x256": (576, 256),
    }

    def __init__(self):
        # Get the intermediate device (usually a GPU device) from ComfyUI's model management
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                # Dropdown to select one of the predefined resolutions, defaulting to Standard Res Square
                "resolution": (
                    list(cls.RESOLUTIONS.keys()),
                    {"default": "Standard Res (1:1) Square 1024x1024"}
                ),
                # Toggle for vertical mode (swaps width and height)
                "vertical": ("BOOLEAN",),
                # Number of latent images to create in the batch
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
        # Look up the chosen resolution (width, height)
        width, height = self.RESOLUTIONS[resolution]
        # Swap width and height if vertical mode is enabled
        if vertical:
            width, height = height, width

        # Create the empty latent tensor.
        # Note: Typically the latent space has 4 channels and each spatial dimension is 1/8th of the image.
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
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
        "Standard Wide (4:3) 1152x896": (1152, 896),
        "Cinematic Wide (3:2) 1216x832": (1216, 832),
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
                "vertical": ("BOOLEAN",),
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
# SDXL Resolutions
class SDXL_Resolutions:
    RESOLUTIONS = {
        "Square (1:1) 1024x1024": (1024, 1024),
        "Standard Wide (4:3) 1152x896": (1152, 896),
        "Cinematic Wide (3:2) 1216x832": (1216, 832),
        "Ultra-Wide (16:9) 1344x768": (1344, 768),
        "Super Ultra-Wide (21:9) 1536x640": (1536, 640),
    }

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (list(cls.RESOLUTIONS.keys()),),
                "vertical": ("BOOLEAN", {"default": False, "tooltip": "Swap width and height if true"})
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "JPS Nodes/Settings"

    def get_resolution(self, resolution, vertical=False):
        # Retrieve width and height from the preset dictionary.
        width, height = self.RESOLUTIONS[resolution]
        # If vertical mode is enabled, swap the dimensions.
        if vertical:
            width, height = height, width
        return int(width), int(height)

########################################################################################################################
# SD 1.5 Empty Latent Image
class Sd15EmptyLatentImage:
    TITLE = "Sd 1.5 Empty Latent Image (With Resolutions)"
    CATEGORY = "latent"
    # Adjusted resolutions to be multiples of 64 (SD 1.5 compatible)
    RESOLUTIONS = {
        "Square (1:1) 512x512": (512, 512),
        "Standard Wide (4:3) 576x448": (576, 448),
        "Portrait (4:5) 448x352": (448, 352),
        "Cinematic Wide (3:2) 576x384": (576, 384),
        "Ultra-Wide (16:9) 640x384": (640, 384),
        "Super Ultra-Wide (21:9) 768x320": (768, 320),
    }

    def __init__(self):
        # Retrieve the intermediate device (usually the GPU) from ComfyUI's model management.
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                # Dropdown selection for one of the predefined SD 1.5 resolutions.
                "resolution": (list(cls.RESOLUTIONS.keys()),),
                # Toggle for vertical mode (swaps width and height).
                "vertical": ("BOOLEAN",),
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
        # SD 1.5 uses 4 latent channels, and spatial dimensions are 1/8th of image size.
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
    CATEGORY = "KJNodes/Upscaling"   # Make sure the category matches a recognized prefix for popups

    def upscale(self, image, upscale_method, total_megapixels):
        b, h, w, c = image.shape

        # Skip scaling if the image already matches an SDXL-safe resolution
        if (w, h) in self.SDXL_SAFE_RESOLUTIONS or (h, w) in self.SDXL_SAFE_RESOLUTIONS:
            return (image,)

        # ComfyUI-native megapixel math
        samples = image.movedim(-1, 1)          # B, C, H, W
        orig_h, orig_w = samples.shape[2], samples.shape[3]

        target_pixels = int(round(total_megapixels * 1024 * 1024))
        scale_by = math.sqrt(target_pixels / (orig_w * orig_h))

        new_w = max(1, round(orig_w * scale_by))
        new_h = max(1, round(orig_h * scale_by))

        scaled = comfy.utils.common_upscale(samples, new_w, new_h, upscale_method, "disabled")
        scaled = scaled.movedim(1, -1)          # back to B, H, W, C
        return (scaled,)

########################################################################################################################

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "Flux Empty Latent Image": FluxEmptyLatentImage,
    "Sdxl Empty Latent Image": SdxlEmptyLatentImage,
    "Sd 1.5 Empty Latent Image": Sd15EmptyLatentImage,
    "SDXL Resolutions": SDXL_Resolutions,
    "Image Scale To Total Pixels (SDXL Safe)": ImageScaleToTotalPixelsSafe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux Empty Latent Image": "Flux Empty Latent Image",
    "Sdxl Empty Latent Image": "SDXL Empty Latent Image",
    "Sd 1.5 Empty Latent Image": "SD 1.5 Empty Latent Image",
    "SDXL Resolutions": "SDXL Resolutions (Settings)",
    "Image Scale To Total Pixels (SDXL Safe)": "Scale Image (SDXL Safe)",
}
