from __future__ import annotations
import torch, comfy, comfy.model_management

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

    # Tuned for Z-Image Turbo:
    # - Rule of 64: every dimension is a multiple of 64
    # - 1MP baseline: 1024x1024 in the standard tier
    # - Ceiling: keep presets below 6.5MP
    MAX_TOTAL_PIXELS = 6_500_000
    MIN_BLOCK = 64
    RESOLUTIONS = {
        "— High Resolutions —": None,
        "Square (1:1) 1536x1536": (1536, 1536),
        "Photo (4:3) 1792x1344": (1792, 1344),
        "Landscape (3:2) 1920x1280": (1920, 1280),
        "Widescreen (16:9) 2048x1152": (2048, 1152),
        "Ultrawide (21:9) 2304x1024": (2304, 1024),

        "— Standard Resolutions —": None,
        "Square (1:1) 1024x1024": (1024, 1024),
        "Photo (4:3) 1152x896": (1152, 896),
        "Landscape (3:2) 1280x832": (1280, 832),
        "Widescreen (16:9) 1344x768": (1344, 768),
        "Ultrawide (21:9) 1536x640": (1536, 640),

        "— Low Resolutions —": None,
        "Square (1:1) 512x512": (512, 512),
        "Photo (4:3) 576x448": (576, 448),
        "Landscape (3:2) 640x448": (640, 448),
        "Widescreen (16:9) 704x384": (704, 384),
        "Ultrawide (21:9) 768x320": (768, 320),
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

        if (width % self.MIN_BLOCK) != 0 or (height % self.MIN_BLOCK) != 0:
            raise ValueError(
                f"Invalid preset {width}x{height}. Z-Image Turbo requires multiples of {self.MIN_BLOCK}."
            )
        if (width * height) > self.MAX_TOTAL_PIXELS:
            raise ValueError(
                f"Invalid preset {width}x{height}. Z-Image Turbo presets must stay at or below {self.MAX_TOTAL_PIXELS:,} pixels."
            )

        latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=self.device)
        return ({"samples": latent},)

########################################################################################################################

NODE_CLASS_MAPPINGS = {
    "Flux Empty Latent Image": FluxEmptyLatentImage,
    "Flux 2 Empty Latent Image": Flux2EmptyLatentImage,
    "Flux Resolution Selector": FluxResolutionSelector,
    "Sdxl Empty Latent Image": SdxlEmptyLatentImage,
    "ZImageTurboEmptyLatentImage": ZImageTurboEmptyLatentImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux Empty Latent Image": "Flux Empty Latent Image MXD",
    "Flux 2 Empty Latent Image": "Flux 2 Empty Latent Image MXD",
    "Flux Resolution Selector": "Flux Resolution Selector MXD",
    "Sdxl Empty Latent Image": "SDXL Empty Latent Image MXD",
    "ZImageTurboEmptyLatentImage": "ZIT Empty Latent Image MXD",
}
