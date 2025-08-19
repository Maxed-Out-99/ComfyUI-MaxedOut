from __future__ import annotations
import torch, math, comfy, os, folder_paths, node_helpers, comfy.model_management, comfy.utils
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import numpy as np
from PIL import Image, ImageDraw, ImageOps

########################################################################################################################
# Flux Empty Latent Image (SD3-compatible)
class FluxEmptyLatentImage:
    DESCRIPTION = """
    - Provides a wide selection of resolutions for Flux for easy selection.

    - Meant to save time from manually entering the same 
    resolutions in the "Empty Latent Image" node over and over.
    """
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
# Flux Resolution Selector (for feeding into FluxEmptyLatentImage)
class FluxResolutionSelector:
    DESCRIPTION = """
    - Provides a dropdown selection of Flux resolutions.
    - Output connects directly to the 'resolution' input of FluxEmptyLatentImage.
    - Useful for separating resolution selection from latent generation.
    """
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

    DESCRIPTION = """
    - Provides all compatible SDXL resolutions easy selection.

    - Meant to save time from manually entering the same 
    resolutions in the "Empty Latent Image" node over and over.
    """
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
# SDXL Resolution Selector
class SdxlResolutionSelector:
    DESCRIPTION = """
    - Provides a dropdown selection of SDXL resolutions.
    - Output connects directly to SDXL Empty Latent Image resolution input.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "resolution": (
                    list(SdxlEmptyLatentImage.RESOLUTIONS.keys()),
                    {"default": "Square (1:1) 1024x1024"}
                ),
            }
        }

    RETURN_TYPES = (list(SdxlEmptyLatentImage.RESOLUTIONS.keys()),)
    RETURN_NAMES = ("resolution",)
    FUNCTION = "select_resolution"
    CATEGORY = "MXD/Latent"

    def select_resolution(self, resolution) -> tuple:
        return (resolution,)

########################################################################################################################
# Image Scale To Total Pixels (SDXL Safe)
class SDXLImageScaleToTotalPixelsSafe:
    DESCRIPTION = """
    - Scales to target megapixel count, preserving aspect ratio.

    - If image matches SDXL resolutions (e.g. those used in
    "SDXL Empty Latent Image" node), scaling is skipped. 

    - Meant for SDXL workflows (e.g. image-to-image, inpainting)
    to auto-scale random images but not images already made with SDXL.
    """
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
    DESCRIPTION = """
    - Scales to target megapixel count, preserving aspect ratio.

    - If image matches Flux-safe resolutions (e.g. those used in
    the Flux Empty Latent Image node), scaling is skipped. 

    - Meant for image-to-image or inpainting workflows to auto-scale 
    arbitrary images, but skip images already matching Flux resolutions.
    """
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
    CATEGORY = "MXD/conditioning"

    def encode_and_guide(self, text, clip, guidance):
        if clip is None:
            raise RuntimeError("CLIP model is None. Your checkpoint may not contain a text encoder.")
        
        tokens = clip.tokenize(text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return (conditioning,)
    
########################################################################################################################    
class FluxResolutionMatcher:
    DESCRIPTION = """
    - For forcing the Flux Empty Latent Image node to auto-match the aspect ratio of the input image.

    - Resolution and vertical outputs plug into the Flux Empty Latent Image node.
    """
    # --- ComfyUI Setup ---
    CATEGORY = "MXD/Latent"
    
    FUNCTION = "match_resolution"
    RETURN_NAMES = ("resolution", "vertical")

    # The list of resolutions this node will match against.
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

    RETURN_TYPES = (list(RESOLUTIONS.keys()), "BOOLEAN")

    # --- Pre-computation at Class Load Time ---
    ASPECT_RATIO_GROUPS = {}
    for res_str, dims in RESOLUTIONS.items():
        if dims is None:
            continue
        group_name = " ".join(res_str.split(' ')[:-1])
        if group_name not in ASPECT_RATIO_GROUPS:
            w, h = dims
            ratio = w / h
            ASPECT_RATIO_GROUPS[group_name] = {'ratio': ratio, 'resolutions': []}
        ASPECT_RATIO_GROUPS[group_name]['resolutions'].append(res_str)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

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

class LatentHalfMasks:
    DESCRIPTION = """
    - Splits latent into clean left/right masks.
    - Designed for dual-character setups with two Apply PuLID nodes.
    - Simplifies by removing many masking/math nodes.
    """

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
    Description = """
    - Overlay an image onto a base image.
    - The position and scale of the overlay are determined by a mask's bounding box.
    - Useful for placing images in specific regions of a base image.
    """
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

class LoadImageBatchMXD:
    CATEGORY = "MXD/Image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_batch"

    @classmethod
    def INPUT_TYPES(cls):
        outputs_root = folder_paths.get_output_directory()
        subdirs = [""] + sorted(
            [d for d in os.listdir(outputs_root) if os.path.isdir(os.path.join(outputs_root, d))]
        )
        return {
            "required": {
                "folder": (tuple(subdirs), {"default": ""}),
            }
        }

    def load_batch(self, folder: str):
        outputs_root = folder_paths.get_output_directory()
        folder_path = os.path.join(outputs_root, folder) if folder else outputs_root

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"No such folder: {folder_path}")

        valid_exts = (".png", ".jpg", ".jpeg", ".webp")
        files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path))
                 if f.lower().endswith(valid_exts)]

        images, masks = [], []
        for path in files:
            i = Image.open(path)
            i = ImageOps.exif_transpose(i)
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

        return (images, masks)

########################################################################################################################


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "Flux Empty Latent Image": FluxEmptyLatentImage,
    "Sdxl Empty Latent Image": SdxlEmptyLatentImage,
    "Flux Resolution Selector": FluxResolutionSelector,
    "Sdxl Resolution Selector": SdxlResolutionSelector,
    "Image Scale To Total Pixels (SDXL Safe)": SDXLImageScaleToTotalPixelsSafe,
    "Flux Image Scale To Total Pixels (Flux Safe)": FluxImageScaleToTotalPixelsSafe,
    "Prompt With Guidance (Flux)": PromptWithGuidance,
    "FluxResolutionMatcher": FluxResolutionMatcher,
    "LatentHalfMasks": LatentHalfMasks,
    "Place Image By Mask": PlaceImageByMask,
    "Crop Image By Mask": CropImageByMask,
    "Load Image Batch MXD": LoadImageBatchMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux Empty Latent Image": "Flux Empty Latent Image MXD",
    "Sdxl Empty Latent Image": "SDXL Empty Latent Image MXD",
    "Flux Resolution Selector": "Flux Resolution Selector MXD",
    "Sdxl Resolution Selector": "SDXL Resolution Selector MXD",
    "Image Scale To Total Pixels (SDXL Safe)": "Scale SDXL Image MXD",
    "Flux Image Scale To Total Pixels (Flux Safe)": "Scale Flux Image MXD",
    "Prompt With Guidance (Flux)": "Prompt with Flux Guidance MXD",
    "FluxResolutionMatcher": "Flux Resolution Matcher MXD",
    "LatentHalfMasks": "Latent to L/R Masks MXD",
    "Place Image By Mask": "Place Image by Mask MXD",
    "Crop Image By Mask": "Crop Image by Mask MXD",
    "Load Image Batch MXD": "Load Image Batch MXD",
}
