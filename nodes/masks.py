"""Mask and mask-driven image operations.

Registered nodes:
  LatentHalfMasks               Latent to L/R Masks MXD
  Get Latent Size               Get Latent Size MXD
  Place Image By Mask           Place Image by Mask MXD
  Crop Image By Mask            Crop Image by Mask MXD
  SmartCropByMaskMXD            Smart Crop by Mask MXD
  BboxDetectorCombinedBatchMXD  BBOX Detector Combined Batch MXD
  ImageAndMaskPreviewMXD        Image and Mask Preview MXD
"""
from __future__ import annotations
import torch, comfy, comfy.utils, folder_paths, random
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageColor
from nodes import SaveImage

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

class SmartCropByMaskMXD:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "mask": ("MASK", ),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "crop"
    CATEGORY = "image/transform"
    DESCRIPTION = "Slides a square crop window horizontally + vertically to center on subject mask."

    def crop(self, image, mask):
        B, H, W, C = image.shape
        mask = mask.round()
        crops = []

        for b in range(B):
            mask_b = mask[min(b, mask.shape[0]-1)]

            # Get non-zero rows and columns
            rows = torch.any(mask_b > 0, dim=1)
            cols = torch.any(mask_b > 0, dim=0)

            # Default to center
            center_x = W // 2
            center_y = H // 2

            # Update center_x from mask if possible
            if torch.any(cols):
                x_min, x_max = torch.where(cols)[0][[0, -1]]
                center_x = (x_min + x_max) // 2

            # Update center_y from mask if possible
            if torch.any(rows):
                y_min, y_max = torch.where(rows)[0][[0, -1]]
                center_y = (y_min + y_max) // 2

            # Compute square crop box
            side = min(H, W)
            half = side // 2

            left = max(0, center_x - half)
            right = min(W, left + side)
            left = right - side  # clamp again

            top = max(0, center_y - half)
            bottom = min(H, top + side)
            top = bottom - side  # clamp again

            # Final crop: safe slicing
            crop = image[b:b+1, top:bottom, left:right, :]
            crops.append(crop)

        return (torch.cat(crops, dim=0), )

########################################################################################################################

class BboxDetectorCombinedBatchMXD:
    DESCRIPTION = "Run an Impact Pack BBOX_DETECTOR combined mask over each image in a batch."
    CATEGORY = "MXD/Detector"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "detect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_detector": ("BBOX_DETECTOR",),
                "images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 4, "min": -512, "max": 512, "step": 1}),
            }
        }

    def detect(self, bbox_detector, images, threshold=0.5, dilation=4):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"[BboxDetectorCombinedBatchMXD] Expected IMAGE tensor [B,H,W,C], got shape {tuple(images.shape)}")

        masks = []
        frame_count, height, width, _ = images.shape
        pbar = comfy.utils.ProgressBar(frame_count)

        for i in range(frame_count):
            frame = images[i:i + 1]
            mask = bbox_detector.detect_combined(frame, threshold, dilation)
            if mask is None:
                mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
            elif torch.is_tensor(mask):
                mask = mask.detach().to(dtype=torch.float32, device="cpu")
            else:
                mask = torch.as_tensor(mask, dtype=torch.float32, device="cpu")

            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            if mask.ndim != 2:
                raise ValueError(f"[BboxDetectorCombinedBatchMXD] Detector returned unexpected mask shape {tuple(mask.shape)} for frame {i}.")

            masks.append(mask.unsqueeze(0))
            pbar.update(1)

        return (torch.cat(masks, dim=0),)

########################################################################################################################

def _parse_mxd_mask_color(color_string):
    if color_string is None:
        return [255, 255, 255]

    text = str(color_string).strip()
    color = [255, 255, 255]

    if "," in text:
        try:
            values = [float(channel.strip()) for channel in text.split(",")]
            if all(0.0 <= value <= 1.0 for value in values):
                color = [int(value * 255) for value in values]
            else:
                color = [int(value) for value in values]
        except Exception:
            color = [255, 255, 255]
    else:
        try:
            color = list(ImageColor.getrgb(text))
        except Exception:
            try:
                value = float(text)
                value = int(value * 255) if 0.0 <= value <= 1.0 else int(value)
                color = [value, value, value]
            except Exception:
                color = [255, 255, 255]

    color = np.clip(color, 0, 255).astype(np.int32).tolist()
    if len(color) < 3:
        color = (color + [color[-1] if color else 255] * 3)[:3]
    return color[:4]


def _mxd_image_batch(image):
    if image is None:
        return None
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"[ImageAndMaskPreviewMXD] Expected IMAGE tensor [B,H,W,C], got shape {tuple(image.shape)}")
    return image.to(dtype=torch.float32)


def _mxd_mask_batch(mask, height=None, width=None, batch_size=None, device=None):
    if mask is None:
        return None

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    elif mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]

    if mask.ndim != 3:
        raise ValueError(f"[ImageAndMaskPreviewMXD] Expected MASK tensor [B,H,W], got shape {tuple(mask.shape)}")

    mask = mask.to(dtype=torch.float32, device=device if device is not None else mask.device).clamp(0.0, 1.0)

    if height is not None and width is not None and (mask.shape[-2] != height or mask.shape[-1] != width):
        mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)

    if batch_size is not None:
        mask = comfy.utils.repeat_to_batch_size(mask, batch_size)

    return mask


class ImageAndMaskPreviewMXD(SaveImage):
    DESCRIPTION = """Return an image with a mask composited over it without creating a node preview."""
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite",)
    FUNCTION = "execute"
    CATEGORY = "MXD/Image"
    OUTPUT_NODE = False

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_color": ("STRING", {"default": "255, 255, 255", "tooltip": "RGB/RGBA CSV, hex, or color name."}),
                "pass_through": ("BOOLEAN", {"default": True, "tooltip": "Legacy option. This node now always returns the composite without creating a preview."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def _build_composite(self, image=None, mask=None, mask_opacity=1.0, mask_color="255, 255, 255"):
        image = _mxd_image_batch(image)

        if image is None and mask is None:
            raise ValueError("[ImageAndMaskPreviewMXD] Connect an image, a mask, or both.")

        if image is None:
            mask = _mxd_mask_batch(mask)
            return mask.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

        if image.shape[-1] == 1:
            image = image.expand(-1, -1, -1, 3).clone()
        elif image.shape[-1] >= 3:
            image = image[..., :3].clone()
        else:
            raise ValueError(f"[ImageAndMaskPreviewMXD] Expected IMAGE tensor with 1 or more channels, got shape {tuple(image.shape)}")
        if mask is None:
            return image

        batch_size, height, width, channels = image.shape
        mask = _mxd_mask_batch(mask, height, width, batch_size, image.device)
        color = _parse_mxd_mask_color(mask_color)
        alpha = mask.mul(float(mask_opacity)).clamp(0.0, 1.0)
        if len(color) == 4:
            alpha = alpha * (color[3] / 255.0)

        rgb = torch.tensor(color[:3], dtype=image.dtype, device=image.device).view(1, 1, 1, channels) / 255.0
        alpha = alpha.unsqueeze(-1)
        return (image * (1.0 - alpha) + rgb * alpha).clamp(0.0, 1.0)

    def execute(self, mask_opacity, mask_color, pass_through, filename_prefix="ComfyUI", image=None, mask=None, prompt=None, extra_pnginfo=None):
        composite = self._build_composite(image=image, mask=mask, mask_opacity=mask_opacity, mask_color=mask_color)
        return (composite,)

########################################################################################################################

NODE_CLASS_MAPPINGS = {
    "LatentHalfMasks": LatentHalfMasks,
    "Get Latent Size": GetLatentSizeMXD,
    "Place Image By Mask": PlaceImageByMask,
    "Crop Image By Mask": CropImageByMask,
    "SmartCropByMaskMXD": SmartCropByMaskMXD,
    "BboxDetectorCombinedBatchMXD": BboxDetectorCombinedBatchMXD,
    "ImageAndMaskPreviewMXD": ImageAndMaskPreviewMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentHalfMasks": "Latent to L/R Masks MXD",
    "Get Latent Size": "Get Latent Size MXD",
    "Place Image By Mask": "Place Image by Mask MXD",
    "Crop Image By Mask": "Crop Image by Mask MXD",
    "SmartCropByMaskMXD": "Smart Crop by Mask MXD",
    "BboxDetectorCombinedBatchMXD": "BBOX Detector Combined Batch MXD",
    "ImageAndMaskPreviewMXD": "Image and Mask Preview MXD",
}
