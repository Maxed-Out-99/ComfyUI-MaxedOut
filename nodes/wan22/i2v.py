"""WAN 2.2 image-to-video conditioning nodes (all require comfy_api; skipped when absent).

Registered nodes (only when HAVE_COMFY_API):
  Wan22ImageToVideoMXD           Wan 2.2 Image to Video MXD
  WAN22_I2V_Video_Prep_MXD       WAN 2.2 Video Prep I2V MXD
  Wan22FirstLastImageToVideoMXD  Wan 2.2 I2V First & Last Frame MXD

These expect pre-sized inputs (use the buckets.py scaler upstream); they do no
scaling or CLIP-vision of their own.
"""
from __future__ import annotations

import torch

import comfy.model_management
import node_helpers, nodes

# Comfy API
try:
    from comfy_api.latest import io
    from comfy_api.input_impl import VideoFromComponents
    from comfy_api.util import VideoComponents
    HAVE_COMFY_API = True
except Exception as _e:
    io = None
    VideoFromComponents = None
    VideoComponents = None
    HAVE_COMFY_API = False
    print(f"[ComfyUI-MaxedOut] comfy_api not available in wan22.i2v: {_e}")

from .buckets import _wan22_scale_image_core


def build_wan22_i2v_conditioning(positive, negative, vae, length, start_image):
    """Bake `start_image` into WAN 2.2 I2V conditioning; returns (positive, negative).

    `start_image` is [F, H, W, C] and already pre-sized; its first `min(F, length)`
    frames become the known frames, the rest of the clip is filled with neutral
    grey and left for the model. Split out of Wan22ImageToVideoMXD so a caller
    can rebuild the conditioning from a crop of the reference frame -- encoding
    a crop is exact, where cropping the encode would not be.
    """
    if start_image is None:
        raise ValueError("start_image must be provided (already pre-sized).")

    frames_in, ih, iw, ch = start_image.shape
    frames_used = min(frames_in, length)
    t = ((length - 1) // 4) + 1

    # create placeholder image tensor
    image = torch.ones(
        (length, ih, iw, ch),
        device=start_image.device,
        dtype=start_image.dtype
    ) * 0.5
    image[:frames_used] = start_image[:frames_used]

    # encode using VAE
    concat_latent_image = vae.encode(image[:, :, :, :3])

    # mask zeros out the frames used
    mask = torch.ones(
        (1, 1, t, concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
        device=image.device,
        dtype=image.dtype
    )
    mask[:, :, :((frames_used - 1) // 4) + 1] = 0.0

    positive = node_helpers.conditioning_set_values(
        positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
    )
    negative = node_helpers.conditioning_set_values(
        negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
    )
    return positive, negative


def build_wan22_flf_conditioning(positive, negative, vae, length, height, width,
                                 start_image=None, end_image=None):
    """Bake start and/or end frames into WAN 2.2 conditioning; returns (positive, negative).

    The first/last variant of build_wan22_i2v_conditioning: known frames are
    pinned at both ends of the clip and the mask frees only the middle. Note the
    mask layout differs from the plain I2V one -- it is built at full frame rate
    and folded to (1, 4, t, h, w) rather than (1, 1, t, h, w).

    Split out of Wan22FirstLastImageToVideoMXD so a caller can rebuild the
    conditioning from crops of the reference frames.
    """
    if start_image is None and end_image is None:
        raise ValueError("at least one of start_image / end_image must be provided.")

    spacial_scale = vae.spacial_compression_encode()
    latent_length = ((length - 1) // 4) + 1

    image = torch.ones((length, height, width, 3)) * 0.5
    mask = torch.ones(
        (1, 1, latent_length * 4, height // spacial_scale, width // spacial_scale)
    )

    if start_image is not None:
        image[:start_image.shape[0]] = start_image
        mask[:, :, :start_image.shape[0] + 3] = 0.0

    if end_image is not None:
        image[-end_image.shape[0]:] = end_image
        mask[:, :, -end_image.shape[0]:] = 0.0

    concat_latent_image = vae.encode(image[:, :, :, :3])
    mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)

    positive = node_helpers.conditioning_set_values(
        positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
    )
    negative = node_helpers.conditioning_set_values(
        negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
    )
    return positive, negative


def _resample_video_frames_to_fps(frames, in_fps, out_fps):
    """
    Resample a frame sequence to a target FPS using nearest-frame selection.
    Preserves clip duration approximately by dropping/duplicating frames,
    instead of only changing FPS metadata (which changes playback speed).
    Returns (frames_out, fps_out, changed).
    """
    if frames is None or frames.ndim != 4:
        raise ValueError("Expected frame tensor with shape [T,H,W,C].")

    if in_fps is None:
        raise ValueError("Input video FPS is missing; cannot force FPS safely.")

    in_fps = float(in_fps)
    out_fps = float(out_fps)
    if in_fps <= 0:
        raise ValueError(f"Invalid input FPS: {in_fps}")
    if out_fps <= 0:
        raise ValueError(f"Invalid target FPS: {out_fps}")

    if frames.shape[0] <= 1:
        return frames, float(out_fps), False

    if abs(in_fps - out_fps) < 1e-6:
        return frames, float(out_fps), False

    n_in = int(frames.shape[0])
    # Match the first/last frame span, then pick nearest frames on that timeline.
    n_out = max(1, int(round(((n_in - 1) * out_fps) / in_fps)) + 1)
    if n_out == n_in:
        # Frame count may stay the same for near-equal FPS; metadata still becomes exact.
        return frames, float(out_fps), False

    idx = torch.linspace(0, n_in - 1, steps=n_out, device=frames.device)
    idx = idx.round().to(dtype=torch.long)
    out = frames.index_select(0, idx)
    return out, float(out_fps), True


# ---------- WAN 2.2 Image to Video (no scaling; expects pre-sized input) ----------
if HAVE_COMFY_API:
    class Wan22ImageToVideoMXD(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="Wan22ImageToVideoMXD",
                display_name="WAN 2.2 Image to Video MXD",
                category="conditioning/video_models",
                description="WAN 2.2 image to video without scaling or CLIP vision.",
                inputs=[
                    io.Conditioning.Input("positive"),
                    io.Conditioning.Input("negative"),
                    io.Vae.Input("vae"),
                    io.Int.Input("length", default=81, min=1, max=16384, step=4),
                    io.Int.Input("batch_size", default=1, min=1, max=4096),
                    io.Image.Input("start_image", optional=False),
                ],
                outputs=[
                    io.Conditioning.Output(display_name="positive"),
                    io.Conditioning.Output(display_name="negative"),
                    io.Latent.Output(display_name="latent"),
                ],
            )

        @classmethod
        def execute(cls, positive, negative, vae, length, batch_size, start_image) -> io.NodeOutput:
            if start_image is None:
                raise ValueError("start_image must be provided (already pre-sized).")

            frames_in, ih, iw, ch = start_image.shape
            t = ((length - 1) // 4) + 1

            latent = torch.zeros(
                [batch_size, 16, t, ih // 8, iw // 8],
                device=comfy.model_management.intermediate_device()
            )

            positive, negative = build_wan22_i2v_conditioning(
                positive, negative, vae, length, start_image
            )

            out_latent = {"samples": latent}
            return io.NodeOutput(positive, negative, out_latent)

    class WAN22_I2V_Video_Prep_MXD:
        """
        Prepare a source video for iterative WAN 2.2 extension:
        - scale entire video using WAN bucket logic
        - output the scaled frame batch directly
        - keep default workflow simple for common use
        """
        CATEGORY = "MXD/video"
        FUNCTION = "prepare"
        RETURN_TYPES = ("VIDEO", "IMAGE", "FLOAT")
        RETURN_NAMES = ("scaled_video", "images", "fps")

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "video": ("VIDEO",),
                    "tier": (["Auto", "480p", "720p"], {"default": "Auto"}),
                    "crop_to_fit": ("BOOLEAN", {
                        "default": True,
                        "label_on": "Perfect Fit (Crops Edges)",
                        "label_off": "Closest Fit (No Crop)"
                    }),
                    "force_fps": ("BOOLEAN", {
                        "default": False,
                        "label_on": "Force FPS",
                        "label_off": "Keep Source FPS",
                        "tooltip": "When enabled, resample frames (drop/duplicate) and set exact target fps."
                    }),
                    "target_fps": ("INT", {
                        "default": 16,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Used when Force FPS is enabled. Output video fps will be set exactly to this value."
                    }),
                    "aspect_mode": (["Auto", "Tall", "Wide", "Square"], {
                        "default": "Auto",
                        "tooltip": "Auto picks wide/tall/square from the source. Use Square/Tall/Wide to force the target bucket shape."
                    }),
                },
            }

        def prepare(self, video, tier="Auto", crop_to_fit=True, force_fps=False, target_fps=16, aspect_mode="Auto"):
            comp = video.get_components()
            if isinstance(comp.images, list):
                if len(comp.images) == 0:
                    raise ValueError("[WAN22_I2V_Video_Prep_MXD] Input video has zero frames.")
                frames = torch.stack(comp.images)
            else:
                frames = comp.images

            if frames is None:
                raise ValueError("[WAN22_I2V_Video_Prep_MXD] Input video has no frames.")
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            if frames.ndim != 4:
                raise ValueError(f"[WAN22_I2V_Video_Prep_MXD] Unexpected frame tensor shape: {tuple(frames.shape)}")
            if frames.shape[0] <= 0:
                raise ValueError("[WAN22_I2V_Video_Prep_MXD] Input video has zero frames.")

            out_frame_rate = float(comp.frame_rate) if comp.frame_rate is not None else None
            if force_fps:
                frames, out_frame_rate, _ = _resample_video_frames_to_fps(
                    frames, comp.frame_rate, target_fps
                )

            # "Auto" in video prep uses the safer extend-friendly behavior.
            # Keep accepting legacy "Safe Auto" values from older saved workflows.
            internal_tier = "Safe Auto" if tier == "Auto" else tier
            scaled_frames, _, _, _ = _wan22_scale_image_core(
                frames,
                tier=internal_tier,
                crop_to_fit=crop_to_fit,
                aspect_mode=aspect_mode,
            )

            scaled_video = VideoFromComponents(
                VideoComponents(
                    images=scaled_frames,
                    audio=comp.audio,
                    frame_rate=out_frame_rate,
                )
            )

            fps = float(out_frame_rate) if out_frame_rate is not None else 0.0
            return (scaled_video, scaled_frames, fps)

    class Wan22FirstLastImageToVideoMXD(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="Wan22FirstLastImageToVideoMXD",
                display_name="WAN 2.2 First & Last I2V MXD",
                category="conditioning/video_models",
                inputs=[
                    io.Conditioning.Input("positive"),
                    io.Conditioning.Input("negative"),
                    io.Vae.Input("vae"),
                    io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                    io.Int.Input("batch_size", default=1, min=1, max=4096),
                    io.Image.Input("start_image", optional=True),
                    io.Image.Input("end_image", optional=True),
                ],
                outputs=[
                    io.Conditioning.Output(display_name="positive"),
                    io.Conditioning.Output(display_name="negative"),
                    io.Latent.Output(display_name="latent"),
                ],
            )

        @classmethod
        def execute(cls, positive, negative, vae, length, batch_size, start_image=None, end_image=None) -> io.NodeOutput:
            spacial_scale = vae.spacial_compression_encode()

            # Assume incoming images are already pre-sized by upstream nodes.
            height, width = start_image.shape[1], start_image.shape[2] if start_image is not None else (vae.latent_channels * spacial_scale, vae.latent_channels * spacial_scale)

            latent = torch.zeros(
                [batch_size, vae.latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale],
                device=comfy.model_management.intermediate_device()
            )

            positive, negative = build_wan22_flf_conditioning(
                positive, negative, vae, length, height, width, start_image, end_image
            )

            out_latent = {"samples": latent}
            return io.NodeOutput(positive, negative, out_latent)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if HAVE_COMFY_API:
    NODE_CLASS_MAPPINGS.update({
        "Wan22ImageToVideoMXD": Wan22ImageToVideoMXD,
        "WAN22_I2V_Video_Prep_MXD": WAN22_I2V_Video_Prep_MXD,
        "Wan22FirstLastImageToVideoMXD": Wan22FirstLastImageToVideoMXD,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "Wan22ImageToVideoMXD": "Wan 2.2 Image to Video MXD",
        "WAN22_I2V_Video_Prep_MXD": "WAN 2.2 Video Prep I2V MXD",
        "Wan22FirstLastImageToVideoMXD": "Wan 2.2 I2V First & Last Frame MXD",
    })
