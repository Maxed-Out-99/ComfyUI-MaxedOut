"""Video frame utilities and video I/O nodes.

Registered nodes (always):
  Frames_Select_StartEnd_MXD   Select Frames MXD
  Frames_Remove_From_Start_MXD Remove Frames MXD
  GroupVideoFramesMXD          Group Video Frames MXD

Registered nodes (only when HAVE_COMFY_API):
  CombineVideos_MXD            Combine Videos MXD
  CreateAndSaveVideoMXD        Save Video MXD (creates and saves in one node)
  LoadVideoMXD                 Load Video MXD (also outputs images/audio/fps/
                               bit_depth, like Get Video Components, in one node)
  SaveVideoMXD                 Save Wan22 Video MXD (merges a prior stage's workflow
                               into the embedded metadata via latent_io helpers)
  PreviewVideoMXD              Preview Video MXD

"""
from __future__ import annotations
import os
from fractions import Fraction

import torch

import folder_paths
import comfy.model_management
from comfy.cli_args import args

# Comfy API
try:
    from comfy_api.latest import io, ui
    from comfy_api.input import VideoInput
    from comfy_api.input_impl import VideoFromFile, VideoFromComponents
    from comfy_api.util import VideoComponents, VideoContainer, VideoCodec
    HAVE_COMFY_API = True
except Exception as _e:
    io = None
    ui = None
    VideoInput = None
    VideoFromFile = None
    VideoFromComponents = None
    VideoComponents = None
    VideoContainer = None
    VideoCodec = None
    HAVE_COMFY_API = False
    print(f"[ComfyUI-MaxedOut] comfy_api not available in wan22.video_ops: {_e}")

from .latent_io import _merge_prior_workflow_into_current

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _frame_window(total, count, offset, mode):
    offset = max(1, min(offset, total))
    count = max(1, min(count, total - offset + 1))

    if mode == "start":
        start_idx = offset - 1
        end_idx = start_idx + count
    elif mode == "end":
        start_idx = max(0, total - offset - count + 1)
        end_idx = start_idx + count
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'start' or 'end'.")

    return start_idx, end_idx


def _select_frames_start_end(frames, count=1, offset=1, mode="end"):
    total = int(frames.shape[0])
    if total <= 0:
        raise ValueError("No frames available for selection.")

    start_idx, end_idx = _frame_window(total, count, offset, mode)
    return frames[start_idx:end_idx].clone()


def _remove_frames_start_end(frames, count=1, offset=1, mode="start"):
    total = int(frames.shape[0])
    if total <= 0:
        raise ValueError("No frames available for removal.")

    start_idx, end_idx = _frame_window(total, count, offset, mode)
    remaining = torch.cat([frames[:start_idx], frames[end_idx:]], dim=0).clone()
    if remaining.shape[0] == 0:
        raise ValueError("Removing this window would leave no frames.")

    return remaining


# ---------- MXD Frames Select Start/End (from start or end of sequence) ----------
class Frames_Select_StartEnd_MXD:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of frames to select"
                }),
                "offset": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "How far into the video to start selection (from start or end)"
                }),
                "mode": (["start", "end"], {
                    "default": "end",
                    "tooltip": "Select frames from the start or end of the sequence"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "main"
    CATEGORY     = "MXD/images"

    def main(self, frames=None, count=1, offset=1, mode="end"):
        selected = _select_frames_start_end(frames, count=count, offset=offset, mode=mode)
        return (selected,)


# ---------- MXD Frames Remove (from start or end of sequence) ----------
class FramesRemoveMXD:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "count": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of frames to remove"
                }),
                "offset": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "How far into the video to start removal (from start or end)"
                }),
                "mode": (["start", "end"], {
                    "default": "start",
                    "tooltip": "Remove frames from the start or end of the sequence"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "main"
    CATEGORY     = "MXD/images"

    def main(self, frames=None, count=10, offset=1, mode="start"):
        remaining = _remove_frames_start_end(frames, count=count, offset=offset, mode=mode)
        return (remaining,)


# Keep this published node's schema frozen for existing workflows.
class Frames_Remove_From_Start_MXD:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "count": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of frames to remove from the start"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "MXD/images"

    def main(self, frames=None, count=10):
        return (frames[count:].clone(),)


class GroupVideoFramesMXD:
    CATEGORY = "MXD/Video"
    TITLE = "Group Video Frames (MXD)"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE_GROUPS",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "group_frames"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "group_size": ("INT", {"default": 81, "min": 1, "max": 5000, "step": 1}),
            }
        }

    def group_frames(self, frames, group_size):
        import math, torch

        all_frames = list(frames)
        total = len(all_frames)
        num_groups = math.ceil(total / group_size)
        grouped_tensors = []

        for i in range(num_groups):
            start = i * group_size
            end = min(start + group_size, total)
            group = all_frames[start:end]

            clean = []
            for f in group:
                # drop redundant singleton batch dim if present
                if f.ndim == 4 and f.shape[0] == 1:
                    f = f.squeeze(0)  # (H,W,C)
                # ensure shape (H,W,C)
                if f.ndim != 3:
                    print(f"[GroupVideoFramesMXD] weird frame shape {f.shape}")
                    continue
                clean.append(f)

            # stack back to (N,H,W,C)
            if len(clean) == 0:
                continue
            stacked = torch.stack(clean, dim=0)
            grouped_tensors.append(stacked)

        print(f"[GroupVideoFramesMXD] Split {total} frames into {len(grouped_tensors)} groups of up to {group_size}.")
        return (grouped_tensors,)


if HAVE_COMFY_API:
    class CombineVideos_MXD:
        """
        Combine two VIDEO inputs end-to-end (sequentially).
        """

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "front_video": ("VIDEO", {"tooltip": "The first video (plays first)"}),
                    "back_video": ("VIDEO", {"tooltip": "The second video (plays after the first)"}),
                },
            }

        RETURN_TYPES = ("VIDEO",)
        RETURN_NAMES = ("video",)
        FUNCTION = "combine"
        CATEGORY = "MXD/video"

        def combine(self, front_video, back_video):
            comp_a = front_video.get_components()
            comp_b = back_video.get_components()

            # Check frame rate consistency
            if comp_a.frame_rate != comp_b.frame_rate:
                raise ValueError(f"FPS mismatch: {comp_a.frame_rate} vs {comp_b.frame_rate}")

            # Concatenate frame tensors along batch/time dimension (dim=0)
            frames_a = torch.stack(comp_a.images) if isinstance(comp_a.images, list) else comp_a.images
            frames_b = torch.stack(comp_b.images) if isinstance(comp_b.images, list) else comp_b.images
            if frames_a.shape[1] != frames_b.shape[1] or frames_a.shape[2] != frames_b.shape[2]:
                raise ValueError(
                    "Resolution mismatch in CombineVideos_MXD: "
                    f"front_video={frames_a.shape[2]}x{frames_a.shape[1]}, "
                    f"back_video={frames_b.shape[2]}x{frames_b.shape[1]}. "
                    "Use 'WAN 2.2 Video Prep I2V MXD' before WAN generation so scaled base video and generated clip match."
                )
            combined_images = torch.cat([frames_a, frames_b], dim=0)

            # Combine audio sequentially
            combined_audio = None
            if comp_a.audio is not None or comp_b.audio is not None:
                def _extract_audio(audio_obj):
                    if audio_obj is None:
                        return None, None, None, None
                    if torch.is_tensor(audio_obj):
                        return audio_obj, None, "tensor", None
                    if isinstance(audio_obj, dict):
                        wave_key = "waveform" if "waveform" in audio_obj else ("samples" if "samples" in audio_obj else None)
                        if wave_key is None or not torch.is_tensor(audio_obj.get(wave_key)):
                            raise TypeError(f"Unsupported audio dict format. Keys: {list(audio_obj.keys())}")
                        return audio_obj[wave_key], audio_obj.get("sample_rate"), "dict", wave_key
                    waveform = getattr(audio_obj, "waveform", None)
                    sample_rate = getattr(audio_obj, "sample_rate", None)
                    if torch.is_tensor(waveform):
                        return waveform, sample_rate, "object", None
                    raise TypeError(f"Unsupported audio payload type: {type(audio_obj).__name__}")

                wave_a, sr_a, kind_a, wave_key_a = _extract_audio(comp_a.audio)
                wave_b, sr_b, kind_b, wave_key_b = _extract_audio(comp_b.audio)
                rank_a = wave_a.ndim if wave_a is not None else None
                rank_b = wave_b.ndim if wave_b is not None else None

                def _to_bct(w):
                    if w is None:
                        return None
                    if w.ndim == 1:
                        return w.unsqueeze(0).unsqueeze(0)  # [1,1,T]
                    if w.ndim == 2:
                        return w.unsqueeze(0)  # [1,C,T]
                    if w.ndim == 3:
                        return w  # [B,C,T]
                    raise ValueError(f"Unsupported audio tensor rank: {w.ndim}")

                wave_a = _to_bct(wave_a)
                wave_b = _to_bct(wave_b)

                if wave_a is None and wave_b is not None:
                    wave_a = torch.zeros((wave_b.shape[0], wave_b.shape[1], 0), dtype=wave_b.dtype, device=wave_b.device)
                if wave_b is None and wave_a is not None:
                    wave_b = torch.zeros((wave_a.shape[0], wave_a.shape[1], 0), dtype=wave_a.dtype, device=wave_a.device)

                if wave_a is not None and wave_b is not None:
                    if wave_a.shape[0] != wave_b.shape[0]:
                        if wave_a.shape[0] == 1:
                            wave_a = wave_a.expand(wave_b.shape[0], -1, -1)
                        elif wave_b.shape[0] == 1:
                            wave_b = wave_b.expand(wave_a.shape[0], -1, -1)
                        else:
                            raise ValueError(f"Audio batch mismatch: {wave_a.shape[0]} vs {wave_b.shape[0]}")

                    if wave_a.shape[1] != wave_b.shape[1]:
                        if wave_a.shape[1] == 1:
                            wave_a = wave_a.expand(-1, wave_b.shape[1], -1)
                        elif wave_b.shape[1] == 1:
                            wave_b = wave_b.expand(-1, wave_a.shape[1], -1)
                        else:
                            raise ValueError(f"Audio channel mismatch: {wave_a.shape[1]} vs {wave_b.shape[1]}")

                if sr_a is not None and sr_b is not None and sr_a != sr_b:
                    raise ValueError(f"Audio sample-rate mismatch: {sr_a} vs {sr_b}")

                combined_wave = torch.cat([wave_a, wave_b], dim=2)
                out_sr = sr_a if sr_a is not None else sr_b

                target_rank = rank_a if rank_a is not None else rank_b
                if target_rank == 1 and combined_wave.shape[0] == 1 and combined_wave.shape[1] == 1:
                    combined_wave = combined_wave.squeeze(0).squeeze(0)
                elif target_rank == 2 and combined_wave.shape[0] == 1:
                    combined_wave = combined_wave.squeeze(0)

                out_kind = kind_a if kind_a is not None else kind_b
                if out_kind == "dict":
                    out_key = wave_key_a if kind_a == "dict" else wave_key_b
                    combined_audio = {out_key or "waveform": combined_wave}
                    if out_sr is not None:
                        combined_audio["sample_rate"] = out_sr
                else:
                    combined_audio = combined_wave

            combined_video = VideoFromComponents(
                VideoComponents(
                    images=combined_images,
                    audio=combined_audio,
                    frame_rate=comp_a.frame_rate,
                )
            )

            return (combined_video,)

    # ---------- Load Video MXD ----------
    class LoadVideoComponentsMXD:
        """Load a video from /input (videos only).

        Also extracts components (images/audio/fps/bit_depth) inline so this
        node covers what LoadVideo + GetVideoComponents would otherwise take two
        nodes to do.
        """

        CATEGORY = "image/video"
        FUNCTION = "load"
        RETURN_TYPES = ("VIDEO", "IMAGE", "AUDIO", "FLOAT", "INT")
        RETURN_NAMES = ("video", "images", "audio", "fps", "bit_depth")
        TITLE = "Load Video MXD"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "file": ("COMBO", {
                        "video_upload": True,
                    }),
                }
            }

        # --- helpers --------------------------------------------------------------

        @staticmethod
        def _resolve_video_path(file: str) -> str:
            """
            Try to resolve `file` in a backwards-compatible way:
            1. If it's an annotated path, let folder_paths handle it.
            2. Otherwise treat it as relative to the input directory.
            """
            # 1) Try annotated style (old workflows / uploads)
            try:
                return folder_paths.get_annotated_filepath(file)
            except Exception:
                pass

            # 2) Fall back to /input relative
            base = folder_paths.get_input_directory()
            candidate = os.path.join(base, file)
            if os.path.isfile(candidate):
                return candidate

            # If all else fails, just return what we got (will error later)
            return candidate

        @staticmethod
        def _is_video_file(path: str) -> bool:
            _, ext = os.path.splitext(path)
            return ext.lower() in VIDEO_EXTS

        # --- main function --------------------------------------------------------

        def load(self, file: str):
            video_path = self._resolve_video_path(file)

            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"[LoadVideoMXD] File not found: {video_path}")

            if not self._is_video_file(video_path):
                raise ValueError(f"[LoadVideoMXD] Not a video file: {video_path}")

            print(f"[LoadVideoMXD] Loaded exactly: {video_path}")
            video = VideoFromFile(video_path)
            components = video.get_components()
            bit_depth = video.get_bit_depth()
            return (video, components.images, components.audio, float(components.frame_rate), bit_depth)

        # --- nice-to-haves --------------------------------------------------------

        @classmethod
        def IS_CHANGED(cls, file: str):
            try:
                p = cls._resolve_video_path(file)
                return os.path.getmtime(p)
            except Exception:
                return 0

        @classmethod
        def VALIDATE_INPUTS(cls, file: str):
            # First, try the annotated path (for backwards compat)
            if folder_paths.exists_annotated_filepath(file):
                resolved = folder_paths.get_annotated_filepath(file)
                if not cls._is_video_file(resolved):
                    return f"This node only accepts video files ({', '.join(sorted(VIDEO_EXTS))})."
                return True

            # Then, try treating it as /input-relative
            base = folder_paths.get_input_directory()
            candidate = os.path.join(base, file)
            if os.path.isfile(candidate):
                if not cls._is_video_file(candidate):
                    return f"This node only accepts video files ({', '.join(sorted(VIDEO_EXTS))})."
                return True

            return f"Invalid video file: {file}"

    # Keep this published node's inputs and outputs frozen for existing workflows.
    class LoadVideoMXD(LoadVideoComponentsMXD):
        RETURN_TYPES = ("VIDEO", "STRING")
        RETURN_NAMES = ("video", "video_path")
        TITLE = "Load Video MXD"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "file": ("COMBO", {
                        "video_upload": True,
                        "remote": {
                            "route": "/mxd/videos/input",
                            "refresh_button": True,
                            "control_after_refresh": "first",
                        },
                    }),
                }
            }

        def load(self, file: str):
            video_path = self._resolve_video_path(file)
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"[LoadVideoMXD] File not found: {video_path}")
            if not self._is_video_file(video_path):
                raise ValueError(f"[LoadVideoMXD] Not a video file: {video_path}")
            print(f"[LoadVideoMXD] Loaded exactly: {video_path}")
            return (VideoFromFile(video_path), video_path)

    # ---------- Save Video MXD ----------
    class SaveVideoMXD(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="SaveVideoMXD",
                display_name="Save Video MXD",
                category="image/video",
                description="Saves the input video to your ComfyUI output directory.",
                inputs=[
                    io.Video.Input("video", tooltip="The video to save."),
                    io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                    io.Combo.Input("format", options=["auto", "mp4"], default="auto", tooltip="The format to save the video as."),
                    io.Combo.Input("codec", options=["auto", "h264"], default="auto", tooltip="The codec to use for the video."),
                    io.Boolean.Input(
                        "embed_workflow",
                        default=True,
                        label_on="embed",
                        label_off="skip",
                        tooltip="When high_workflow is connected, merge it into this video's embedded workflow "
                                "so dragging the final video into ComfyUI shows both the high-noise stage and "
                                "this stage together.",
                    ),
                    io.String.Input(
                        "high_workflow",
                        optional=True,
                        force_input=True,
                        tooltip="Connect a Load Latent node's 'high_workflow' output here to carry the "
                                "high-noise stage's workflow into this video's metadata.",
                    ),
                ],
                hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
                is_output_node=True,
            )

        @classmethod
        def execute(cls, video: VideoInput, filename_prefix: str, format: str, codec: str,
                    embed_workflow: bool = True, high_workflow: str = "") -> io.NodeOutput:
            width, height = video.get_dimensions()
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix,
                folder_paths.get_output_directory(),
                width,
                height
            )

            saved_metadata = None
            if not args.disable_metadata:
                metadata = {}
                if cls.hidden.extra_pnginfo is not None:
                    metadata.update(cls.hidden.extra_pnginfo)
                if cls.hidden.prompt is not None:
                    metadata["prompt"] = cls.hidden.prompt
                if embed_workflow and high_workflow:
                    current_workflow = metadata.get("workflow")
                    merged_workflow = _merge_prior_workflow_into_current(high_workflow, current_workflow)
                    if merged_workflow is not current_workflow:
                        metadata["workflow"] = merged_workflow
                if len(metadata) > 0:
                    saved_metadata = metadata

            file = f"{filename}_{counter:05}_.{VideoContainer.get_extension(format)}"
            video.save_to(
                os.path.join(full_output_folder, file),
                format=VideoContainer(format),
                codec=codec,
                metadata=saved_metadata
            )

            return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))

    # ---------- Create + Save Video MXD ----------
    class CreateAndSaveVideoMXD(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="CreateAndSaveVideoMXD",
                display_name="Create and Save Video MXD",
                search_aliases=["create video", "images to video", "export video"],
                category="video",
                description="Creates a video from images and saves it to the ComfyUI output directory.",
                inputs=[
                    io.Image.Input("images", tooltip="The images to create a video from."),
                    io.Float.Input("fps", default=30.0, min=1.0, max=120.0, step=1.0),
                    io.String.Input(
                        "filename_prefix",
                        default="video/ComfyUI",
                        tooltip="The prefix for the saved file. This may include formatting information.",
                    ),
                    io.Combo.Input(
                        "format",
                        options=VideoContainer.as_input(),
                        default="auto",
                        tooltip="The format to save the video as.",
                    ),
                    io.DynamicCombo.Input(
                        "codec",
                        options=[
                            io.DynamicCombo.Option("auto", []),
                            io.DynamicCombo.Option(
                                "h264",
                                [
                                    io.DynamicCombo.Input(
                                        "encoding",
                                        display_name="encoding mode",
                                        options=[
                                            io.DynamicCombo.Option("auto", []),
                                            io.DynamicCombo.Option(
                                                "re-encode",
                                                [
                                                    io.Float.Input(
                                                        "crf",
                                                        default=23.0,
                                                        min=0.0,
                                                        max=51.0,
                                                        step=1.0,
                                                        tooltip="Lower values produce higher quality and larger files.",
                                                    )
                                                ],
                                            ),
                                        ],
                                        optional=True,
                                        tooltip="Automatic preserves compatible H.264 streams. Re-encode applies a custom CRF.",
                                    )
                                ],
                            ),
                        ],
                        tooltip="The codec to use for the video.",
                    ),
                    io.Audio.Input("audio", optional=True, tooltip="The audio to add to the video."),
                    io.Int.Input(
                        "bit_depth",
                        min=8,
                        max=10,
                        default=8,
                        step=2,
                        optional=True,
                        display_mode=io.NumberDisplay.number,
                        tooltip="10-bit keeps smoother gradients, but some players and nodes may not support it.",
                    ),
                ],
                hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
                outputs=[io.Video.Output("video")],
                is_output_node=True,
            )

        @classmethod
        def execute(
            cls,
            images,
            fps: float,
            filename_prefix: str,
            format: str,
            codec: io.DynamicCombo.Type,
            audio=None,
            bit_depth: int = 8,
        ) -> io.NodeOutput:
            video = VideoFromComponents(
                VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps)),
                bit_depth=bit_depth,
            )
            codec_name = codec["codec"]
            encoding = codec.get("encoding") or {}
            width, height = video.get_dimensions()
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix,
                folder_paths.get_output_directory(),
                width,
                height,
            )

            saved_metadata = None
            if not args.disable_metadata:
                metadata = {}
                if cls.hidden.extra_pnginfo is not None:
                    metadata.update(cls.hidden.extra_pnginfo)
                if cls.hidden.prompt is not None:
                    metadata["prompt"] = cls.hidden.prompt
                if metadata:
                    saved_metadata = metadata

            file = f"{filename}_{counter:05}_.{VideoContainer.get_extension(format)}"
            video.save_to(
                os.path.join(full_output_folder, file),
                format=VideoContainer(format),
                codec=codec_name,
                metadata=saved_metadata,
                crf=encoding.get("crf"),
            )

            return io.NodeOutput(
                video,
                ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
            )

    class PreviewVideoMXD(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="PreviewVideoMXD",
                display_name="Preview Video MXD",
                category="image/video",
                description="Preview a video without saving output (optional pass-through).",
                inputs=[
                    io.Video.Input("input_video", tooltip="Video to preview."),
                ],
                outputs=[
                    io.Video.Output("output_video", tooltip="Passes the same video forward."),
                ],
                # Allow this node to run even when output_video is not connected.
                is_output_node=True,
            )

        @classmethod
        def execute(cls, input_video: VideoInput):
            # Save a temporary H264 file so ComfyUI has something to preview
            out_dir = os.path.join(folder_paths.get_output_directory(), "previews")
            os.makedirs(out_dir, exist_ok=True)

            preview_path = os.path.join(out_dir, "preview_temp.mp4")
            input_video.save_to(preview_path, format="mp4", codec="h264")

            # Return the raw video object (not a tuple)
            return io.NodeOutput(
                input_video,
                ui=ui.PreviewVideo([
                    ui.SavedResult("preview_temp.mp4", "previews", io.FolderType.output)
                ])
            )


NODE_CLASS_MAPPINGS = {
    "Frames_Remove_From_Start_MXD": Frames_Remove_From_Start_MXD,
    "FramesRemoveMXD": FramesRemoveMXD,
    "GroupVideoFramesMXD": GroupVideoFramesMXD,
    "Frames_Select_StartEnd_MXD": Frames_Select_StartEnd_MXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Frames_Remove_From_Start_MXD": "Remove Frames From Start MXD",
    "FramesRemoveMXD": "Remove Frames MXD",
    "GroupVideoFramesMXD": "Group Video Frames MXD",
    "Frames_Select_StartEnd_MXD": "Select Frames MXD",
}

if HAVE_COMFY_API:
    NODE_CLASS_MAPPINGS.update({
        "CombineVideos_MXD": CombineVideos_MXD,
        "CreateAndSaveVideoMXD": CreateAndSaveVideoMXD,
        "LoadVideoMXD": LoadVideoMXD,
        "LoadVideoComponentsMXD": LoadVideoComponentsMXD,
        "SaveVideoMXD": SaveVideoMXD,
        "PreviewVideoMXD": PreviewVideoMXD,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "CombineVideos_MXD": "Combine Videos MXD",
        "CreateAndSaveVideoMXD": "Create and Save Video MXD",
        "LoadVideoMXD": "Load Video MXD",
        "LoadVideoComponentsMXD": "Load Video + Components MXD",
        "SaveVideoMXD": "Save Video MXD",
        "PreviewVideoMXD": "Preview Video MXD",
    })
