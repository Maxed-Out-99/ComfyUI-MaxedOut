from comfy_api.latest import io
import torch
import node_helpers
import comfy
import comfy.latent_formats


class Wan22FirstLastImageToVideoSVIMXD(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Wan22FirstLastImageToVideoSVIMXD",
            display_name="WAN 2.2 First/Last I2V SVI MXD",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("length", default=81, min=1, max=8192, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("end_image", optional=True),
                io.Latent.Input("prev_latent", optional=True),
                io.Int.Input("continue_frames_count", default=5, min=0, max=20, step=1, optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        length,
        batch_size,
        start_image=None,
        end_image=None,
        prev_latent=None,
        continue_frames_count=5,
    ) -> io.NodeOutput:
        _ = end_image  # SVI-only node keeps this input for wiring compatibility and intentionally ignores it.

        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        total_latents = ((length - 1) // 4) + 1
        device = comfy.model_management.intermediate_device()

        prev_samples = None
        if prev_latent is not None:
            if not isinstance(prev_latent, dict) or prev_latent.get("samples") is None:
                raise ValueError("prev_latent was provided but does not contain a valid 'samples' tensor.")
            prev_samples = prev_latent["samples"]

        if start_image is None and prev_samples is None:
            raise ValueError("SVI node requires either start_image or prev_latent.")

        if start_image is not None:
            anchor_latent = vae.encode(start_image[:1, :, :, :3])
            h = anchor_latent.shape[-2]
            w = anchor_latent.shape[-1]
        else:
            h = prev_samples.shape[-2]
            w = prev_samples.shape[-1]
            anchor_latent = torch.zeros(
                [1, latent_channels, 1, h, w],
                device=device,
            )

        latent = torch.zeros(
            [batch_size, latent_channels, total_latents, h, w],
            device=device,
        )

        cond_parts = [anchor_latent]
        anchor_t = anchor_latent.shape[2]
        motion_t_limit = max(0, total_latents - anchor_t)

        if prev_samples is not None and continue_frames_count > 0 and motion_t_limit > 0:
            motion_t = min(continue_frames_count, prev_samples.shape[2])
            motion_latent = prev_samples[:, :, -motion_t:].clone()

            if motion_latent.shape[-2] != h or motion_latent.shape[-1] != w:
                raise ValueError("prev_latent spatial size does not match current SVI anchor latent size.")

            if motion_latent.shape[2] > motion_t_limit:
                motion_latent = motion_latent[:, :, -motion_t_limit:]

            cond_parts.append(motion_latent)

        image_cond_latent = torch.cat(cond_parts, dim=2)
        padding_size = total_latents - image_cond_latent.shape[2]

        if padding_size > 0:
            padding = torch.zeros(
                [1, latent_channels, padding_size, h, w],
                dtype=image_cond_latent.dtype,
                device=image_cond_latent.device,
            )
            padding = comfy.latent_formats.Wan21().process_out(padding)
            image_cond_latent = torch.cat([image_cond_latent, padding], dim=2)

        # SVI uses latent-time masks directly (1,1,T,H,W), unlike frame-domain 4-phase mask reshaping used by non-SVI nodes.
        mask_svi = torch.ones(
            [1, 1, total_latents, h, w],
            device=image_cond_latent.device,
            dtype=image_cond_latent.dtype,
        )
        mask_svi[:, :, :1] = 0.0

        conditioning_values = {
            "concat_latent_image": image_cond_latent,
            "concat_mask": mask_svi,
        }
        positive_out = node_helpers.conditioning_set_values(positive, conditioning_values)
        negative_out = node_helpers.conditioning_set_values(negative, conditioning_values)

        out_latent = {"samples": latent}
        return io.NodeOutput(positive_out, negative_out, out_latent)


NODE_CLASS_MAPPINGS = {
    "Wan22FirstLastImageToVideoSVIMXD": Wan22FirstLastImageToVideoSVIMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan22FirstLastImageToVideoSVIMXD": "WAN 2.2 First/Last I2V SVI MXD",
}
