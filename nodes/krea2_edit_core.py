"""Internal Krea 2 Edit implementation used by the MXD wrapper nodes.

Adapted from ComfyUI-Krea2Edit by Conrad Locke:
https://github.com/lbouaraba/comfyui-krea2edit

Upstream revision: 86f886dac23013d88996e3a2e99093ba44d322fb
Upstream license: Apache License 2.0 (see THIRD_PARTY_LICENSES.md).

This file was modified for ComfyUI-MaxedOut by extracting only the image-fit
and diffusion-forward helpers needed by the MXD nodes. The upstream public
nodes, workflow, and packaging code are intentionally not duplicated.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding


def _imgids(bs, frame, height, width, device):
    ids = torch.zeros(height, width, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(height, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(width, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, height * width, 3).repeat(bs, 1, 1)


def _imgids_offset(bs, frame, grid_h, grid_w, target_h, target_w, device):
    """Build stride-1 positions centered within the target token grid."""
    off_h = max(0.0, (target_h - grid_h) / 2)
    off_w = max(0.0, (target_w - grid_w) / 2)
    ids = torch.zeros(grid_h, grid_w, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (
        torch.arange(grid_h, device=device, dtype=torch.float32) + off_h
    )[:, None]
    ids[..., 2] = (
        torch.arange(grid_w, device=device, dtype=torch.float32) + off_w
    )[None, :]
    return ids.reshape(1, grid_h * grid_w, 3).repeat(bs, 1, 1)


def _to_4d(value):
    """Convert (B,C,T,H,W) to (B*T,C,H,W); pass 4D tensors through."""
    if value.ndim == 5:
        batch, channels, frames, height, width = value.shape
        return value.reshape(batch * frames, channels, height, width)
    return value


def _fit_src(source, height, width):
    """Center-crop a source latent to the target aspect ratio, then resize."""
    source_h, source_w = source.shape[-2:]
    if (source_h, source_w) == (height, width):
        return source
    scale = max(height / source_h, width / source_w)
    crop_h = min(source_h, int(round(height / scale)))
    crop_w = min(source_w, int(round(width / scale)))
    top = (source_h - crop_h) // 2
    left = (source_w - crop_w) // 2
    source = source[..., top : top + crop_h, left : left + crop_w]
    return F.interpolate(source.float(), size=(height, width), mode="bilinear")


def fit_encode_image(image, vae, height, width, cache, key, fit_mode="crop"):
    """Fit an image in pixel space and VAE-encode it at the target grid."""
    key = key + (fit_mode,)
    if key in cache:
        return cache[key]

    print(
        f"[Krea 2 Edit MXD] source mode={fit_mode} "
        f"input={tuple(image.shape)} target_latent={height}x{width}",
        flush=True,
    )
    pixel_h, pixel_w = height * 8, width * 8
    source = image.movedim(-1, 1)
    image_h, image_w = source.shape[-2:]

    if fit_mode == "fit":
        scale = min(pixel_h / image_h, pixel_w / image_w)
        crop_tolerance = 0.08
        if (
            image_h * scale >= pixel_h * (1 - crop_tolerance)
            and image_w * scale >= pixel_w * (1 - crop_tolerance)
        ):
            fill_scale = max(pixel_h / image_h, pixel_w / image_w)
            crop_h = min(image_h, int(round(pixel_h / fill_scale)))
            crop_w = min(image_w, int(round(pixel_w / fill_scale)))
            top = (image_h - crop_h) // 2
            left = (image_w - crop_w) // 2
            source = source[..., top : top + crop_h, left : left + crop_w]
            new_h, new_w = pixel_h, pixel_w
        else:
            new_h = min(
                max(16, int(image_h * scale) // 16 * 16),
                max(16, pixel_h // 16 * 16),
            )
            new_w = min(
                max(16, int(image_w * scale) // 16 * 16),
                max(16, pixel_w // 16 * 16),
            )
            crop_h = min(image_h, max(1, int(round(new_h / scale))))
            crop_w = min(image_w, max(1, int(round(new_w / scale))))
            top = (image_h - crop_h) // 2
            left = (image_w - crop_w) // 2
            source = source[..., top : top + crop_h, left : left + crop_w]

        source = F.interpolate(
            source.float(), size=(new_h, new_w), mode="bicubic", antialias=True
        )
        latent = vae.encode(source.movedim(1, -1)[..., :3].clamp(0, 1))
        cache[key] = latent
        return latent

    scale = max(pixel_h / image_h, pixel_w / image_w)
    crop_h = min(image_h, int(round(pixel_h / scale)))
    crop_w = min(image_w, int(round(pixel_w / scale)))
    top = (image_h - crop_h) // 2
    left = (image_w - crop_w) // 2
    source = source[..., top : top + crop_h, left : left + crop_w]
    source = F.interpolate(
        source.float(), size=(pixel_h, pixel_w), mode="bicubic", antialias=True
    )
    latent = vae.encode(source.movedim(1, -1)[..., :3].clamp(0, 1))
    cache[key] = latent
    return latent


def _ref_attn_bias(
    boosts,
    boost_masks,
    text_length,
    source_lengths,
    target_length,
    mask_sizes,
    device,
    dtype,
):
    """Build the reference-fidelity attention bias."""
    offsets = [text_length]
    for source_length in source_lengths:
        offsets.append(offsets[-1] + source_length)
    target_start = offsets[-1]
    total_length = target_start + target_length
    bias = torch.zeros(
        1, 1, total_length, total_length, device=device, dtype=dtype
    )

    for index, boost in enumerate(boosts):
        if boost == 1.0:
            continue
        offset = offsets[index]
        source_length = source_lengths[index]
        boost_mask = boost_masks[index] if boost_masks is not None else None
        if (
            boost_mask is not None
            and mask_sizes is not None
        ):
            mask = boost_mask[:1]
            if mask.ndim == 2:
                mask = mask[None]
            mask = F.interpolate(
                mask[None].float(), size=mask_sizes[index], mode="area"
            )[0, 0]
            columns = offset + torch.nonzero(
                mask.reshape(-1) > 0.5, as_tuple=True
            )[0].to(device)
        else:
            columns = torch.arange(offset, offset + source_length, device=device)
        bias[:, :, target_start:, columns] = math.log(max(boost, 1e-4))
    return bias


def krea2_edit_forward(
    model,
    x,
    timesteps,
    context,
    source_latent,
    transformer_options,
    image_boosts=None,
    image_boost_masks=None,
    ref_native=False,
    pos_mode="anchor",
):
    """Run Krea 2 with clean source blocks prepended to the noisy target."""
    patch = model.patch

    temporal = x.ndim == 5
    if temporal:
        batch_5d, _channels_5d, frames_5d, height_5d, width_5d = x.shape
    x = _to_4d(x)
    batch_size, _channels, original_h, original_w = x.shape

    x = comfy.ldm.common_dit.pad_to_patch_size(
        x, (patch, patch), padding_mode="replicate"
    )
    height, width = x.shape[-2], x.shape[-1]
    grid_h, grid_w = height // patch, width // patch

    source_list = (
        source_latent
        if isinstance(source_latent, (list, tuple))
        else [source_latent]
    )
    sources = []
    for latent in source_list:
        source = _to_4d(latent).to(x.device, x.dtype)
        if source.shape[0] != batch_size:
            source = source[:1].expand(batch_size, *source.shape[1:])
        if not ref_native and source.shape[-2:] != (height, width):
            source = _fit_src(source, height, width).to(x.dtype)
        sources.append(
            comfy.ldm.common_dit.pad_to_patch_size(
                source, (patch, patch), padding_mode="replicate"
            )
        )
    source_grids = [
        (source.shape[-2] // patch, source.shape[-1] // patch)
        for source in sources
    ]

    context = model._unpack_context(context)
    target_image = model.first(
        rearrange(
            x,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=patch,
            pw=patch,
        )
    )
    source_images = [
        model.first(
            rearrange(
                source,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=patch,
                pw=patch,
            )
        )
        for source in sources
    ]

    timestep = model.tmlp(
        timestep_embedding(timesteps, model.tdim)
        .unsqueeze(1)
        .to(target_image.dtype)
    )
    timestep_vector = model.tproj(timestep)
    context = model.txtfusion(
        context, mask=None, transformer_options=transformer_options
    )
    context = model.txtmlp(context)

    text_length = context.shape[1]
    target_length = target_image.shape[1]
    source_length = sum(image.shape[1] for image in source_images)
    combined = torch.cat([context] + source_images + [target_image], dim=1)

    if pos_mode == "stride1" and ref_native:
        reference_ids = [
            _imgids_offset(
                batch_size,
                index + 1,
                source_h,
                source_w,
                grid_h,
                grid_w,
                combined.device,
            )
            for index, (source_h, source_w) in enumerate(source_grids)
        ]
    else:
        reference_ids = [
            _imgids(
                batch_size,
                index + 1,
                source_h,
                source_w,
                combined.device,
            )
            for index, (source_h, source_w) in enumerate(source_grids)
        ]
    positions = torch.cat(
        [
            torch.zeros(
                batch_size,
                text_length,
                3,
                device=combined.device,
                dtype=torch.float32,
            )
        ]
        + reference_ids
        + [_imgids(batch_size, 0, grid_h, grid_w, combined.device)],
        dim=1,
    )
    frequencies = model.pe_embedder(positions)

    attention_bias = None
    boosts = image_boosts or [1.0] * len(source_images)
    if any(boost != 1.0 for boost in boosts):
        attention_bias = _ref_attn_bias(
            boosts,
            image_boost_masks,
            text_length,
            [image.shape[1] for image in source_images],
            target_length,
            source_grids,
            combined.device,
            combined.dtype,
        )

    for block in model.blocks:
        combined = block(
            combined,
            timestep_vector,
            frequencies,
            attention_bias,
            transformer_options=transformer_options,
        )

    final = model.last(combined, timestep)
    output = final[
        :, text_length + source_length : text_length + source_length + target_length
    ]
    output = rearrange(
        output,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=grid_h,
        w=grid_w,
        ph=patch,
        pw=patch,
        c=model.channels,
    )
    output = output[:, :, :original_h, :original_w]
    if temporal:
        output = output.reshape(
            batch_5d, frames_5d, model.channels, height_5d, width_5d
        ).movedim(1, 2)
    return output
