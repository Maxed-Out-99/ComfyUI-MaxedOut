"""Prompt/conditioning nodes.

Registered nodes:
  Prompt With Guidance (Flux)  Prompt with Flux Guidance MXD
  KreaSeedVarianceMXD          Krea Seed Variance
  KreaLayerVarianceMXD         Krea Layer Variance
  QwenImageEditSingleMXD       Qwen Image Edit + Latent MXD   (needs comfy_api)
  QwenImageEditTripleMXD       Qwen Image Edit Prompt MXD (Triple)  (needs comfy_api)
"""
from __future__ import annotations
import torch, comfy, math, node_helpers, comfy.model_management, comfy.utils
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
try:
    from comfy_api.latest import io
    HAVE_COMFY_API = True
except Exception as _e:
    io = None
    HAVE_COMFY_API = False
    print(f"[ComfyUI-MaxedOut] comfy_api not available in prompts: {_e}")

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
# Krea 2 Turbo seed variance

_KREA_TAP_COUNT = 12
_KREA_TAP_DIM = 2560
_KREA_FEATURE_DIM = _KREA_TAP_COUNT * _KREA_TAP_DIM


def _early_conditioning(clean_conditioning, noisy_conditioning, end_percent):
    early = node_helpers.conditioning_set_values(
        noisy_conditioning,
        {"start_percent": 0.0, "end_percent": end_percent},
    )
    late = node_helpers.conditioning_set_values(
        clean_conditioning,
        {"start_percent": end_percent, "end_percent": 1.0},
    )
    return early + late


def _conditioning_with_tensor(conditioning_entry, tensor):
    if len(conditioning_entry) < 2:
        return conditioning_entry
    return (tensor, conditioning_entry[1].copy())


class KreaSeedVarianceMXD(ComfyNodeABC):
    DESCRIPTION = (
        "Adds seed-dependent Gaussian noise to a small fraction of Krea conditioning values during the first 20% "
        "of sampling. Amount 20 matches the Balanced Krea behavior of RBG Smart Seed Variance."
    )

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "conditioning": (IO.CONDITIONING,),
                "amount": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.5,
                        "tooltip": "20 matches the RBG Balanced preset for Krea 2. Higher values increase both the noise and the fraction changed.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Controls only the conditioning variation. Set this widget to randomize or increment between generations.",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply_variance"
    CATEGORY = "MXD/conditioning"

    def apply_variance(self, conditioning, amount, seed):
        if amount <= 0.0:
            return (conditioning,)

        # RBG couples density and amplitude. At amount 20 this is 1.9% of Krea values with Gaussian sigma 20.
        density = min(1.0, amount * 0.00095)
        noisy_conditioning = []

        for index, entry in enumerate(conditioning):
            if len(entry) < 2 or not isinstance(entry[0], torch.Tensor):
                noisy_conditioning.append(entry)
                continue

            source = entry[0]
            modified = source.clone()
            generator = torch.Generator(device=source.device)
            generator.manual_seed((int(seed) + index) % (1 << 64))

            flat = modified.reshape(-1)
            selected = torch.rand(flat.shape, device=flat.device, generator=generator) < density
            selected_count = int(selected.sum().item())
            if selected_count:
                noise = torch.randn(
                    selected_count,
                    device=flat.device,
                    dtype=flat.dtype,
                    generator=generator,
                )
                flat[selected] += noise * amount

            noisy_conditioning.append(_conditioning_with_tensor(entry, modified))

        return (_early_conditioning(conditioning, noisy_conditioning, 0.20),)


class KreaLayerVarianceMXD(ComfyNodeABC):
    DESCRIPTION = (
        "Applies a seeded, norm-preserving rotation to each active token in each of Krea 2's 12 text-encoder "
        "layers during the first 25% of sampling. This is experimental and designed specifically for Krea 2."
    )

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "conditioning": (IO.CONDITIONING,),
                "strength": (
                    "FLOAT",
                    {
                        "default": 6.0,
                        "min": 0.0,
                        "max": 45.0,
                        "step": 0.5,
                        "tooltip": "Rotation in degrees within each Krea text layer. Start at 6; raise gradually for more variation.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Controls only the layer-aware conditioning variation. Set this widget to randomize or increment between generations.",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply_variance"
    CATEGORY = "MXD/conditioning"

    @staticmethod
    def _active_tokens(metadata, tensor):
        attention_mask = metadata.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor):
            return None

        mask = attention_mask.to(device=tensor.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim != 2 or mask.shape[-1] != tensor.shape[1]:
            return None
        if mask.shape[0] == 1 and tensor.shape[0] != 1:
            mask = mask.expand(tensor.shape[0], -1)
        if mask.shape[0] != tensor.shape[0]:
            return None
        return mask

    def apply_variance(self, conditioning, strength, seed):
        if strength <= 0.0:
            return (conditioning,)

        angle = math.radians(min(float(strength), 45.0))
        cosine = math.cos(angle)
        sine = math.sin(angle)
        noisy_conditioning = []

        for index, entry in enumerate(conditioning):
            if len(entry) < 2 or not isinstance(entry[0], torch.Tensor):
                noisy_conditioning.append(entry)
                continue

            source = entry[0]
            if source.ndim != 3 or source.shape[-1] != _KREA_FEATURE_DIM:
                raise ValueError(
                    "Krea Layer Variance requires Krea 2 conditioning shaped [batch, tokens, 30720] "
                    "from a CLIP loader using type 'krea2'."
                )

            original_dtype = source.dtype
            layers = source.reshape(source.shape[0], source.shape[1], _KREA_TAP_COUNT, _KREA_TAP_DIM).float()
            generator = torch.Generator(device=source.device)
            generator.manual_seed((int(seed) + index) % (1 << 64))
            noise = torch.randn(layers.shape, device=source.device, dtype=torch.float32, generator=generator)

            source_norm_sq = torch.sum(layers * layers, dim=-1, keepdim=True)
            projection = torch.sum(noise * layers, dim=-1, keepdim=True) / source_norm_sq.clamp_min(1e-12)
            noise.sub_(projection * layers)

            source_norm = torch.sqrt(source_norm_sq)
            noise_norm = torch.linalg.vector_norm(noise, dim=-1, keepdim=True).clamp_min(1e-12)
            noise.mul_(source_norm / noise_norm)
            rotated = layers.mul(cosine).add_(noise, alpha=sine)

            active_tokens = self._active_tokens(entry[1], source)
            if active_tokens is not None:
                rotated = torch.where(active_tokens[:, :, None, None], rotated, layers)

            modified = rotated.reshape_as(source).to(dtype=original_dtype)
            noisy_conditioning.append(_conditioning_with_tensor(entry, modified))

        return (_early_conditioning(conditioning, noisy_conditioning, 0.25),)


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

NODE_CLASS_MAPPINGS = {
    "Prompt With Guidance (Flux)": PromptWithGuidance,
    "KreaSeedVarianceMXD": KreaSeedVarianceMXD,
    "KreaLayerVarianceMXD": KreaLayerVarianceMXD,
}

if HAVE_COMFY_API:
    NODE_CLASS_MAPPINGS.update({
        "QwenImageEditSingleMXD": QwenImageEditSingleMXD,
        "QwenImageEditTripleMXD": QwenImageEditTripleMXD,
    })

NODE_DISPLAY_NAME_MAPPINGS = {
    "Prompt With Guidance (Flux)": "Prompt with Flux Guidance MXD",
    "KreaSeedVarianceMXD": "Krea Seed Variance",
    "KreaLayerVarianceMXD": "Krea Layer Variance",
}

if HAVE_COMFY_API:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "QwenImageEditSingleMXD": "Qwen Image Edit + Latent MXD",
        "QwenImageEditTripleMXD": "Qwen Image Edit Prompt MXD (Triple)",
    })
